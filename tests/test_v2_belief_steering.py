import jax
import jax.numpy as jnp
import numpy as np
import pytest
import json
from unittest.mock import Mock, patch

from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import steering_core as core
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_prediction import Policy, target
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_pulse import exact_trace
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_aggregate import aggregate
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.belief_steering import restore, persist, STUDIES


@pytest.mark.parametrize('mode', core.MODES)
def test_targets(mode):
    b = np.array([[.2, .5, .3], [.4, .1, .5], [0., 0., 1.], [-.2, -.3, 1.5]])
    actual = target(b, mode)
    np.testing.assert_allclose(core.target(jnp.asarray(b), core.MODES.index(mode)), actual, atol=1e-7)
    np.testing.assert_allclose(actual.sum(-1), 1)
    if mode in ('point', 'mirror'):
        np.testing.assert_array_equal(actual[:, 2], b[:, 2])
    if mode == 'mirror':
        np.testing.assert_allclose(target(actual, mode), b)
    if mode.startswith('rho_'):
        np.testing.assert_allclose(actual[-1, :2], (1-actual[-1, 2])/2)
    assert target(np.empty((0, 3)), mode).shape == (0, 3)
    with pytest.raises(ValueError):
        target(np.array([[np.nan, 0, 1]]), mode)


def fixture():
    spec = models.ModelSpec('transformer', 6, 3, 64, 4)
    params = models.init_params(spec, jax.random.PRNGKey(7))
    maps = core.empty_maps()
    maps['decoder'] = jax.random.normal(jax.random.PRNGKey(8), (5, 64, 2))*.01
    maps['encoder'] = jax.random.normal(jax.random.PRNGKey(9), (5, 2, 64))
    observations = jax.random.normal(jax.random.PRNGKey(10), (40, 2, 6))
    return spec, params, maps, observations


def test_zero_intervention_parity():
    spec, params, maps, observations = fixture()
    cache = models.init_carry(spec, 2)
    a = jax.jit(lambda c, o, t: models.step(spec, params, c, o, t))
    b = jax.jit(lambda c, o, t: core.instrumented_step(spec, params, c, o, t, maps, 1, 1, 0.))
    for t, obs in enumerate(observations):
        logits, _, _, expected_cache = a(cache, obs, t)
        actual, actual_cache, _, _ = b(cache, obs, t)
        np.testing.assert_allclose(actual, logits, atol=2e-6, rtol=2e-6)
        np.testing.assert_array_equal(np.argmax(actual, -1), np.argmax(logits, -1))
        for key in cache:
            np.testing.assert_allclose(actual_cache[key], expected_cache[key], atol=2e-6, rtol=2e-6)
        cache = expected_cache


def test_zero_pulse_and_immutable_cache():
    spec, params, maps, observations = fixture()
    cache = models.init_carry(spec, 2)
    original = jax.tree_util.tree_map(np.asarray, cache)
    _, intact = core.pulse_branch(spec, params, maps, cache, observations, 0, -1, 1, 0.)
    _, branch = core.pulse_branch(spec, params, maps, cache, observations, 0, 1, 1, 0.)
    _, changed = core.pulse_branch(spec, params, maps, cache, observations, 0, 1, 1, 1.)
    for left, right in zip(intact, branch):
        np.testing.assert_array_equal(left, right)
    assert np.max(np.abs(np.asarray(changed[0])-np.asarray(intact[0]))) > 1e-5
    for key in cache:
        np.testing.assert_array_equal(cache[key], original[key])


def test_zero_emissions_and_quotient():
    policy = Policy(2, n=10, max_iterations=4000)
    assert policy.convergence['converged']
    assert np.isfinite(policy.actions).all()
    b = np.array([[.8, .1, .1], [.1, .3, .6]])
    observations = np.tile(np.array([[[0, 0, 1, 0, 1, 0]]]), (12, 2, 1))
    full = exact_trace(b, observations, 2, 'rho_flip')
    coarse = exact_trace(b, observations, 2, 'rho_flip', coarse=True)
    np.testing.assert_allclose(full[..., 0], coarse[..., 0], atol=1e-12)
    null = exact_trace(b, observations, 2, 'mirror')
    np.testing.assert_allclose(null[..., 0], 0, atol=1e-12)
    with pytest.raises(ValueError):
        aggregate([])


def test_restore_rejects_bad_hash(tmp_path):
    parent = dict(b2_prefix='test', b2_bucket='bucket', checkpoint_file='checkpoints/final.npz')
    client = Mock()
    client.get_object.return_value = {'Body': Mock(read=lambda: json.dumps({'files': [dict(relative_path=parent['checkpoint_file'], sha256='incorrect')]}).encode())}
    storage = Mock(s3_client=lambda: client)
    (tmp_path/'final.npz').write_bytes(b'not a trusted checkpoint')
    with patch('experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.belief_steering.locate_parent_run', return_value=parent), patch('harness.storage.b2.B2StorageConfig.from_env', return_value=storage):
        with pytest.raises(ValueError, match='SHA256 mismatch'):
            restore(tmp_path, STUDIES[0], tmp_path)


def test_aggregate_complete_and_duplicate(tmp_path):
    paths = []
    for i, (study, variant) in enumerate((s, v) for s in STUDIES for v in (1, 2, 3)):
        report = dict(study=study, variant=variant, smoke=False, protocol={}, checkpoint={'sha256': str(i)}, provenance={},
            confirmation=[dict(condition='intact', site=-1, strength=0., metrics={'episode_occupancy': [.1, .3]})], pulse={'conditions': {}})
        path = tmp_path/f'{i}.json'
        path.write_text(json.dumps(report))
        paths.append(path)
    result = aggregate(paths)
    assert len(result['table']) == 6
    assert all(x['occupancy_difference'] == 0 for x in result['epsilon_minus_no_epsilon'])
    with pytest.raises(ValueError):
        aggregate(paths[:-1]+[paths[0]])


def test_remote_durability_contract(tmp_path):
    raw = tmp_path/'artifacts'
    raw.mkdir()
    report_path = tmp_path/'report.json'
    report_path.write_text('{}')
    report_path.with_suffix('.md').write_text('# report')
    report = dict(study=STUDIES[0], variant=1, provenance={}, checkpoint={})
    remote = dict(status='completed', base_uri='s3://test/analysis/')
    with patch('experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.belief_steering.upload_run_artifacts', return_value=remote) as upload:
        persist(tmp_path, raw, report_path, report)
    assert upload.call_count == 1
    manifest = json.loads((tmp_path/'results/belief_steering_v2_seed42_run/run_manifest.json').read_text())
    assert manifest['status'] == 'completed'
    assert manifest['remote_artifacts']['status'] == 'completed'
    assert (raw/'report.json').exists()
