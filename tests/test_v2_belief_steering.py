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
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_storage import compact, pulse_summary, require_artifacts, upload_full, write_summary


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
    full_policy = Policy(3, n=10, max_iterations=4000)
    assert full_policy.convergence['converged']
    b = np.array([[.8, .1, .1], [.1, .3, .6]])
    observations = np.tile(np.array([[[0, 0, 1, 0, 1, 0]]]), (12, 2, 1))
    full = exact_trace(b, observations, 2, 'rho_flip')
    coarse = exact_trace(b, observations, 2, 'rho_flip', coarse=True)
    np.testing.assert_allclose(full[..., 0], coarse[..., 0], atol=1e-12)
    null = exact_trace(b, observations, 2, 'mirror')
    np.testing.assert_allclose(null[..., 0], 0, atol=1e-12)
    rng = np.random.default_rng(11)
    beliefs = rng.dirichlet(np.ones(3), 1000)
    for mode in ('point', 'mirror'):
        for strength in core.LAMBDAS:
            installed = (1-strength)*beliefs+strength*target(beliefs, mode)
            np.testing.assert_array_equal(policy(installed), policy(beliefs))
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
            confirmation=[dict(condition='intact', site=-1, strength=0., metrics={'episode_occupancy': [.1, .3]})], pulse={'conditions': {}},
            probes={'exclusion_8': [dict(b1=.8, b2=.9, rho=.99, delta=.7)]}, norm_matching={'point_site3': {'effective_lambda': 2}})
        path = tmp_path/f'{i}.json'
        path.write_text(json.dumps(report))
        paths.append(path)
    result = aggregate(paths)
    assert len(result['table']) == 6
    assert len(result['probes']) == 6
    assert len(result['norm_matching']) == 6
    assert all(x['occupancy_difference'] == 0 for x in result['epsilon_minus_no_epsilon'])
    reference = {'sha256': 'full-payload-hash', 'key': 'test/full.json'}
    output = tmp_path/'results'/'summary.json'
    summary = write_summary(output, result, reference)
    assert summary == compact(aggregate(paths), reference)
    assert 'episode_occupancy' not in summary['table'][0]['metrics']
    assert len(output.read_text().splitlines()) < 3000
    with pytest.raises(ValueError):
        aggregate(paths[:-1]+[paths[0]])
    paths[0].write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='full artifacts'):
        aggregate(paths)


def test_compact_agent_and_empty_pulse(tmp_path):
    trace = dict(absolute=[[float(k), 0.] for k in range(129)], signed=[[0., 0.]]*129,
                 decay={'rho': {'multiplier': .9}, 'delta': {'multiplier': None}})
    report = dict(study=STUDIES[0], variant=1, selected={'mirror': .5}, probes={}, norm_matching={},
        confirmation=[dict(condition='curve_mirror', strength=lam, site=3,
            metrics={'occupancy': .4, 'episode_occupancy': [.3, .5]}) for lam in (0., .5, 1.)],
        tuning={'large': [0]*10000}, pulse=dict(horizon=128, conditions={'mirror_block1':
            dict(measured=trace, full_filter=trace, tv=[0.]*129, action_disagreement=[0.]*129)}))
    result = compact(report, {'sha256': 'x'})
    assert len(result['confirmation']) == 1 and 'tuning' not in result
    assert 'episode_occupancy' not in result['confirmation'][0]['metrics']
    measured = result['pulse']['conditions']['mirror_block1']['measured']
    assert measured['last_nonzero_lag'] == 128 and measured['absolute_at_lag']['128'] == [128., 0.]
    assert 'signed' not in measured and 'absolute' not in measured
    assert pulse_summary({'measured': dict(absolute=[], decay={})})['measured']['last_nonzero_lag'] is None
    with pytest.raises(ValueError, match='compact summary'):
        compact(result, {})
    with pytest.raises(ValueError, match='unrecognized'):
        compact({}, {})
    with pytest.raises(ValueError, match='artifacts'):
        require_artifacts(tmp_path/'results/full.json')
    require_artifacts(tmp_path/'artifacts/full.json')


def test_full_payload_upload_hash_verification(tmp_path):
    path = tmp_path/'full.json'
    path.write_text('{"full": true}')
    client = Mock()
    client.get_object.return_value = {'Body': Mock(read=path.read_bytes)}
    storage = Mock(bucket='test-bucket', s3_client=lambda: client)
    with patch('harness.storage.b2.B2StorageConfig.from_env', return_value=storage):
        reference = upload_full(path, 'analysis/full.json')
        assert reference['uri'] == 's3://test-bucket/analysis/full.json'
        client.upload_file.assert_called_once()
        client.get_object.return_value = {'Body': Mock(read=lambda: b'wrong bytes')}
        with pytest.raises(ValueError, match='B2 SHA256 mismatch'):
            upload_full(path, 'analysis/full.json')


def test_compact_predictions():
    point = dict(occupancy=.5, occupancy_se=.02, action_fractions=[1., 0., 0.])
    full = dict(headline={'v3': {'candidate': 'g0.995_n300'}}, results={
        'v3_g0.995_n300': dict(convergence={'converged': True}, curves={'mirror': {'1.0': point}}),
        'v3_g0.995_n100': dict(convergence={'converged': True}, curves={'mirror': {'1.0': point}})})
    summary = compact(full, {'key': 'predictions/full.json'})
    assert len(summary['convergence']) == 2
    assert list(summary['headline_curves']) == ['v3_g0.995_n300']
    assert summary['headline_curves']['v3_g0.995_n300']['mirror']['1.0'] == dict(occupancy=.5, occupancy_se=.02)
    assert compact(dict(headline={}, results={}), {})['headline_curves'] == {}


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
    manifest = json.loads((tmp_path/'results/belief_steering_v2_seed42_stable_run/run_manifest.json').read_text())
    assert manifest['status'] == 'completed'
    assert manifest['remote_artifacts']['status'] == 'completed'
    assert (raw/'report.json').exists()


def test_affine_discards_float32_null_directions():
    rng = np.random.default_rng(13)
    z = rng.normal(size=(2000, 3))
    x = np.column_stack((z, z.sum(-1)+rng.normal(scale=1e-8, size=len(z))))
    y = z @ np.array([[.2, .1], [.1, -.2], [.3, .4]])+rng.normal(scale=.001, size=(len(z), 2))
    weight, bias = core.affine(x, y)
    assert np.linalg.norm(weight) < 2
    np.testing.assert_allclose(x.astype(np.float32) @ weight.astype(np.float32)+bias, y, atol=.005)


def test_decoder_requests_full_precision(monkeypatch):
    original = jnp.matmul
    calls = []

    def matmul(*args, **kwargs):
        calls.append(kwargs.get('precision'))
        return original(*args, **kwargs)

    monkeypatch.setattr(jnp, 'matmul', matmul)
    belief = core.decode(jnp.ones((2, 64)), core.empty_maps(), 4)
    np.testing.assert_allclose(belief, [[0, 0, 1], [0, 0, 1]])
    assert calls == [jax.lax.Precision.HIGHEST]
