import ast
import inspect
import subprocess

from . import analyze
from . import steer_core as core, steer_prediction as prediction, steer_moving as moving
from . import steer_moving_base as base
from .steer_storage import compact
from .steering_run import moving_runs
from .steering_report import aggregate, ladder_rows
from .vendor_steering import LADDER, MOVING, PREFIX

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('mode', core.MODES)
def test_ladder_targets_normal_empty_error(mode):
    b = np.array([[.2, .5, .3], [.4, .1, .5], [0., 0., 1.], [-.2, -.3, 1.5]])
    actual = prediction.target(b, mode)
    np.testing.assert_allclose(core.target(jnp.asarray(b), core.MODES.index(mode)), actual, atol=1e-7)
    np.testing.assert_allclose(actual.sum(-1), 1)
    if mode in ('point', 'mirror'):
        np.testing.assert_array_equal(actual[:, 2], b[:, 2])
    assert prediction.target(np.empty((0, 3)), mode).shape == (0, 3)
    with pytest.raises(ValueError):
        prediction.target(np.array([[np.nan, 0, 1]]), mode)


@pytest.mark.parametrize('n_obs,n_actions', [(6, 3), (18, 9)])
def test_forward_zero_write_matches_frozen_actor(n_obs, n_actions):
    from experiments.belief_quotient_history_jax_2026_11 import models
    spec = models.ModelSpec('transformer', n_obs, n_actions, 64, 4)
    params = models.init_params(spec, jax.random.PRNGKey(7))
    observations = jax.random.normal(jax.random.PRNGKey(10), (35, 2, n_obs))
    maps = core.empty_maps()
    original = jax.jit(lambda c, o, t: models.step(spec, params, c, o, t))
    if n_actions == 3:
        adapted = jax.jit(lambda c, o, t: core.instrumented_step(spec, params, c, o, t, maps, 1, 1, 0.)[:2])
    else:
        adapted = jax.jit(lambda c, o, t: (base.forward(spec, params, c, o, t)[0], base.forward(spec, params, c, o, t)[2]))
    cache = models.init_carry(spec, 2)
    for t, obs in enumerate(observations):
        logits, _, _, expected = original(cache, obs, t)
        actual, carry = adapted(cache, obs, t)
        np.testing.assert_allclose(actual, logits, atol=2e-6, rtol=2e-6)
        for k in cache:
            np.testing.assert_allclose(carry[k], expected[k], atol=2e-6, rtol=2e-6)
        cache = expected


def test_moving_only_schedule_qmap_and_preserves_unsteered_factor():
    assert moving.REWARDS == ('moving',)
    assert moving.STEERED_FACTORS == {'moving': (False, True)}
    np.testing.assert_array_equal(moving.reward_components('moving', 0)[0], moving.reward_components('moving', 256)[0])
    np.testing.assert_array_equal(moving.reward_components('moving', 0)[1], 1-moving.reward_components('moving', 256)[1])
    q = moving.q_tables('moving', 'discounted')
    early, late = base.ACTION_TABLE[q[10].argmax(1)][:, 1], base.ACTION_TABLE[q[266].argmax(1)][:, 1]
    assert early[6] == 1 and late[6] == 2
    with pytest.raises(ValueError):
        moving.q_tables('moving', 'invalid')
    joint = np.random.default_rng(0).dirichlet(np.ones(9), size=4)
    anchors = jnp.array([[2, 0, 1], [2, 0, 1]])
    target = moving.belief_target(jnp.asarray(joint[:, :8]), 'joint', jnp.asarray(q[0]), anchors, (False, True))
    full = np.column_stack((target, 1-np.asarray(target).sum(1))).reshape(-1, 3, 3)
    np.testing.assert_allclose(full.sum(2), joint.reshape(-1, 3, 3).sum(2), atol=1e-6)
    factors = base.factor_beliefs(joint)
    decoded = factors[:, :, :2].reshape(-1, 4)
    target = moving.belief_target(decoded, 'marginal', jnp.asarray(q[0]), anchors, (False, True))
    np.testing.assert_allclose(base.decoded_factor_beliefs(target, 'marginal')[:, 0], factors[:, 0], atol=1e-6)
    for name, width in (('marginal', 4), ('joint', 8)):
        empty = moving.belief_target(jnp.empty((0, width)), name, jnp.asarray(q[0]), anchors, (False, True))
        assert empty.shape == (0, width)
    assert len(moving_runs()) == 10


def test_moving_inventory_rejects_empty_and_incomplete(tmp_path, monkeypatch):
    from . import steering_run
    original = steering_run.REPO
    monkeypatch.setattr(steering_run, 'REPO', tmp_path)
    with pytest.raises(ValueError, match='expected exactly five'):
        moving_runs()
    monkeypatch.setattr(steering_run, 'REPO', original)
    original_read = steering_run.read

    def incomplete(path):
        result = original_read(path)
        result['status'] = 'running'
        return result

    monkeypatch.setattr(steering_run, 'read', incomplete)
    with pytest.raises(ValueError, match='incomplete run'):
        moving_runs()


def test_compaction_and_aggregation_normal_empty_error():
    report = dict(seed=42, variant=2, confirmation=[dict(condition='intact', site=-1, strength=0.,
        metrics=dict(occupancy=.4, episode_occupancy=[.3, .5]))], pulse={'conditions': {}}, tuning={'large': [0]*1000})
    result = compact(report, {'sha256': 'test'})
    assert 'episode_occupancy' not in result['confirmation'][0]['metrics']
    assert 'tuning' not in result
    assert aggregate([report], ladder_rows)['means']['intact@-1']['occupancy'] == .4
    report['confirmation'][0]['metrics']['action_given_state'] = [[None]*3]*3
    assert 'action_given_state' not in aggregate([report], ladder_rows)['means']['intact@-1']
    with pytest.raises(ValueError):
        aggregate([], ladder_rows)
    with pytest.raises(ValueError):
        aggregate([report, report], ladder_rows)
    with pytest.raises(ValueError):
        ladder_rows({'confirmation': []})


def test_pinned_functions_and_ladder_numerical_driver_unchanged():
    from . import steer_pulse, steer_ladder, steer_reference
    cases = [(core, LADDER, PREFIX+'steering_core.py'),
             (prediction, LADDER, PREFIX+'steering_prediction.py'),
             (steer_pulse, LADDER, PREFIX+'steering_pulse.py'),
             (base, MOVING, 'experiments/two_factor_moved_reward_steering_v2_jax_2026_11/steering.py'),
             (moving, MOVING, 'experiments/two_factor_coupled_moving_reward_steering_v2_jax_2026_11/steering.py'),
             (steer_reference, LADDER, 'experiments/belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11/steering_reference.py')]
    for module, rev, path in cases:
        original = ast.parse(subprocess.check_output(['git', '-C', str(analyze.REPO), 'show', rev+':'+path], text=True))
        lookup = {n.name: n for n in original.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for node in ast.parse(inspect.getsource(module)).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert ast.dump(node) == ast.dump(lookup[node.name]), node.name
    original = subprocess.check_output(['git', '-C', str(analyze.REPO), 'show', LADDER+':'+PREFIX+'belief_steering.py'], text=True)
    numerical = original[original.index('    rows, length ='):original.index('    harness_root =')]
    adapted = inspect.getsource(steer_ladder.run)
    actual = adapted[adapted.index('    rows, length ='):adapted.index('    report = dict')]
    assert numerical == actual
