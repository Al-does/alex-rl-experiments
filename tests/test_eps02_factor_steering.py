from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip('jax')

from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import environment as v2_environment
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_prediction import Policy
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_pulse import exact_trace
from experiments.reward_factors_persistent_nuisance_jax_2026_11.shared import factor_process, model_spec
from experiments.reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11 import steering_factor as sf


def test_targets_move_only_the_named_coordinate():
    b = np.array([[.2, .5, .3], [.6, .1, .3]])
    point, mirror, flip, corner = sf.targets(b, np)
    np.testing.assert_allclose(point, [[.7, 0, .3], [0, .7, .3]])
    np.testing.assert_allclose(mirror, b[:, [1, 0, 2]])
    np.testing.assert_allclose(flip[:, 2], .7)
    np.testing.assert_allclose(flip[:, 0]/flip[:, 1], b[:, 0]/b[:, 1])
    np.testing.assert_allclose(corner, [[1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(np.asarray(sf.targets(jnp.asarray(b))), sf.targets(b, np), atol=1e-6)


def random_maps(kind, rng):
    maps = sf.empty_maps(kind)
    return {k: jnp.asarray(rng.normal(size=v.shape)*.1, jnp.float32) for k, v in maps.items()}


def test_displacements_leave_the_other_factor_fixed():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(7, 64)), jnp.float32)
    for factor in (0, 1):
        for index in range(len(sf.MODES)):
            mode = factor*len(sf.MODES)+index
            change = np.asarray(sf.marginal_displacement(x, random_maps('marginal', rng), 3, mode))
            other = change[:, 2:] if factor == 0 else change[:, :2]
            assert not other.any() and change.any()
            maps = random_maps('joint', rng)
            change = np.asarray(sf.joint_displacement(x, maps, 3, mode))
            joint = np.asarray(sf.full(sf.decoded(x, maps, 3))).reshape(-1, 3, 3)
            moved = joint+np.column_stack((change, -change.sum(-1))).reshape(-1, 3, 3)
            axis = 2 if factor == 1 else 1
            np.testing.assert_allclose(moved.sum(axis), joint.sum(axis), atol=1e-4)


def test_unsteered_rollout_matches_trainer_and_changes_nothing():
    process = factor_process('reward_both', environment)
    spec = replace(model_spec('transformer', process), context_len=32)
    params = models.init_params(spec, jax.random.PRNGKey(0))
    maps = random_maps('marginal', np.random.default_rng(1))
    intact = sf.rollout(spec, process, params, maps, jax.random.PRNGKey(5), 3, 40)
    assert not np.asarray(intact['changed']).any()
    steered = sf.rollout(spec, process, params, maps, jax.random.PRNGKey(5), 3, 40, 2, 6, 1.)
    assert np.asarray(steered['norms'])[..., 0].min() > 0
    result = sf.metrics(steered)
    assert len(result['occupancy']) == 2 and len(result['action_change']) == 2


def test_references_use_the_study_emission_matrix():
    leaky, exact = Policy(2, n=20, environment=environment), Policy(2, n=20, environment=v2_environment)
    assert exact.quotient and not leaky.quotient
    obs = np.zeros((3, 4, 6))
    obs[:, :, 0] = 1
    obs[:, :, 3] = 1
    b = np.tile([.08, .02, .9], (4, 1))
    # After an A token, the reward-state coordinate depends on delta only when state 3 can emit A.
    assert np.abs(exact_trace(b, obs, 2, 'mirror', environment=v2_environment)[1:, :, 0]).max() < 1e-12
    assert np.abs(exact_trace(b, obs, 2, 'mirror', environment=environment)[1:, :, 0]).max() > .01
    lumped = exact_trace(b, obs, 2, 'rho_flip', coarse=True, environment=environment)
    assert np.isfinite(lumped).all() and np.abs(lumped[1:, :, 1]).max() == 0
