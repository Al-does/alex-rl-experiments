import jax.numpy as jnp
import numpy as np
import pytest

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
from experiments.reward_factors_replay_edit_2026_11 import replay, transfer


def example():
    rng = np.random.default_rng(42)
    tokens = np.eye(9)[rng.integers(9, size=(4, 24))]
    actions = np.eye(9)[rng.integers(9, size=(4, 24))]
    actions[:, 0] = 0
    return np.concatenate((tokens, actions), -1)


@pytest.mark.parametrize("f", (0, 1))
def test_exact_factor_replay(f):
    obs = example()
    b = replay.factors(environment, obs)
    edited = replay.edit(obs, f, [5, 11], np.random.default_rng(12))
    np.testing.assert_array_equal(edited[..., 9:], obs[..., 9:])
    alt = replay.factors(environment, edited)
    np.testing.assert_array_equal(alt[..., 1 - f, :], b[..., 1 - f, :])
    assert np.max(np.abs(alt[..., f, :] - b[..., f, :])) > .1
    mirrored = replay.mirror_inputs(obs, 1 << f)
    np.testing.assert_array_equal(replay.mirror_inputs(mirrored, 1 << f), obs)
    mm = replay.factors(environment, mirrored)
    np.testing.assert_array_equal(mm[..., 1 - f, :], b[..., 1 - f, :])
    np.testing.assert_allclose(mm[..., f, :], b[..., f, replay.TOKEN_SWAP])


def test_factor_marginals_and_fidelity():
    z = np.log(np.array([[.41, .12, .1, .09, .08, .07, .06, .05, .02]]))
    ref = z[:, replay.PERM[2]]
    old = replay.fold_actions(np.exp(replay.ladder.log_probs(z)))
    new = replay.fold_actions(np.exp(replay.ladder.log_probs(ref)))
    np.testing.assert_allclose(old[:, 0], new[:, 0])
    np.testing.assert_allclose(old[:, 1, replay.ACTION_SWAP], new[:, 1])
    assert replay.marginal_fidelity(z, ref, ref, np.array([.6]))["high"]["fidelity"] == [None, 1.]
    assert replay.marginal_fidelity(z, ref, z, np.array([.1]))["low"]["fidelity"] == [None, 0.]


def test_reachable_opposite_and_empirical_targets():
    b = jnp.array([[[.2, .2, .6], [.75, .1, .15]], [[.2, .2, .6], [.2, .2, .6]]])
    actions = jnp.array([[0, 2], [0, 2]])
    learning_map = jnp.array([[1, 2, 0], [1, 2, 0]])
    prototype = jnp.array([.1, .7, .2])
    prototypes = jnp.broadcast_to(prototype, (2, 3, 3, 3))
    result, frame, support = transfer.empirical_target(
        b, actions, learning_map, prototypes, jnp.ones((2, 3), bool), jnp.array([False, True]))
    np.testing.assert_array_equal(result[:, 0], b[:, 0])
    np.testing.assert_array_equal(result[0, 1], b[0, 1, replay.TOKEN_SWAP])
    np.testing.assert_array_equal(result[1, 1], prototype)
    np.testing.assert_array_equal(frame, [2, 0])
    np.testing.assert_array_equal(support, [True, False])
