import jax.numpy as jnp
import numpy as np

from experiments.two_factor_coupled_moving_reward_steering_v2_jax_2026_11 import steering as s
from experiments.two_factor_moved_reward_steering_v2_jax_2026_11.steering import ACTION_TABLE

NOOP, X, Y = 0, 1, 2


def _state(first, second):
    return 3 * first + second


def test_reward_schedules():
    xor = s.reward_components("xor", 0)[0]
    assert xor[_state(2, 0)] == xor[_state(0, 2)] == 1
    assert xor[_state(2, 2)] == xor[_state(0, 1)] == 0
    even, odd = s.reward_components("moving", 0), s.reward_components("moving", s.PERIOD)
    np.testing.assert_array_equal(even[0], odd[0])
    assert even[1][_state(0, 2)] == 1 and odd[1][_state(0, 2)] == 0
    assert odd[1][_state(0, 0)] == 1


def test_xor_known_state_policy_couples_factors():
    policy = s.q_tables("xor", "discounted")[0].argmax(1)
    factor_2 = ACTION_TABLE[policy][:, 1]
    # factor 2 in state 1: pushed into state 3 when factor 1 is out, kept out when factor 1 is in
    assert factor_2[_state(0, 0)] == X
    assert factor_2[_state(2, 0)] == Y


def test_moving_map_flips_with_phase():
    q = s.q_tables("moving", "discounted")
    early, late = ACTION_TABLE[q[10].argmax(1)][:, 1], ACTION_TABLE[q[s.PERIOD + 10].argmax(1)][:, 1]
    assert early[_state(2, 0)] == X and late[_state(2, 0)] == Y
    assert early[_state(2, 2)] == NOOP and late[_state(2, 2)] != NOOP


def test_belief_target_respects_steered_factors():
    rng = np.random.default_rng(0)
    joint = rng.dirichlet(np.ones(9), size=4)
    decoded = jnp.asarray(joint[:, :8], jnp.float32)
    anchors = jnp.asarray([[2, 0, 1], [2, 0, 1]])
    q = jnp.asarray(s.q_tables("moving", "discounted")[0], jnp.float32)
    target = s.belief_target(decoded, "joint", q, anchors, (False, True))
    full = np.column_stack((np.asarray(target), 1 - np.asarray(target).sum(1))).reshape(-1, 3, 3)
    np.testing.assert_allclose(full.sum(2), joint.reshape(-1, 3, 3).sum(2), atol=1e-6)
    assert np.allclose(full.sum(1).max(1), 1, atol=1e-6)
    both = s.belief_target(decoded, "marginal", q, anchors, (True, True))
    assert set(np.round(np.asarray(both), 6).ravel().tolist()) <= {0.0, 1.0}
