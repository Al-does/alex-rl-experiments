import numpy as np

from envs.hmm import HMMEnv
from experiments.mess3_reward_state_action_symmetry_cycle_5.task import (
    NEGATIVE_ACTION,
    NOOP_ACTION,
    POSITIVE_ACTION,
)
from experiments.two_factor_moved_reward_steering_2026_11.action_map import (
    best_factor_action,
    best_joint_action,
    next_reward_table,
)
from experiments.two_factor_moved_reward_steering_2026_11.steering import (
    BeliefMap,
    decoded_factor_beliefs,
    steering_delta,
)
from experiments.two_factor_moved_reward_steering_2026_11.task import (
    MOVED_REWARD_STATES,
    SOURCE_REWARD_STATES,
    environment_config,
)
from experiments.two_factor_reward_state_SAC_cycle_2.process import decode_joint_indices
from experiments.two_factor_reward_state_SAC_cycle_2.task import ACTION_PAIRS


def test_source_map_recovers_full_state_quotient_actions():
    actions = best_factor_action(np.eye(3), SOURCE_REWARD_STATES[1])
    assert actions.tolist() == [POSITIVE_ACTION, NEGATIVE_ACTION, NOOP_ACTION]


def test_moved_map_point_masses_and_noop_region():
    actions = best_factor_action(np.eye(3), MOVED_REWARD_STATES[1])
    assert actions[0] == NEGATIVE_ACTION
    assert actions[1] == POSITIVE_ACTION
    assert actions[2] in (POSITIVE_ACTION, NEGATIVE_ACTION)
    assert best_factor_action(np.array([0.3, 0.7, 0.0]), 0) == NOOP_ACTION
    table = next_reward_table(0)
    np.testing.assert_allclose(table[2, 1], table[2, 2])


def test_joint_map_pairs_factor_maps():
    beliefs = np.array([[[0, 0, 1.0], [1.0, 0, 0]]])
    joint = best_joint_action(beliefs, MOVED_REWARD_STATES)
    assert ACTION_PAIRS[int(joint[0])] == (NOOP_ACTION, NEGATIVE_ACTION)


def test_environment_pays_factor_2_for_moved_state():
    env = HMMEnv({**environment_config(), "diagnostics": {"state": True, "rewards": True}})
    _, info = env.reset(seed=3)
    for step in range(200):
        state = decode_joint_indices(int(info["state_current"]))
        _, reward, _, _, info = env.step(step % 9)
        components = info["reward_components"]
        assert components["factor_1_occupancy_reward"] == float(state[0] == 2)
        assert components["factor_2_occupancy_reward"] == float(state[1] == 0)
        assert reward == sum(components.values())


def test_joint_target_keeps_factor_1_and_anchors_factor_2():
    joint = np.array([
        [0.12, 0.02, 0.06, 0.30, 0.05, 0.15, 0.18, 0.03, 0.09],
        [0.02, 0.20, 0.03, 0.05, 0.40, 0.05, 0.03, 0.20, 0.02],
    ])
    mapping = BeliefMap(np.eye(8), np.zeros(8), np.eye(8), "joint")
    shift, anchor = steering_delta(joint[:, :8], mapping, reward_state=0)
    steered = decoded_factor_beliefs(joint[:, :8] + shift, "joint")
    original = decoded_factor_beliefs(joint[:, :8], "joint")
    np.testing.assert_allclose(steered[:, 0], original[:, 0])
    np.testing.assert_allclose(steered[:, 1], np.eye(3)[anchor], atol=1e-12)
    # Factor 2 mostly in state 0 -> want Tilt Y -> source policy's Tilt Y state 1.
    assert anchor[0] == 1
    # Factor 2 mostly in state 1 -> want Tilt X -> source policy's Tilt X state 0.
    assert anchor[1] == 0


def test_marginal_target_replaces_only_factor_2():
    decoded = np.array([[0.2, 0.5, 0.7, 0.2]])
    mapping = BeliefMap(np.eye(4), np.zeros(4), np.eye(4), "marginal")
    shift, anchor = steering_delta(decoded, mapping, reward_state=0)
    np.testing.assert_allclose(shift[0, :2], 0)
    np.testing.assert_allclose((decoded + shift)[0, 2:], np.eye(3)[anchor[0]][:2])
