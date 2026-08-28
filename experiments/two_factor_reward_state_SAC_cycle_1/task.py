"""Flat product actions and factor-selective reward-state occupancy."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.hmm import ActionDecision, HMMModel, TransitionEvent

from experiments.two_factor_reward_state_SAC_cycle_1.process import (
    FACTOR_CARDINALITY,
    TRANSITION_MATRIX,
    decode_joint_indices,
)


REWARD_STATE = 2
CONDITIONS = ("reward_both", "reward_factor_1", "reward_factor_2")
# This deliberately follows the requested presentation order rather than
# lexicographic product order. The policy receives only the flat index.
ACTION_PAIRS = (
    (0, 0),
    (1, 1),
    (1, 2),
    (0, 1),
    (0, 2),
    (2, 1),
    (2, 2),
    (1, 0),
    (2, 0),
)
ACTION_LABELS = (
    "noop_noop",
    "factor_1_forward_factor_2_forward",
    "factor_1_forward_factor_2_backward",
    "factor_1_noop_factor_2_forward",
    "factor_1_noop_factor_2_backward",
    "factor_1_backward_factor_2_forward",
    "factor_1_backward_factor_2_backward",
    "factor_1_forward_factor_2_noop",
    "factor_1_backward_factor_2_noop",
)
N_ACTIONS = len(ACTION_PAIRS)


def shifted_transition(shift: int) -> np.ndarray:
    """Map old destination ``j`` to ``(j + shift) mod 3`` in every row."""

    if shift not in (0, 1, 2):
        raise ValueError("shift must be 0, 1, or 2")
    return np.roll(TRANSITION_MATRIX, shift=shift, axis=1)


def joint_transition(action: int) -> np.ndarray:
    """Return the independent joint transition selected by one flat action."""

    index = int(action)
    if not 0 <= index < N_ACTIONS:
        raise ValueError("action must lie in [0, 9)")
    first, second = ACTION_PAIRS[index]
    return np.kron(shifted_transition(first), shifted_transition(second))


class TwoFactorShiftTask:
    """Control both factors while rewarding a selected subset in state 2."""

    requires_belief = False

    def __init__(self, *, model: HMMModel, condition: str) -> None:
        if model.n_states != FACTOR_CARDINALITY**2:
            raise ValueError("two-factor shift control requires nine joint states")
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}")
        self.condition = condition
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.action_observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(N_ACTIONS,),
            dtype=np.float32,
        )
        self._transition_matrices = tuple(joint_transition(i) for i in range(N_ACTIONS))
        for transition in self._transition_matrices:
            transition.setflags(write=False)

    def reset(self) -> None:
        pass

    def transition_matrix_for_action(self, action: int) -> np.ndarray:
        index = int(action)
        if not self.action_space.contains(index):
            raise ValueError(f"action {index} is outside the action space")
        return self._transition_matrices[index]

    def resolve_action(
        self,
        action: int,
        state: int,
        model: HMMModel,
    ) -> ActionDecision:
        del state, model
        executed = int(action)
        return ActionDecision(
            requested_action=executed,
            executed_action=executed,
            transition_matrix=self.transition_matrix_for_action(executed),
        )

    def reward(
        self,
        event: TransitionEvent,
        decision: ActionDecision,
    ) -> tuple[float, dict[str, float]]:
        del decision
        first, second = decode_joint_indices(event.state_before)
        first_reward = float(first == REWARD_STATE)
        second_reward = float(second == REWARD_STATE)
        if self.condition == "reward_both":
            reward = first_reward + second_reward
        elif self.condition == "reward_factor_1":
            reward = first_reward
        else:
            reward = second_reward
        return reward, {
            "factor_1_occupancy_reward": first_reward,
            "factor_2_occupancy_reward": second_reward,
        }

    def encode_action(self, executed_action: int) -> np.ndarray:
        encoded = np.zeros(N_ACTIONS, dtype=np.float32)
        encoded[int(executed_action)] = 1.0
        return encoded
