"""Independent product actions built from action-symmetry variant 3."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.hmm import ActionDecision, HMMModel, TransitionEvent
from experiments.mess3_reward_state_action_symmetry_cycle_5.design import EFFECT_SIZE
from experiments.mess3_reward_state_action_symmetry_cycle_5.task import (
    NEGATIVE_ACTION,
    NOOP_ACTION,
    POSITIVE_ACTION,
)
from experiments.two_factor_reward_state_SAC_cycle_2.process import (
    FACTOR_CARDINALITY,
    TRANSITION_MATRIX,
    decode_joint_indices,
)


REWARD_STATE = 2
CONDITIONS = ("reward_both", "reward_factor_1")
ACTION_PAIRS = (
    (NOOP_ACTION, NOOP_ACTION),
    (POSITIVE_ACTION, POSITIVE_ACTION),
    (POSITIVE_ACTION, NEGATIVE_ACTION),
    (NOOP_ACTION, POSITIVE_ACTION),
    (NOOP_ACTION, NEGATIVE_ACTION),
    (NEGATIVE_ACTION, POSITIVE_ACTION),
    (NEGATIVE_ACTION, NEGATIVE_ACTION),
    (POSITIVE_ACTION, NOOP_ACTION),
    (NEGATIVE_ACTION, NOOP_ACTION),
)
ACTION_LABELS = tuple(
    f"factor_1_{('noop', 'positive', 'negative')[first]}_"
    f"factor_2_{('noop', 'positive', 'negative')[second]}"
    for first, second in ACTION_PAIRS
)
N_ACTIONS = len(ACTION_PAIRS)
VARIANT_3_DIRECTIONS = np.array(
    [
        [0.0, 1.0, -1.0],
        [0.0, -1.0, 1.0],
        [0.0, -1.0, -1.0],
    ],
    dtype=np.float64,
)
VARIANT_3_DIRECTIONS.setflags(write=False)


def _tilt_reward_state_probability(
    transition: np.ndarray,
    directions: np.ndarray,
    effect_size: float,
) -> np.ndarray:
    log_weights = np.log(np.asarray(transition, dtype=np.float64))
    log_weights[:, REWARD_STATE] += effect_size * directions
    log_weights -= log_weights.max(axis=1, keepdims=True)
    weights = np.exp(log_weights)
    return weights / weights.sum(axis=1, keepdims=True)


def factor_transition(action: int) -> np.ndarray:
    index = int(action)
    if index not in (NOOP_ACTION, POSITIVE_ACTION, NEGATIVE_ACTION):
        raise ValueError("factor action must be noop, positive, or negative")
    if index == NOOP_ACTION:
        return TRANSITION_MATRIX
    transition = _tilt_reward_state_probability(
        TRANSITION_MATRIX,
        VARIANT_3_DIRECTIONS[:, index],
        EFFECT_SIZE,
    )
    transition.setflags(write=False)
    return transition


def joint_transition(action: int) -> np.ndarray:
    index = int(action)
    if not 0 <= index < N_ACTIONS:
        raise ValueError("action must lie in [0, 9)")
    first, second = ACTION_PAIRS[index]
    return np.kron(factor_transition(first), factor_transition(second))


class TwoFactorVariant3Task:
    """Apply variant-3 transition control independently to both factors."""

    requires_belief = False

    def __init__(self, *, model: HMMModel, condition: str) -> None:
        if model.n_states != FACTOR_CARDINALITY**2:
            raise ValueError("two-factor control requires nine joint states")
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
        reward = (
            first_reward + second_reward
            if self.condition == "reward_both"
            else first_reward
        )
        return reward, {
            "factor_1_occupancy_reward": first_reward,
            "factor_2_occupancy_reward": second_reward,
        }

    def encode_action(self, executed_action: int) -> np.ndarray:
        encoded = np.zeros(N_ACTIONS, dtype=np.float32)
        encoded[int(executed_action)] = 1.0
        return encoded
