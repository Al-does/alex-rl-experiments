"""Discrete, state-dependent transition control for reward-state MESS3."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.hmm import ActionDecision, HMMModel, TransitionEvent


NOOP_ACTION = 0
POSITIVE_ACTION = 1
NEGATIVE_ACTION = 2
N_ACTIONS = 3
REWARD_STATE = 2

_DIRECTIONS = {
    1: np.array(
        [
            [0.0, 1.0, -1.0],
            [0.0, 1.0, -1.0],
            [0.0, 1.0, -1.0],
        ]
    ),
    2: np.array(
        [
            [0.0, 1.0, -1.0],
            [0.0, 1.0, -1.0],
            [0.0, -1.0, -1.0],
        ]
    ),
    3: np.array(
        [
            [0.0, 1.0, -1.0],
            [0.0, -1.0, 1.0],
            [0.0, -1.0, -1.0],
        ]
    ),
}
for _directions in _DIRECTIONS.values():
    _directions.setflags(write=False)


def _tilt_reward_state_probability(
    transition: np.ndarray,
    directions: np.ndarray,
    effect_size: float,
) -> np.ndarray:
    """Tilt state-2 odds while preserving each row's other odds ratio."""

    log_weights = np.log(np.asarray(transition, dtype=np.float64))
    log_weights[:, REWARD_STATE] += effect_size * directions
    log_weights -= log_weights.max(axis=1, keepdims=True)
    weights = np.exp(log_weights)
    return weights / weights.sum(axis=1, keepdims=True)


class ActionSymmetryTask:
    """Reward state 2 under one of three discrete transition-control variants.

    RLlib actions are zero-based. They correspond to the first (noop), second
    (positive), and third (negative) actions in the experiment description.
    """

    requires_belief = False

    def __init__(
        self,
        *,
        model: HMMModel,
        variant: int,
        effect_size: float = 1.5,
    ) -> None:
        if model.n_states != 3:
            raise ValueError("action-symmetry control requires a three-state HMM")
        if variant not in _DIRECTIONS:
            raise ValueError("variant must be one of 1, 2, or 3")
        if not np.isfinite(effect_size) or effect_size <= 0.0:
            raise ValueError("effect_size must be finite and positive")
        if (model.transition_matrix <= 0.0).any():
            raise ValueError("action-symmetry control requires positive transitions")

        self.variant = int(variant)
        self.effect_size = float(effect_size)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.action_observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(N_ACTIONS,),
            dtype=np.float32,
        )
        reference = np.array(
            model.transition_matrix,
            dtype=np.float64,
            copy=True,
        )
        reference.setflags(write=False)
        self.reference_transition_matrix = reference
        directions = _DIRECTIONS[self.variant]
        self._transition_matrices = (
            self.reference_transition_matrix,
            _tilt_reward_state_probability(
                self.reference_transition_matrix,
                directions[:, POSITIVE_ACTION],
                self.effect_size,
            ),
            _tilt_reward_state_probability(
                self.reference_transition_matrix,
                directions[:, NEGATIVE_ACTION],
                self.effect_size,
            ),
        )
        for transition in self._transition_matrices[1:]:
            transition.setflags(write=False)

    def reset(self) -> None:
        pass

    def transition_matrix_for_action(self, action: int) -> np.ndarray:
        """Return the immutable transition law associated with one action."""

        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"action {action} is outside the action space")
        return self._transition_matrices[action]

    def resolve_action(
        self,
        action: int,
        state: int,
        model: HMMModel,
    ) -> ActionDecision:
        del state, model
        executed = int(action)
        transition = self.transition_matrix_for_action(executed)
        return ActionDecision(
            requested_action=executed,
            executed_action=executed,
            transition_matrix=transition,
            metadata={
                "reference_transition_matrix": self.reference_transition_matrix,
            },
        )

    def reward(
        self,
        event: TransitionEvent,
        decision: ActionDecision,
    ) -> tuple[float, dict[str, float]]:
        del decision
        occupancy = float(event.state_before == REWARD_STATE)
        return occupancy, {"occupancy_reward": occupancy}

    def encode_action(self, executed_action: int) -> np.ndarray:
        encoded = np.zeros(N_ACTIONS, dtype=np.float32)
        encoded[int(executed_action)] = 1.0
        return encoded
