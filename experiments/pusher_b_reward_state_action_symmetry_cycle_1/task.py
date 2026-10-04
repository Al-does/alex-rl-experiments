from __future__ import annotations

from numbers import Integral

import gymnasium as gym
import numpy as np

from envs.hmm import ActionDecision, HMMModel, TransitionEvent
from experiments.pusher_b_reward_state_action_symmetry_cycle_1.process import (
    EFFECT_SIZE,
    STATE_COUNT,
)


NOOP_ACTION = 0
POSITIVE_ACTION = 1
NEGATIVE_ACTION = 2
N_ACTIONS = 3


def direction_matrix(variant: int, reward_state: int) -> np.ndarray:
    if variant not in (1, 2, 3):
        raise ValueError("variant must be one of 1, 2, or 3")
    if not 0 <= reward_state < STATE_COUNT:
        raise ValueError("reward_state is outside the state space")
    directions = np.zeros((STATE_COUNT, N_ACTIONS), dtype=np.float64)
    directions[:, NEGATIVE_ACTION] = -1.0
    if variant == 1:
        directions[:, POSITIVE_ACTION] = 1.0
    elif variant == 2:
        directions[:, POSITIVE_ACTION] = 1.0
        directions[reward_state, POSITIVE_ACTION] = -1.0
    else:
        positive_state = (reward_state + 1) % STATE_COUNT
        negative_state = (reward_state + 2) % STATE_COUNT
        directions[positive_state, POSITIVE_ACTION] = 1.0
        directions[positive_state, NEGATIVE_ACTION] = -1.0
        directions[negative_state, POSITIVE_ACTION] = -1.0
        directions[negative_state, NEGATIVE_ACTION] = 1.0
        directions[reward_state, POSITIVE_ACTION] = -1.0
        directions[reward_state, NEGATIVE_ACTION] = -1.0
    directions.setflags(write=False)
    return directions


def tilt_reward_state_probability(
    transition: np.ndarray,
    directions: np.ndarray,
    effect_size: float,
    reward_state: int,
) -> np.ndarray:
    matrix = np.asarray(transition, dtype=np.float64)
    direction = np.asarray(directions, dtype=np.float64)
    if matrix.shape != (STATE_COUNT, STATE_COUNT):
        raise ValueError("transition must be a three-state matrix")
    if direction.shape != (STATE_COUNT,):
        raise ValueError("directions must contain one value per source state")
    if not np.isfinite(effect_size) or effect_size <= 0.0:
        raise ValueError("effect_size must be finite and positive")
    reward_probability = matrix[:, reward_state]
    odds_multiplier = np.exp(effect_size * direction)
    denominator = (
        1.0 - reward_probability + reward_probability * odds_multiplier
    )
    tilted_reward_probability = (
        reward_probability * odds_multiplier / denominator
    )
    controlled = np.zeros_like(matrix)
    other_states = np.arange(STATE_COUNT) != reward_state
    for state in range(STATE_COUNT):
        q = reward_probability[state]
        controlled[state, reward_state] = tilted_reward_probability[state]
        controlled[state, other_states] = (
            matrix[state, other_states]
            * (1.0 - tilted_reward_probability[state])
            / (1.0 - q)
        )
    return controlled


def controlled_edge_matrices(
    reference_edges: np.ndarray,
    controlled_transition: np.ndarray,
) -> np.ndarray:
    edges = np.asarray(reference_edges, dtype=np.float64)
    reference_transition = edges.sum(axis=0)
    if controlled_transition.shape != reference_transition.shape:
        raise ValueError("controlled transition has the wrong shape")
    if np.any((reference_transition == 0.0) & (controlled_transition != 0.0)):
        raise ValueError("controlled transition must preserve edge support")
    ratio = np.divide(
        controlled_transition,
        reference_transition,
        out=np.zeros_like(controlled_transition),
        where=reference_transition > 0.0,
    )
    controlled = edges * ratio[None, :, :]
    if not np.allclose(controlled.sum(axis=0), controlled_transition):
        raise AssertionError("controlled edge kernels do not match transition")
    return controlled


class ActionSymmetryTask:
    requires_belief = False

    def __init__(
        self,
        *,
        model: HMMModel,
        variant: int,
        reward_state: str,
        effect_size: float = EFFECT_SIZE,
    ) -> None:
        if model.n_states != STATE_COUNT:
            raise ValueError("action symmetry requires a three-state HMM")
        if model.edge_transition_matrices is None:
            raise ValueError("action symmetry requires an edge-emitting model")
        if reward_state not in model.state_labels:
            raise ValueError(f"unknown reward state: {reward_state!r}")
        if variant not in (1, 2, 3):
            raise ValueError("variant must be one of 1, 2, or 3")
        if not np.isfinite(effect_size) or effect_size <= 0.0:
            raise ValueError("effect_size must be finite and positive")
        if np.any(model.transition_matrix <= 0.0):
            raise ValueError("action symmetry requires positive transitions")

        self.variant = int(variant)
        self.reward_state_label = reward_state
        self.reward_state = model.state_labels.index(reward_state)
        self.effect_size = float(effect_size)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.action_observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(N_ACTIONS,),
            dtype=np.float32,
        )
        self.directions = direction_matrix(self.variant, self.reward_state)
        self.reference_transition_matrix = np.array(
            model.transition_matrix,
            dtype=np.float64,
            copy=True,
        )
        self.reference_transition_matrix.setflags(write=False)
        self.reference_edge_transition_matrices = np.array(
            model.edge_transition_matrices,
            dtype=np.float64,
            copy=True,
        )
        self.reference_edge_transition_matrices.setflags(write=False)
        self._transition_matrices = tuple(
            self._transition_for_action(action) for action in range(N_ACTIONS)
        )
        self._edge_transition_matrices = tuple(
            controlled_edge_matrices(
                self.reference_edge_transition_matrices,
                transition,
            )
            for transition in self._transition_matrices
        )
        for value in (
            *self._transition_matrices,
            *self._edge_transition_matrices,
        ):
            value.setflags(write=False)

    def _transition_for_action(self, action: int) -> np.ndarray:
        if action == NOOP_ACTION:
            return self.reference_transition_matrix.copy()
        return tilt_reward_state_probability(
            self.reference_transition_matrix,
            self.directions[:, action],
            self.effect_size,
            self.reward_state,
        )

    def reset(self) -> None:
        pass

    def transition_matrix_for_action(self, action: int) -> np.ndarray:
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise ValueError("action must be an integer")
        executed = int(action)
        if not self.action_space.contains(executed):
            raise ValueError(f"action {executed} is outside the action space")
        return self._transition_matrices[executed]

    def edge_transition_matrices_for_action(self, action: int) -> np.ndarray:
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise ValueError("action must be an integer")
        executed = int(action)
        if not self.action_space.contains(executed):
            raise ValueError(f"action {executed} is outside the action space")
        return self._edge_transition_matrices[executed]

    def resolve_action(
        self,
        action: int,
        state: int,
        model: HMMModel,
    ) -> ActionDecision:
        del state, model
        transition = self.transition_matrix_for_action(action)
        edges = self.edge_transition_matrices_for_action(action)
        executed = int(action)
        return ActionDecision(
            requested_action=executed,
            executed_action=executed,
            transition_matrix=transition,
            edge_transition_matrices=edges,
            metadata={
                "reference_transition_matrix": (
                    self.reference_transition_matrix
                ),
                "reference_edge_transition_matrices": (
                    self.reference_edge_transition_matrices
                ),
                "reward_state": self.reward_state,
            },
        )

    def reward(
        self,
        event: TransitionEvent,
        decision: ActionDecision,
    ) -> tuple[float, dict[str, float]]:
        del decision
        occupancy = float(event.state_before == self.reward_state)
        return occupancy, {"occupancy_reward": occupancy}

    def encode_action(self, executed_action: int) -> np.ndarray:
        encoded = np.zeros(N_ACTIONS, dtype=np.float32)
        encoded[int(executed_action)] = 1.0
        return encoded
