from __future__ import annotations

from numbers import Integral

import gymnasium as gym
import numpy as np

from envs.hmm import ActionDecision, HMMModel, TransitionEvent
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.process import (
    COMPONENT_COUNT,
    EFFECT_SIZE,
    STATES_PER_COMPONENT,
)


NOOP_ACTION = 0
POSITIVE_ACTION = 1
NEGATIVE_ACTION = 2
N_ACTIONS = 3
REWARD_STATE = 2

DIRECTIONS = {
    1: np.asarray(
        [
            [0.0, 1.0, -1.0],
            [0.0, 1.0, -1.0],
            [0.0, 1.0, -1.0],
        ],
        dtype=np.float64,
    ),
    2: np.asarray(
        [
            [0.0, 1.0, -1.0],
            [0.0, 1.0, -1.0],
            [0.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    ),
    3: np.asarray(
        [
            [0.0, 1.0, -1.0],
            [0.0, -1.0, 1.0],
            [0.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    ),
}
for _directions in DIRECTIONS.values():
    _directions.setflags(write=False)


def tilt_reward_state_probability(
    transition: np.ndarray,
    directions: np.ndarray,
    effect_size: float,
) -> np.ndarray:
    matrix = np.asarray(transition, dtype=np.float64)
    direction = np.asarray(directions, dtype=np.float64)
    if matrix.shape != (STATES_PER_COMPONENT, STATES_PER_COMPONENT):
        raise ValueError("transition must be one three-state component")
    if direction.shape != (STATES_PER_COMPONENT,):
        raise ValueError("directions must contain one value per local state")
    reward_probability = matrix[:, REWARD_STATE]
    odds_multiplier = np.exp(effect_size * direction)
    denominator = 1.0 - reward_probability + reward_probability * odds_multiplier
    tilted_reward_probability = (
        reward_probability * odds_multiplier / denominator
    )
    controlled = np.zeros_like(matrix)
    other_states = np.arange(STATES_PER_COMPONENT) != REWARD_STATE
    for state in range(STATES_PER_COMPONENT):
        q = reward_probability[state]
        controlled[state, REWARD_STATE] = tilted_reward_probability[state]
        if q < 1.0:
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
        effect_size: float = EFFECT_SIZE,
    ) -> None:
        if model.n_states != COMPONENT_COUNT * STATES_PER_COMPONENT:
            raise ValueError("action symmetry requires dual three-state MESS3")
        if model.edge_transition_matrices is None:
            raise ValueError("action symmetry requires an edge-emitting model")
        if variant not in DIRECTIONS:
            raise ValueError("variant must be one of 1, 2, or 3")
        if not np.isfinite(effect_size) or effect_size <= 0.0:
            raise ValueError("effect_size must be finite and positive")

        self.variant = int(variant)
        self.effect_size = float(effect_size)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.action_observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(N_ACTIONS,),
            dtype=np.float32,
        )
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
        transition = np.zeros_like(self.reference_transition_matrix)
        directions = DIRECTIONS[self.variant][:, action]
        for component in range(COMPONENT_COUNT):
            start = component * STATES_PER_COMPONENT
            stop = start + STATES_PER_COMPONENT
            transition[start:stop, start:stop] = tilt_reward_state_probability(
                self.reference_transition_matrix[start:stop, start:stop],
                directions,
                self.effect_size,
            )
        return transition

    def reset(self) -> None:
        pass

    def transition_matrix_for_action(self, action: int) -> np.ndarray:
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise ValueError("action must be an integer")
        resolved = int(action)
        if not self.action_space.contains(resolved):
            raise ValueError(f"action {resolved} is outside the action space")
        return self._transition_matrices[resolved]

    def edge_transition_matrices_for_action(self, action: int) -> np.ndarray:
        self.transition_matrix_for_action(action)
        return self._edge_transition_matrices[int(action)]

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
                "reference_transition_matrix": self.reference_transition_matrix,
            },
        )

    def reward(
        self,
        event: TransitionEvent,
        decision: ActionDecision,
    ) -> tuple[float, dict[str, float]]:
        del decision
        local_state = event.state_before % STATES_PER_COMPONENT
        occupancy = float(local_state == REWARD_STATE)
        return occupancy, {"occupancy_reward": occupancy}

    def encode_action(self, executed_action: int) -> np.ndarray:
        encoded = np.zeros(N_ACTIONS, dtype=np.float32)
        encoded[int(executed_action)] = 1.0
        return encoded
