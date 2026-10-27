from __future__ import annotations

import numpy as np

from envs.hmm import HMMModel, stationary_distribution
from experiments.pusher_b.process import (
    CONTEXT_LENGTH,
    EPISODE_LENGTH,
    STATE_COUNT,
    TOKEN_COUNT,
)


EFFECT_SIZE = 2.5
REWARD_STATE = "A"
PERSISTENCE = 0.90
CLOCKWISE_PROBABILITY = 0.06
COUNTERCLOCKWISE_PROBABILITY = 0.04
TRANSITION_MATRIX = np.asarray(
    [
        [PERSISTENCE, CLOCKWISE_PROBABILITY, COUNTERCLOCKWISE_PROBABILITY],
        [COUNTERCLOCKWISE_PROBABILITY, PERSISTENCE, CLOCKWISE_PROBABILITY],
        [CLOCKWISE_PROBABILITY, COUNTERCLOCKWISE_PROBABILITY, PERSISTENCE],
    ],
    dtype=np.float64,
)
DESTINATION_EMISSION_MATRIX = np.asarray(
    [
        [0.50, 0.50],
        [0.75, 0.25],
        [0.25, 0.75],
    ],
    dtype=np.float64,
)
TRANSITION_MATRIX.setflags(write=False)
DESTINATION_EMISSION_MATRIX.setflags(write=False)


def sticky_cycle_edge_matrices() -> np.ndarray:
    edges = np.einsum(
        "ij,jx->xij",
        TRANSITION_MATRIX,
        DESTINATION_EMISSION_MATRIX,
    )
    edges.setflags(write=False)
    return edges


def sticky_cycle_model() -> HMMModel:
    edges = sticky_cycle_edge_matrices()
    return HMMModel(
        initial_distribution=stationary_distribution(TRANSITION_MATRIX),
        transition_matrix=TRANSITION_MATRIX,
        emission_matrix=edges.sum(axis=2).T,
        edge_transition_matrices=edges,
        state_labels=("A", "B", "C"),
        token_labels=("0", "1"),
    )


def environment_config(variant: int) -> dict[str, object]:
    if variant not in (2, 3):
        raise ValueError("variant must be 2 or 3")
    return {
        "model": {
            "factory": (
                "experiments.pusher_b_reward_state_action_symmetry_cycle_2"
                ".process:sticky_cycle_model"
            ),
        },
        "task": {
            "class": (
                "experiments.pusher_b_reward_state_action_symmetry_cycle_1"
                ".task:ActionSymmetryTask"
            ),
            "kwargs": {
                "variant": variant,
                "reward_state": REWARD_STATE,
                "effect_size": EFFECT_SIZE,
            },
        },
        "observation": {
            "token": {"depth": 1},
            "action": {"depth": 1},
        },
        "delay": 1,
        "episode_length": EPISODE_LENGTH,
        "randomize_first_episode_length": False,
    }


__all__ = [
    "CLOCKWISE_PROBABILITY",
    "CONTEXT_LENGTH",
    "COUNTERCLOCKWISE_PROBABILITY",
    "DESTINATION_EMISSION_MATRIX",
    "EFFECT_SIZE",
    "EPISODE_LENGTH",
    "PERSISTENCE",
    "REWARD_STATE",
    "STATE_COUNT",
    "TOKEN_COUNT",
    "TRANSITION_MATRIX",
    "environment_config",
    "sticky_cycle_edge_matrices",
    "sticky_cycle_model",
]
