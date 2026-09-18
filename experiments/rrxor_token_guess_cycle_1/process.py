"""Exact Random-Random-XOR process and delayed-observation environment."""

from __future__ import annotations

from collections import deque

import numpy as np

from envs.hmm import HMMModel


CONTEXT_LENGTH = 10
EPISODE_LENGTH = CONTEXT_LENGTH + 1
RRXOR_STATIONARY = np.array(
    [1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
    dtype=np.float64,
)
RRXOR_TENSOR = np.array(
    [
        [
            [0.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.5],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    ],
    dtype=np.float64,
)


def rrxor_model() -> HMMModel:
    """Return the five-state edge-emitting process from Shai et al."""

    transition = RRXOR_TENSOR.sum(axis=0)
    emission = RRXOR_TENSOR.sum(axis=2).T
    return HMMModel(
        initial_distribution=RRXOR_STATIONARY,
        transition_matrix=transition,
        emission_matrix=emission,
        state_labels=("random_1", "random_2_from_0", "random_2_from_1", "xor_1", "xor_0"),
        token_labels=("0", "1"),
        edge_transition_matrices=RRXOR_TENSOR,
    )


def filter_source_belief(
    belief: np.ndarray,
    token: int,
) -> np.ndarray:
    """Condition the source-state belief on one edge-emitted token."""

    if token not in (0, 1):
        raise ValueError("RRXOR token must be 0 or 1")
    posterior = np.asarray(belief, dtype=np.float64) @ RRXOR_TENSOR[token]
    total = float(posterior.sum())
    if total <= 0.0:
        raise ValueError("token is impossible under the supplied belief")
    return posterior / total


def next_token_probabilities(belief: np.ndarray) -> np.ndarray:
    """Return the exact pending-token distribution from a source belief."""

    return np.einsum(
        "...i,xij->...x",
        np.asarray(belief, dtype=np.float64),
        RRXOR_TENSOR,
    )


def reachable_beliefs() -> np.ndarray:
    """Enumerate the finite mixed-state presentation reachable from stationarity."""

    pending: deque[np.ndarray] = deque([RRXOR_STATIONARY])
    beliefs: list[np.ndarray] = []
    keys: set[tuple[float, ...]] = set()
    while pending:
        belief = pending.popleft()
        key = tuple(np.round(belief, decimals=12))
        if key in keys:
            continue
        keys.add(key)
        beliefs.append(belief)
        probabilities = next_token_probabilities(belief)
        for token, probability in enumerate(probabilities):
            if probability > 0.0:
                pending.append(filter_source_belief(belief, token))
    return np.stack(beliefs)


def environment_config() -> dict[str, object]:
    """Configure paper-length prefixes with token-only delayed observations."""

    return {
        "model": {
            "factory": (
                "experiments.rrxor_token_guess_cycle_1.process:rrxor_model"
            ),
        },
        "task": {
            "class": (
                "experiments.rrxor_token_guess_cycle_1.task:"
                "RRXORTokenGuessTask"
            ),
        },
        "observation": {
            "token": {"offset": 0, "depth": 1},
            "action": None,
            "belief": False,
            "hidden_state": False,
        },
        "diagnostics": {
            "state": True,
            "belief": True,
            "raw_belief": True,
            "tokens": True,
            "rewards": True,
            "transitions": False,
        },
        "delay": 1,
        "episode_length": EPISODE_LENGTH,
        "randomize_first_episode_length": False,
    }
