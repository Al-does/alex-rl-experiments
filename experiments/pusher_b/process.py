from __future__ import annotations

import numpy as np

from envs.hmm import HMMModel, stationary_distribution


PRESETS = {
    "b90": {"d": 0.90, "b": 0.90},
    "b10": {"d": 0.90, "b": 0.10},
}
STATE_COUNT = 3
TOKEN_COUNT = 2
CONTEXT_LENGTH = 128
EPISODE_LENGTH = 127
BOS_TOKEN = TOKEN_COUNT
VOCAB_SIZE = TOKEN_COUNT + 1


def pusher_b_edge_matrices(*, d: float, b: float) -> np.ndarray:
    if not 0.0 <= d <= 1.0:
        raise ValueError("d must lie in [0, 1]")
    if not 0.0 <= b <= 1.0:
        raise ValueError("b must lie in [0, 1]")
    return np.asarray(
        [
            [
                [(1.0 - d) / 4.0, d / 2.0, (1.0 - d) / 4.0],
                [b * (1.0 - d) / 2.0, b * (1.0 - d) / 2.0, b * d],
                [
                    (1.0 - b) * d,
                    (1.0 - b) * (1.0 - d) / 2.0,
                    (1.0 - b) * (1.0 - d) / 2.0,
                ],
            ],
            [
                [(1.0 - d) / 4.0, (1.0 - d) / 4.0, d / 2.0],
                [
                    (1.0 - b) * d,
                    (1.0 - b) * (1.0 - d) / 2.0,
                    (1.0 - b) * (1.0 - d) / 2.0,
                ],
                [b * (1.0 - d) / 2.0, b * d, b * (1.0 - d) / 2.0],
            ],
        ],
        dtype=np.float64,
    )


def _parameters(preset: str) -> dict[str, float]:
    try:
        return PRESETS[preset]
    except KeyError as error:
        raise ValueError(f"unknown Pusher-B preset: {preset!r}") from error


def pusher_b_model(preset: str) -> HMMModel:
    parameters = _parameters(preset)
    edges = pusher_b_edge_matrices(**parameters)
    transition = edges.sum(axis=0)
    emission = edges.sum(axis=2).T
    return HMMModel(
        initial_distribution=stationary_distribution(transition),
        transition_matrix=transition,
        emission_matrix=emission,
        edge_transition_matrices=edges,
        state_labels=("A", "B", "C"),
        token_labels=("0", "1"),
    )


def pusher_b_model_b90() -> HMMModel:
    return pusher_b_model("b90")


def pusher_b_model_b10() -> HMMModel:
    return pusher_b_model("b10")


def environment_config(preset: str) -> dict[str, object]:
    _parameters(preset)
    return {
        "model": {
            "factory": (
                f"experiments.pusher_b.process:pusher_b_model_{preset}"
            ),
        },
        "task": {
            "class": "experiments.pusher_b.task:NextTokenGuessTask",
        },
        "observation": {
            "token": {"depth": 1},
            "action": None,
        },
        "delay": 1,
        "episode_length": EPISODE_LENGTH,
        "randomize_first_episode_length": False,
    }
