"""Pusher-B HMM (same edge matrices as ``experiments/pusher_b/process.py``)."""

from __future__ import annotations

import jax
import numpy as np
from envs.hmm import HMMModel, stationary_distribution
from envs.hmm.jax_env import JaxHMMEnv

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
    return np.asarray(
        [
            [
                [(1.0 - d) / 4.0, d / 2.0, (1.0 - d) / 4.0],
                [b * (1.0 - d) / 2.0, b * (1.0 - d) / 2.0, b * d],
                [(1.0 - b) * d, (1.0 - b) * (1.0 - d) / 2.0, (1.0 - b) * (1.0 - d) / 2.0],
            ],
            [
                [(1.0 - d) / 4.0, (1.0 - d) / 4.0, d / 2.0],
                [(1.0 - b) * d, (1.0 - b) * (1.0 - d) / 2.0, (1.0 - b) * (1.0 - d) / 2.0],
                [b * (1.0 - d) / 2.0, b * d, b * (1.0 - d) / 2.0],
            ],
        ],
        dtype=np.float64,
    )


def pusher_b_model(preset: str) -> HMMModel:
    if preset not in PRESETS:
        raise ValueError(f"unknown Pusher-B preset: {preset!r}")
    edges = pusher_b_edge_matrices(**PRESETS[preset])
    transition = edges.sum(axis=0)
    return HMMModel(
        initial_distribution=stationary_distribution(transition),
        transition_matrix=transition,
        emission_matrix=edges.sum(axis=2).T,
        edge_transition_matrices=edges,
        state_labels=("A", "B", "C"),
        token_labels=("0", "1"),
    )


def token_guess_reward(action: jax.Array, raw_token_before: jax.Array) -> jax.Array:
    """``experiments.pusher_b.task.NextTokenGuessTask.reward``."""

    return action == raw_token_before


def make_env(preset: str) -> JaxHMMEnv:
    """The ``experiments.pusher_b.process.environment_config`` env, in JAX."""

    return JaxHMMEnv.from_model(
        pusher_b_model(preset),
        episode_length=EPISODE_LENGTH,
        reward_fn=token_guess_reward,
        delay=1,
    )
