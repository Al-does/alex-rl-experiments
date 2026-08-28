"""Paper-matched independent MESS3 factors and mixed-radix joint tokens."""

from __future__ import annotations

from typing import Any

import numpy as np

from envs.hmm import HMMModel
from envs.mess3.model import control_model


MESS3_ALPHA = 0.6
MESS3_X = 0.15
MESS3_Y = 1.0 - 2.0 * MESS3_X
FACTOR_CARDINALITY = 3
FACTOR_STATE_DIMENSION = 3
FACTOR_COUNTS = (2, 3)

PAPER_TRANSITION_MATRIX = np.full(
    (FACTOR_STATE_DIMENSION, FACTOR_STATE_DIMENSION),
    MESS3_X,
    dtype=np.float64,
)
np.fill_diagonal(PAPER_TRANSITION_MATRIX, MESS3_Y)
PAPER_TRANSITION_MATRIX.setflags(write=False)


def paper_mess3_model() -> HMMModel:
    """Build Appendix C.1.1 MESS3 with ``alpha=.6`` and ``x=.15``."""

    return control_model(
        alpha=MESS3_ALPHA,
        transition_matrix=PAPER_TRANSITION_MATRIX,
    )


def factor_specifications(factor_count: int) -> list[dict[str, Any]]:
    """Return RLlib-serializable independent factor factory specifications."""

    if factor_count not in FACTOR_COUNTS:
        raise ValueError(f"factor_count must be one of {FACTOR_COUNTS}")
    return [
        {
            "factory": (
                "experiments.factored_representations_reproduction_PPO_2026_08."
                "process:paper_mess3_model"
            )
        }
        for _ in range(factor_count)
    ]


def joint_token_count(factor_count: int) -> int:
    if factor_count not in FACTOR_COUNTS:
        raise ValueError(f"factor_count must be one of {FACTOR_COUNTS}")
    return FACTOR_CARDINALITY**factor_count


def encode_joint_tokens(subtokens: np.ndarray) -> np.ndarray:
    """Encode final-axis factor subtokens using the harness mixed-radix order."""

    values = np.asarray(subtokens, dtype=np.int64)
    if values.ndim < 1 or values.shape[-1] not in FACTOR_COUNTS:
        raise ValueError("subtokens must end with a supported factor count")
    if ((values < 0) | (values >= FACTOR_CARDINALITY)).any():
        raise ValueError("factor subtokens must lie in [0, 3)")
    powers = FACTOR_CARDINALITY ** np.arange(
        values.shape[-1] - 1,
        -1,
        -1,
        dtype=np.int64,
    )
    return values @ powers


def decode_joint_tokens(tokens: np.ndarray, factor_count: int) -> np.ndarray:
    """Decode mixed-radix joint tokens into one subtoken per factor."""

    count = joint_token_count(factor_count)
    values = np.asarray(tokens, dtype=np.int64)
    if ((values < 0) | (values >= count)).any():
        raise ValueError(f"joint tokens must lie in [0, {count})")
    powers = FACTOR_CARDINALITY ** np.arange(
        factor_count - 1,
        -1,
        -1,
        dtype=np.int64,
    )
    return (values[..., None] // powers) % FACTOR_CARDINALITY


def environment_config(factor_count: int) -> dict[str, Any]:
    """Return delayed next-joint-token prediction over independent factors."""

    return {
        "model": {
            "factory": "envs.hmm:factored_model",
            "kwargs": {"factors": factor_specifications(factor_count)},
        },
        "task": {
            "class": (
                "experiments.factored_representations_reproduction_PPO_2026_08."
                "task:NextJointTokenGuessTask"
            )
        },
        "observation": {"action": None},
        "delay": 1,
        # BOS plus eight decision positions matches the paper's n_ctx=9 setup.
        "episode_length": 9,
        "randomize_first_episode_length": True,
    }
