"""Two sticky, independently controlled MESS3 factors observed jointly."""

from __future__ import annotations

from typing import Any

import numpy as np

from envs.hmm import HMMModel
from envs.mess3.model import control_model


FACTOR_COUNT = 2
FACTOR_CARDINALITY = 3
JOINT_STATE_COUNT = FACTOR_CARDINALITY**FACTOR_COUNT
JOINT_TOKEN_COUNT = FACTOR_CARDINALITY**FACTOR_COUNT
MESS3_ALPHA = 0.85
LOCAL_CONTEXT_LENGTH = 10
TRANSFORMER_LAYERS = 4
# Cycle 5's banded transformer retains n_layers * context_len prior frames.
TRANSFORMER_LOOKBACK = TRANSFORMER_LAYERS * LOCAL_CONTEXT_LENGTH
SAC_HISTORY_LENGTH = TRANSFORMER_LOOKBACK + 1
TRANSITION_MATRIX = np.array(
    [
        [0.75, 0.15, 0.10],
        [0.15, 0.75, 0.10],
        [0.30, 0.30, 0.40],
    ],
    dtype=np.float64,
)
TRANSITION_MATRIX.setflags(write=False)


def controlled_factor_model() -> HMMModel:
    return control_model(
        alpha=MESS3_ALPHA,
        transition_matrix=TRANSITION_MATRIX,
    )


def factor_specifications() -> list[dict[str, Any]]:
    factory = (
        "experiments.two_factor_reward_state_SAC_cycle_2.process:"
        "controlled_factor_model"
    )
    return [{"factory": factory} for _ in range(FACTOR_COUNT)]


def decode_joint_indices(indices: np.ndarray | int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if ((values < 0) | (values >= JOINT_STATE_COUNT)).any():
        raise ValueError("joint indices must lie in [0, 9)")
    return np.stack((values // FACTOR_CARDINALITY, values % FACTOR_CARDINALITY), -1)


def environment_config(condition: str) -> dict[str, Any]:
    return {
        "model": {
            "factory": "envs.hmm:factored_model",
            "kwargs": {"factors": factor_specifications()},
        },
        "task": {
            "class": (
                "experiments.two_factor_reward_state_SAC_cycle_2.task:"
                "TwoFactorVariant3Task"
            ),
            "kwargs": {"condition": condition},
        },
        "observation": {
            "token": {"depth": SAC_HISTORY_LENGTH},
            "action": {"depth": SAC_HISTORY_LENGTH},
        },
        "delay": 0,
        "episode_length": 1024,
        "randomize_first_episode_length": True,
    }
