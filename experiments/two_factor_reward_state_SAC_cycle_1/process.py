"""Two independent controlled MESS3 factors observed through joint tokens."""

from __future__ import annotations

from typing import Any

import numpy as np

from envs.hmm import HMMModel
from envs.mess3.model import control_model


FACTOR_COUNT = 2
FACTOR_CARDINALITY = 3
JOINT_STATE_COUNT = FACTOR_CARDINALITY**FACTOR_COUNT
JOINT_TOKEN_COUNT = FACTOR_CARDINALITY**FACTOR_COUNT
MESS3_ALPHA = 0.55
CONTEXT_LENGTH = 64
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
    """Build one factor with the preregistered transition and emission laws."""

    return control_model(
        alpha=MESS3_ALPHA,
        transition_matrix=TRANSITION_MATRIX,
    )


def factor_specifications() -> list[dict[str, Any]]:
    """Return two serializable independent-factor specifications."""

    factory = (
        "experiments.two_factor_reward_state_SAC_cycle_1.process:"
        "controlled_factor_model"
    )
    return [{"factory": factory} for _ in range(FACTOR_COUNT)]


def decode_joint_indices(indices: np.ndarray | int) -> np.ndarray:
    """Decode Cartesian-product state/token indices into factor coordinates."""

    values = np.asarray(indices, dtype=np.int64)
    if ((values < 0) | (values >= JOINT_STATE_COUNT)).any():
        raise ValueError("joint indices must lie in [0, 9)")
    return np.stack((values // FACTOR_CARDINALITY, values % FACTOR_CARDINALITY), -1)


def environment_config(condition: str) -> dict[str, Any]:
    """Build the action-aware fixed-history environment for one reward arm."""

    return {
        "model": {
            "factory": "envs.hmm:factored_model",
            "kwargs": {"factors": factor_specifications()},
        },
        "task": {
            "class": (
                "experiments.two_factor_reward_state_SAC_cycle_1.task:"
                "TwoFactorShiftTask"
            ),
            "kwargs": {"condition": condition},
        },
        # SAC is stateless in RLlib, so both visible joint tokens and the actions
        # that generated them are retained in a fixed newest-first window.
        "observation": {
            "token": {"depth": CONTEXT_LENGTH},
            "action": {"depth": CONTEXT_LENGTH},
        },
        "delay": 0,
        "episode_length": 1024,
        "randomize_first_episode_length": True,
    }
