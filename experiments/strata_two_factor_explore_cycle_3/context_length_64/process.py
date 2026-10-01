from __future__ import annotations

from typing import Any

import numpy as np


STRATA_ALPHA = 0.98
STRATA_T0 = 0.30
STRATA_T1 = 0.80
CONTROL_STRENGTH = 1.0
REWARD_STATE = 0
CONTEXT_LENGTH = 64
EPISODE_LENGTH = 1024
FACTOR_COUNT = 2
JOINT_TOKEN_COUNT = 4
CONDITIONS = ("reward_both", "reward_factor_1")
ACTION_PAIRS = tuple((first, second) for first in range(3) for second in range(3))


def factor_specifications() -> list[dict[str, Any]]:
    return [
        {
            "factory": "envs.strata.model:strata_model",
            "kwargs": {
                "alpha": STRATA_ALPHA,
                "t0": STRATA_T0,
                "t1": STRATA_T1,
            },
        }
        for _ in range(FACTOR_COUNT)
    ]


def decode_joint_indices(indices: np.ndarray | int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if ((values < 0) | (values >= 9)).any():
        raise ValueError("joint state indices must lie in [0, 9)")
    return np.stack((values // 3, values % 3), axis=-1)


def environment_config(
    condition: str,
    reward_state: int = REWARD_STATE,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if isinstance(reward_state, bool) or reward_state not in (0, 1, 2):
        raise ValueError("reward_state must be 0, 1, or 2")
    return {
        "model": {
            "factory": "envs.hmm:factored_model",
            "kwargs": {"factors": factor_specifications()},
        },
        "task": {
            "class": "envs.strata.tasks.reward_state:StrataRewardTask",
            "kwargs": {
                "reward_state": reward_state,
                "rewarded_factors": [0, 1] if condition == "reward_both" else [0],
                "alpha": STRATA_ALPHA,
                "t0": STRATA_T0,
                "t1": STRATA_T1,
                "strength": CONTROL_STRENGTH,
            },
        },
        "observation": {"token": {"depth": 1}, "action": {"depth": 1}},
        "delay": 0,
        "episode_length": EPISODE_LENGTH,
        "randomize_first_episode_length": True,
    }
