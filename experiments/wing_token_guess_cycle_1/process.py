from __future__ import annotations

from typing import Any


WING_ALPHA = 0.94
WING_X = 0.4
CONTEXT_LENGTH = 32
EPISODE_LENGTH = 1024
FACTOR_COUNT = 2
JOINT_TOKEN_COUNT = 2**FACTOR_COUNT


def factor_specifications() -> list[dict[str, Any]]:
    return [
        {
            "factory": "envs.wing.model:wing_model",
            "kwargs": {"alpha": WING_ALPHA, "x": WING_X},
        }
        for _ in range(FACTOR_COUNT)
    ]


def environment_config() -> dict[str, Any]:
    return {
        "model": {
            "factory": "envs.hmm:factored_model",
            "kwargs": {"factors": factor_specifications()},
        },
        "task": {
            "class": (
                "experiments.wing_token_guess_cycle_1.task:"
                "WingTokenGuessTask"
            ),
        },
        "observation": {
            "token": {"depth": 1},
            "action": None,
        },
        "delay": 1,
        "episode_length": EPISODE_LENGTH,
        "randomize_first_episode_length": True,
    }
