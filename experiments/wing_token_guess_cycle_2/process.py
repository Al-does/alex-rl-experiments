from __future__ import annotations

WING_ALPHA = 0.94
WING_X = 0.4
CONTEXT_LENGTH = 32
EPISODE_LENGTH = 1024
FACTOR_COUNT = 1
TOKEN_COUNT = 2


def environment_config() -> dict[str, object]:
    return {
        "model": {
            "factory": "envs.wing.model:wing_model",
            "kwargs": {"alpha": WING_ALPHA, "x": WING_X},
        },
        "task": {
            "class": (
                "experiments.wing_token_guess_cycle_2.task:"
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
