from __future__ import annotations

STRATA_ALPHA = 0.97
STRATA_T0 = 0.38
STRATA_T1 = 0.54
CONTEXT_LENGTH = 32
EPISODE_LENGTH = 1024
FACTOR_COUNT = 1
TOKEN_COUNT = 2


def environment_config() -> dict[str, object]:
    return {
        "model": {
            "factory": "envs.strata.model:strata_model",
            "kwargs": {
                "alpha": STRATA_ALPHA,
                "t0": STRATA_T0,
                "t1": STRATA_T1,
            },
        },
        "task": {
            "class": (
                "experiments.strata_token_guess_cycle_1.task:"
                "StrataTokenGuessTask"
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
