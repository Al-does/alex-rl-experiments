from __future__ import annotations

from experiments.pusher_b.process import (
    CONTEXT_LENGTH,
)
from experiments.pusher_b.process import (
    environment_config as pusher_b_environment_config,
)


def environment_config(
    cost: float,
    *,
    observation_depth: int = 1,
) -> dict[str, object]:
    if observation_depth not in (1, CONTEXT_LENGTH):
        raise ValueError(
            f"observation_depth must be 1 or {CONTEXT_LENGTH}"
        )
    config = pusher_b_environment_config("b10")
    config["task"] = {
        "class": (
            "experiments.pusher_b10_guess_or_pass_2026_10.task:"
            "GuessOrPassTask"
        ),
        "kwargs": {"cost": float(cost)},
    }
    config["observation"] = {
        "token": {"depth": observation_depth},
        "action": None,
    }
    config["diagnostics"] = {"rewards": True}
    return config


def ppo_environment_config(cost: float) -> dict[str, object]:
    return environment_config(cost, observation_depth=1)


def sac_environment_config(cost: float) -> dict[str, object]:
    return environment_config(cost, observation_depth=CONTEXT_LENGTH)
