from __future__ import annotations

from typing import Any

from envs.gol.model import controlled_kernels


SPEED = "half"
VARIANTS = (2, 3)


def environment_config(variant: int, speed: str = SPEED) -> dict[str, Any]:
    controlled_kernels(variant=variant, speed=speed)
    return {
        "model": {
            "factory": "envs.gol.model:gol_model",
            "kwargs": {"variant": variant, "speed": speed},
        },
        "task": {
            "class": "envs.gol.tasks.reward_state:GolRewardTask",
            "kwargs": {"variant": variant, "speed": speed},
        },
        "observation": {
            "token": {"offset": 0, "depth": 1},
            "action": {"offset": 0, "depth": 1},
            "belief": False,
            "hidden_state": False,
        },
        "delay": 0,
        "reset_emission": False,
        "episode_length": None,
        "randomize_first_episode_length": False,
    }
