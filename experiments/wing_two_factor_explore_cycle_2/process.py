from __future__ import annotations

from typing import Any

from experiments.wing_two_factor_explore_cycle_1.process import (
    ACTION_PAIRS,
    CONDITIONS,
    CONTEXT_LENGTH,
    EPISODE_LENGTH,
    FACTOR_COUNT,
    JOINT_TOKEN_COUNT,
    WING_ALPHA,
    WING_X,
    environment_config as cycle_1_environment_config,
)


CONTROL_STRENGTH = 1.0
REWARD_STATE = 0


def environment_config(condition: str) -> dict[str, Any]:
    config = cycle_1_environment_config(condition, REWARD_STATE)
    config["task"]["kwargs"]["strength"] = CONTROL_STRENGTH
    return config
