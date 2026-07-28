"""IQN continuous control with reward ``occupancy - 0.05 ||w||₂``."""

from __future__ import annotations

from experiments.mess3_reward_state_cycle_1.iqn_control import (
    build_config as build_iqn_config,
)
from experiments.mess3_reward_state_cycle_1.iqn_control import run_condition
from harness.context import RunContext


ACTION_NORM_COEFFICIENT = 0.05
TASK_KWARGS: dict[str, object] = {
    "action_norm_coefficient": ACTION_NORM_COEFFICIENT,
}


def build_config(context: RunContext):
    return build_iqn_config(context, task_kwargs=TASK_KWARGS)


def run(context: RunContext):
    return run_condition(
        context,
        condition="iqn_action_norm",
        task_kwargs=TASK_KWARGS,
    )
