"""IQN continuous control with reward for state-2 occupancy only."""

from __future__ import annotations

from experiments.mess3_reward_state_cycle_1.iqn_control import (
    build_config as build_iqn_config,
)
from experiments.mess3_reward_state_cycle_1.iqn_control import run_condition
from harness.context import RunContext


TASK_KWARGS: dict[str, object] = {}


def build_config(context: RunContext):
    return build_iqn_config(context, task_kwargs=TASK_KWARGS)


def run(context: RunContext):
    return run_condition(
        context,
        condition="iqn_occupancy_only",
        task_kwargs=TASK_KWARGS,
    )
