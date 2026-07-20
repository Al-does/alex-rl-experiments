"""Continuous MESS3 control with reward for state-2 occupancy only."""

from __future__ import annotations

from experiments.mess3_belief_geometry_2026_07.control_costs import (
    build_config as build_control_config,
)
from experiments.mess3_belief_geometry_2026_07.control_costs import (
    run_condition,
)
from harness.context import RunContext


TASK_KWARGS: dict[str, object] = {}


def build_config(context: RunContext):
    return build_control_config(context, task_kwargs=TASK_KWARGS)


def run(context: RunContext):
    return run_condition(
        context,
        condition="occupancy_only",
        task_kwargs=TASK_KWARGS,
    )
