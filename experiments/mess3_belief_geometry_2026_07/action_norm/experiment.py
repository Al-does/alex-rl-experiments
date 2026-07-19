"""Continuous MESS3 control with reward ``occupancy - 0.05 ||w||_2``."""

from __future__ import annotations

from experiments.mess3_belief_geometry_2026_07.control_costs import (
    build_config as build_control_config,
)
from experiments.mess3_belief_geometry_2026_07.control_costs import (
    run_condition,
)
from harness.context import RunContext


# At the action-box corners this costs 0.354, comparable to the selected
# KL/4 condition's state-dependent 0.025--0.50 corner costs.
ACTION_NORM_COEFFICIENT = 0.05
TASK_KWARGS: dict[str, object] = {
    "action_norm_coefficient": ACTION_NORM_COEFFICIENT,
}


def build_config(context: RunContext):
    return build_control_config(context, task_kwargs=TASK_KWARGS)


def run(context: RunContext):
    return run_condition(
        context,
        condition="action_norm",
        task_kwargs=TASK_KWARGS,
    )
