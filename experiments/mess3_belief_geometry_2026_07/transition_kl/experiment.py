"""Continuous MESS3 control with reward ``occupancy - KL / 4``."""

from __future__ import annotations

from experiments.mess3_belief_geometry_2026_07.control_costs import (
    build_config as build_control_config,
)
from experiments.mess3_belief_geometry_2026_07.control_costs import (
    run_condition,
)
from harness.context import RunContext


# The pre-cutover analytic sweep selected beta=4 because it retained a large
# belief-policy advantage without collapsing control toward zero.
TASK_KWARGS: dict[str, object] = {"transition_kl_beta": 4.0}


def build_config(context: RunContext):
    return build_control_config(context, task_kwargs=TASK_KWARGS)


def run(context: RunContext):
    return run_condition(
        context,
        condition="transition_kl",
        task_kwargs=TASK_KWARGS,
    )
