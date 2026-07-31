"""Factoring costs about two thirds of its maximum (epsilon = 0.85)."""

from harness.context import RunContext

from experiments.mess3_feedback_cycle_1.shared import run_condition


def run(context: RunContext):
    return run_condition(context, "factoring_costly")
