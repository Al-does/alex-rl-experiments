"""Deterministic composition: the guess rotates the chain exactly."""

from harness.context import RunContext

from experiments.mess3_feedback_cycle_1.shared import run_condition


def run(context: RunContext):
    return run_condition(context, "full_feedback")
