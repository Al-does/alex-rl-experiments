"""Most guesses rotate the chain, so action-blind beliefs are near useless."""

from harness.context import RunContext

from experiments.mess3_feedback_cycle_1.shared import run_condition


def run(context: RunContext):
    return run_condition(context, "strong_feedback")
