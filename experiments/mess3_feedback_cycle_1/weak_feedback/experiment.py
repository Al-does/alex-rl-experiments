"""A minority of guesses rotate the chain, leaving the register ambiguous."""

from harness.context import RunContext

from experiments.mess3_feedback_cycle_1.shared import run_condition


def run(context: RunContext):
    return run_condition(context, "weak_feedback")
