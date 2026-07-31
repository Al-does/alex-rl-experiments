"""Control arm: the guess is scored but never moves the hidden process."""

from harness.context import RunContext

from experiments.mess3_feedback_cycle_1.shared import run_condition


def run(context: RunContext):
    return run_condition(context, "no_feedback")
