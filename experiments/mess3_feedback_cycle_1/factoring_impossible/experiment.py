"""Register sub-token is pure noise: factoring is vacuous (epsilon = 1)."""

from harness.context import RunContext

from experiments.mess3_feedback_cycle_1.shared import run_condition


def run(context: RunContext):
    return run_condition(context, "factoring_impossible")
