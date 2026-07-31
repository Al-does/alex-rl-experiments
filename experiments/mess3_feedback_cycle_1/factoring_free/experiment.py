"""Register reports itself exactly: factoring is lossless (epsilon = 0)."""

from harness.context import RunContext

from experiments.mess3_feedback_cycle_1.shared import run_condition


def run(context: RunContext):
    return run_condition(context, "factoring_free")
