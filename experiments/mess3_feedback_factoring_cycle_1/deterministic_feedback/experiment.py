"""Every guess rotates the register exactly (kappa = 1)."""

from harness.context import RunContext

from experiments.mess3_feedback_factoring_cycle_1.shared import run_condition


def run(context: RunContext):
    return run_condition(context, "deterministic_feedback")
