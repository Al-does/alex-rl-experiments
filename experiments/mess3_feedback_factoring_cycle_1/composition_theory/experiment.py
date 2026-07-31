"""Network-free sweep of ceilings, single-HMM identifiability, and factoring."""

from harness.context import RunContext

from experiments.mess3_feedback_factoring_cycle_1.theory import run_theory


def run(context: RunContext):
    return run_theory(context)
