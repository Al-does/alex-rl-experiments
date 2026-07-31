"""Counterfactual for factoring_free with the previous guess hidden."""

from harness.context import RunContext

from experiments.mess3_feedback_factoring_cycle_1.shared import run_condition


def run(context: RunContext):
    return run_condition(context, "factoring_free_blind")
