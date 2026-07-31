"""Ablation: identical feedback with the previous guess hidden from the input."""

from harness.context import RunContext

from experiments.mess3_feedback_cycle_1.shared import run_condition


def run(context: RunContext):
    return run_condition(context, "strong_feedback_blind")
