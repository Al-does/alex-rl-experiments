"""Gamma-zero synchronous advantage actor-critic condition."""

from harness.context import RunContext

from experiments.mess3_token_guess_cycle_2.shared import run_condition


def run(context: RunContext):
    return run_condition(context, "a2c")
