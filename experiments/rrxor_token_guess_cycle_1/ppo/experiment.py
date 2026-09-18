"""Runnable RRXOR delayed-token PPO leaf."""

from harness.context import RunContext

from experiments.rrxor_token_guess_cycle_1.shared import run_condition


def run(context: RunContext):
    return run_condition(context)
