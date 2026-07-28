"""Gamma-zero clipped PPO baseline."""

from harness.context import RunContext

from experiments.mess3_token_guess_cycle_2.shared import run_condition


def run(context: RunContext):
    return run_condition(context, "ppo")
