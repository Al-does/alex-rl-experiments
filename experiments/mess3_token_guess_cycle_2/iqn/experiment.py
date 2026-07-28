"""Gamma-zero PPO with an implicit-quantile value critic."""

from harness.context import RunContext

from experiments.mess3_token_guess_cycle_2.shared import run_condition


def run(context: RunContext):
    return run_condition(context, "iqn")
