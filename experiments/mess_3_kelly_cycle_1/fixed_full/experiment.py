"""Train PPO token guesses with a fixed near-full Kelly wager."""

from experiments.mess_3_kelly_cycle_1.shared import (
    build_config as _build_config,
    run_condition,
)


CONDITION = "fixed_full"


def build_config(context):
    return _build_config(context, CONDITION)


def run(context):
    return run_condition(context, CONDITION)
