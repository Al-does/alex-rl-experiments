"""REINFORCE arm rewarding state 2 in both factors."""

from experiments.two_factor_reward_state_REINFORCE_cycle_3.shared import (
    build_config as _build_config,
    run_condition,
)
from harness.context import RunContext


CONDITION = "reward_both"


def build_config(context: RunContext):
    return _build_config(context, CONDITION)


def run(context: RunContext):
    return run_condition(context, CONDITION)
