"""PPO arm rewarding state 2 in both factors."""

from harness.context import RunContext

from experiments.two_factor_reward_state_PPO_cycle_1.shared import (
    build_config as _build_config,
    run_condition,
)


CONDITION = "reward_both"


def build_config(context: RunContext):
    return _build_config(context, CONDITION)


def run(context: RunContext):
    return run_condition(context, CONDITION)
