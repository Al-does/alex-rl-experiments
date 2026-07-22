"""Scalar-value PPO reward-state control with gamma 0.99."""

from experiments.mess3_reward_state_kelly_iqn_2026_07.shared import (
    build_config as build_battery_config,
    run_condition,
)
from harness.context import RunContext


def build_config(context: RunContext):
    return build_battery_config(
        context,
        gamma=0.99,
        use_iqn=False,
        use_kelly=False,
    )


def run(context: RunContext):
    return run_condition(
        context,
        condition="ppo_gamma_099",
        gamma=0.99,
        use_iqn=False,
        use_kelly=False,
    )
