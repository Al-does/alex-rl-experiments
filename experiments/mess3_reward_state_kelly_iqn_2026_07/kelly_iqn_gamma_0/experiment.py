"""Predictive Kelly IQN PPO reward-state control with gamma zero."""

from experiments.mess3_reward_state_kelly_iqn_2026_07.shared import (
    build_config as build_battery_config,
    run_condition,
)
from harness.context import RunContext


def build_config(context: RunContext):
    return build_battery_config(
        context,
        gamma=0.0,
        use_iqn=True,
        use_kelly=True,
    )


def run(context: RunContext):
    return run_condition(
        context,
        condition="kelly_iqn_gamma_0",
        gamma=0.0,
        use_iqn=True,
        use_kelly=True,
    )
