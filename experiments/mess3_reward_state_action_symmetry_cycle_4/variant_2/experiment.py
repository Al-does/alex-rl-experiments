"""Sticky-state PPO condition for action-symmetry variant 2."""

from experiments.mess3_reward_state_action_symmetry_cycle_4.shared import (
    build_config as _build_config,
    run_condition,
)
from harness.context import RunContext


def build_config(context: RunContext):
    return _build_config(context, 2)


def run(context: RunContext):
    return run_condition(context, 2)
