"""REINFORCE condition for sticky-state action-symmetry variant 3."""

from experiments.mess3_reward_state_action_symmetry_cycle_6.shared import (
    build_config as _build_config,
    run_condition,
)
from harness.context import RunContext


def build_config(context: RunContext):
    return _build_config(context, 3)


def run(context: RunContext):
    return run_condition(context, 3)
