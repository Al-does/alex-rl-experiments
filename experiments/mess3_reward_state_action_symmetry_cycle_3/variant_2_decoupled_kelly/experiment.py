"""Variant 2 PPO plus decoupled Kelly (token-guess style)."""

from harness.context import RunContext

from experiments.mess3_reward_state_action_symmetry_cycle_3.shared import (
    build_config as _build_config,
    run_condition,
)


def build_config(context: RunContext):
    return _build_config(context, 2, "decoupled_kelly")


def run(context: RunContext):
    return run_condition(context, 2, "decoupled_kelly")
