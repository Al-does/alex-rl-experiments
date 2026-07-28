"""Variant 2 intervention with no observation delay."""

from harness.context import RunContext

from experiments.mess3_reward_state_action_symmetry_cycle_1.shared import (
    build_config as _build_config,
    run_condition,
)


EXPLORATORY_ENV_STEPS = 1_000_000


def build_config(context: RunContext):
    return _build_config(context, 2, delay=0)


def run(context: RunContext):
    return run_condition(
        context,
        2,
        condition="variant_2_no_delay",
        delay=0,
        total_env_steps=EXPLORATORY_ENV_STEPS,
    )
