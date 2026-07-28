"""Variant 2 with high initial entropy annealed to zero over 5M steps."""

from harness.context import RunContext

from experiments.mess3_reward_state_action_symmetry_cycle_2.shared import (
    ENTROPY_ANNEAL_SCHEDULE,
    build_config as _build_config,
    run_condition,
)


def build_config(context: RunContext):
    return _build_config(context, 2, entropy_coeff=ENTROPY_ANNEAL_SCHEDULE)


def run(context: RunContext):
    return run_condition(context, 2, entropy_coeff=ENTROPY_ANNEAL_SCHEDULE)
