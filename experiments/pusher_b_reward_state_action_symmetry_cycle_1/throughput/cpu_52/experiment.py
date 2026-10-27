from __future__ import annotations

from harness.context import RunContext

from experiments.pusher_b_reward_state_action_symmetry_cycle_1.throughput.shared import (
    build_config as _build_config,
    run_condition,
)


MAX_ENV_RUNNERS = 52


def build_config(context: RunContext):
    return _build_config(context, max_env_runners=MAX_ENV_RUNNERS)


def run(context: RunContext):
    return run_condition(
        context,
        label="b10_reward_b_variant_2_cpu_52",
        max_env_runners=MAX_ENV_RUNNERS,
    )
