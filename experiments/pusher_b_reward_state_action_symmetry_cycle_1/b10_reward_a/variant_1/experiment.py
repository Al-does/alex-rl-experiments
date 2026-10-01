from harness.context import RunContext

from experiments.pusher_b_reward_state_action_symmetry_cycle_1.shared import (
    build_config as _build_config,
    run_condition,
)


def build_config(context: RunContext):
    return _build_config(
        context,
        preset="b10",
        variant=1,
        reward_state="A",
    )


def run(context: RunContext):
    return run_condition(
        context,
        preset="b10",
        variant=1,
        reward_state="A",
    )
