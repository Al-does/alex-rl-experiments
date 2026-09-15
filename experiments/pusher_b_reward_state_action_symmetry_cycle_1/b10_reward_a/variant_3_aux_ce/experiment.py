"""Variant 3 with a next-token cross-entropy auxiliary loss."""

from __future__ import annotations

from harness.context import RunContext

from experiments.pusher_b_reward_state_action_symmetry_cycle_1.shared import (
    build_config as _build_config,
    run_condition,
)

# Sized for the ~123-effective-core Vast host (EPYC 9554): leaves ~13
# cores for the driver, learner, and Ray overhead.
PREFERRED_ENV_RUNNERS = 110
# On-box benchmark measured ~4,590 env steps/s steady-state (57.8 s per
# 265k-step iteration); 154M steps matches the ~33,500 s wall-clock of
# run 20260914T093608Z-3e9fde2a.
TARGET_ENV_STEPS = 154_000_000


def build_config(context: RunContext):
    return _build_config(
        context,
        preset="b10",
        variant=3,
        reward_state="A",
        preferred_env_runners=PREFERRED_ENV_RUNNERS,
        next_token_aux=True,
    )


def run(context: RunContext):
    return run_condition(
        context,
        preset="b10",
        variant=3,
        reward_state="A",
        preferred_env_runners=PREFERRED_ENV_RUNNERS,
        next_token_aux=True,
        total_env_steps=TARGET_ENV_STEPS,
    )
