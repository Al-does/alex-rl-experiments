"""Intermediate-budget PPO run for CUDA and learning-curve validation."""

from harness.context import RunContext

from experiments.mess3_token_guess_cycle_2.shared import (
    VALIDATION_ENV_STEPS,
    run_condition,
)


def run(context: RunContext):
    if context.smoke:
        raise ValueError("ppo_validation uses its fixed 131,072-step budget")
    return run_condition(
        context,
        "ppo",
        target_steps_override=VALIDATION_ENV_STEPS,
    )
