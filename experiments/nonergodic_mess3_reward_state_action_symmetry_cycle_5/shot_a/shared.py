from __future__ import annotations

from functools import partial

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.shared import (
    build_config as build_base_config,
    resolved_recipe as resolve_base_recipe,
    run_condition as run_base_condition,
)
from harness.context import RunContext


LEARNING_RATE_SCHEDULE = [
    [0, 5e-5],
    [5_000_000, 5e-5],
    [10_000_000, 1e-5],
]
ENTROPY_COEFF_SCHEDULE = [
    [0, 0.01],
    [6_000_000, 0.01],
    [8_000_000, 0.0],
]
VALUE_LOSS_COEFF = 0.25


def build_config(context: RunContext, variant: int) -> PPOConfig:
    return build_base_config(context, variant).training(
        lr=LEARNING_RATE_SCHEDULE,
        vf_loss_coeff=VALUE_LOSS_COEFF,
        entropy_coeff=ENTROPY_COEFF_SCHEDULE,
    )


def resolved_recipe(context: RunContext, variant: int) -> dict[str, object]:
    recipe = resolve_base_recipe(context, variant)
    recipe.update(
        {
            "condition": f"shot_a_variant_{variant}",
            "training_profile": (
                "PR 127 Shot A optimizer schedules and loss coefficients"
            ),
            "learning_rate": LEARNING_RATE_SCHEDULE,
            "value_loss_coeff": VALUE_LOSS_COEFF,
            "entropy_coeff": ENTROPY_COEFF_SCHEDULE,
        }
    )
    return recipe


def run_condition(context: RunContext, variant: int) -> dict[str, object]:
    return run_base_condition(
        context,
        variant,
        condition=f"shot_a_variant_{variant}",
        config_builder=partial(build_config, variant=variant),
        recipe_builder=partial(resolved_recipe, variant=variant),
    )
