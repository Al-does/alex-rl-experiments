"""Variant 3 REINFORCE with categorical sampling temperature T=1.5."""

from __future__ import annotations

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.mess3_reward_state_action_symmetry_cycle_6.shared import (
    BASE_MODEL_CONFIG,
    build_config as _build_config,
    run_condition,
)
from harness.context import RunContext

VARIANT = 3
SAMPLING_TEMPERATURE = 1.5
MODEL_CONFIG = {
    **BASE_MODEL_CONFIG,
    "sampling_temperature": SAMPLING_TEMPERATURE,
}


def build_config(context: RunContext) -> PPOConfig:
    return _build_config(context, VARIANT, model_config=MODEL_CONFIG)


def run(context: RunContext):
    return run_condition(
        context,
        VARIANT,
        model_config=MODEL_CONFIG,
        recipe_overrides={
            "experiment_arm": "variant_3_sampling_temp",
            "sampling_temperature": SAMPLING_TEMPERATURE,
            "temperature_semantics": (
                "policy logits divided by 1.5 in rollout sampling and "
                "train-time log-probability evaluation"
            ),
        },
    )
