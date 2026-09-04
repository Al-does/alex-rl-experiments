"""Variant 2 REINFORCE with true 32k collection batches and fixed lower LR."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.mess3_reward_state_action_symmetry_cycle_6.shared import (
    build_config as _build_config,
    run_condition,
)
from harness.context import RunContext

VARIANT = 2
LEARNING_RATE = 2e-4
NUM_ENV_RUNNERS = 4
NUM_ENVS_PER_ENV_RUNNER = 8
EXPECTED_COLLECTION_STEPS = 32_768


def build_config(context: RunContext) -> PPOConfig:
    """Build the fixed-LR arm, preserving cheap local smoke resources."""
    config = _build_config(context, VARIANT).training(lr=LEARNING_RATE)
    if context.smoke:
        return config
    return config.env_runners(
        num_env_runners=NUM_ENV_RUNNERS,
        num_envs_per_env_runner=NUM_ENVS_PER_ENV_RUNNER,
    )


def _config_builder(context: RunContext, variant: int) -> PPOConfig:
    if variant != VARIANT:
        raise ValueError(f"this experiment only supports variant {VARIANT}")
    return build_config(context)


def run(context: RunContext):
    return run_condition(
        context,
        VARIANT,
        config_builder=_config_builder,
        recipe_overrides={
            "experiment_arm": "small_batch_lr_2e4",
            "learning_rate": LEARNING_RATE,
            "collection_geometry": {
                "num_env_runners": NUM_ENV_RUNNERS,
                "num_envs_per_env_runner": NUM_ENVS_PER_ENV_RUNNER,
                "episode_length": 1024,
                "expected_collection_steps": EXPECTED_COLLECTION_STEPS,
            },
        },
    )
