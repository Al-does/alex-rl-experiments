"""Variant 2 REINFORCE with small-batch updates, lr=2e-4, and sampling temperature."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.mess3_reward_state_action_symmetry_cycle_6.shared import (
    BASE_MODEL_CONFIG,
    build_config as _build_shared_config,
    run_condition,
)
from experiments.mess3_reward_state_action_symmetry_cycle_6.variant_2_small_batch_lr_2e4.experiment import (
    EXPECTED_COLLECTION_STEPS,
    LEARNING_RATE,
    NUM_ENVS_PER_ENV_RUNNER,
    NUM_ENV_RUNNERS,
    VARIANT,
)
from harness.context import RunContext

# T>1 softens the categorical policy at rollout and train time (logits / T).
# 1.5 is a moderate increase in stochasticity without an entropy loss term.
SAMPLING_TEMPERATURE = 1.5


def build_config(context: RunContext) -> PPOConfig:
    """Small-batch REINFORCE with on-policy temperature-scaled multinomial sampling."""
    model_config = {
        **BASE_MODEL_CONFIG,
        "sampling_temperature": SAMPLING_TEMPERATURE,
    }
    config = _build_shared_config(
        context,
        VARIANT,
        model_config=model_config,
    ).training(lr=LEARNING_RATE)
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
            "experiment_arm": "small_batch_lr_2e4_sampling_temp",
            "learning_rate": LEARNING_RATE,
            "entropy_coeff": 0.0,
            "sampling_temperature": SAMPLING_TEMPERATURE,
            "sampling_temperature_rationale": (
                "Scale policy logits by 1/T in both rollout sampling and "
                "train-time log-prob evaluation so multinomial draws stay "
                "on-policy; T=1.5 adds moderate exploration without an "
                "entropy bonus."
            ),
            "model_config": {
                **BASE_MODEL_CONFIG,
                "sampling_temperature": SAMPLING_TEMPERATURE,
            },
            "collection_geometry": {
                "num_env_runners": NUM_ENV_RUNNERS,
                "num_envs_per_env_runner": NUM_ENVS_PER_ENV_RUNNER,
                "episode_length": 1024,
                "expected_collection_steps": EXPECTED_COLLECTION_STEPS,
            },
        },
    )
