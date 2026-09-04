"""Variant 2 REINFORCE with small-batch updates, lr=2e-4, and entropy bonus."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.mess3_reward_state_action_symmetry_cycle_6.shared import (
    run_condition,
)
from experiments.mess3_reward_state_action_symmetry_cycle_6.variant_2_small_batch_lr_2e4.experiment import (
    EXPECTED_COLLECTION_STEPS,
    LEARNING_RATE,
    NUM_ENVS_PER_ENV_RUNNER,
    NUM_ENV_RUNNERS,
    VARIANT,
    build_config as _build_small_batch_config,
)
from harness.context import RunContext

ENTROPY_COEFF = 0.01


def build_config(context: RunContext) -> PPOConfig:
    """Small-batch REINFORCE with a modest fixed entropy coefficient."""
    return _build_small_batch_config(context).training(
        entropy_coeff=ENTROPY_COEFF,
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
            "experiment_arm": "small_batch_lr_2e4_entropy",
            "learning_rate": LEARNING_RATE,
            "entropy_coeff": ENTROPY_COEFF,
            "entropy_regularization_rationale": (
                "Fixed 0.01 bonus on policy entropy to encourage exploration "
                "without dominating the Monte Carlo REINFORCE gradient; "
                "below the 0.03 PPO convention used elsewhere in this repo."
            ),
            "collection_geometry": {
                "num_env_runners": NUM_ENV_RUNNERS,
                "num_envs_per_env_runner": NUM_ENVS_PER_ENV_RUNNER,
                "episode_length": 1024,
                "expected_collection_steps": EXPECTED_COLLECTION_STEPS,
            },
        },
    )
