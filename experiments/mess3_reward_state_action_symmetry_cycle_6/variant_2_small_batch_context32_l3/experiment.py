"""Variant 2 REINFORCE with small-batch updates and a shorter, deeper transformer."""

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

CONTEXT_LEN = 32
N_LAYERS = 3
CONTEXT32_L3_MODEL_CONFIG = {
    **BASE_MODEL_CONFIG,
    "context_len": CONTEXT_LEN,
    "n_layers": N_LAYERS,
}


def build_config(context: RunContext) -> PPOConfig:
    """Small-batch REINFORCE with context_len=32 and n_layers=3."""
    config = _build_shared_config(
        context,
        VARIANT,
        model_config=CONTEXT32_L3_MODEL_CONFIG,
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
            "experiment_arm": "small_batch_lr_2e4_context32_l3",
            "learning_rate": LEARNING_RATE,
            "model_config": CONTEXT32_L3_MODEL_CONFIG,
            "transformer_lookback": N_LAYERS * CONTEXT_LEN,
            "episode_length": 1024,
            "architecture_rationale": (
                "Shorter per-layer context (32 vs 10) with fewer layers (3 vs 4) "
                "yields lookback 96 instead of 40; episode_length stays 1024."
            ),
            "collection_geometry": {
                "num_env_runners": NUM_ENV_RUNNERS,
                "num_envs_per_env_runner": NUM_ENVS_PER_ENV_RUNNER,
                "episode_length": 1024,
                "expected_collection_steps": EXPECTED_COLLECTION_STEPS,
            },
        },
    )
