"""One-rewarding REINFORCE with context32/l3 and small-batch updates."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.two_factor_reward_state_REINFORCE_cycle_4.shared import (
    CONTEXT32_L3_MODEL_CONFIG,
    build_config as _build_config,
    run_condition,
)
from harness.context import RunContext

CONDITION = "reward_factor_1"
LEARNING_RATE = 2e-4
NUM_ENV_RUNNERS = 4
NUM_ENVS_PER_ENV_RUNNER = 8
EXPECTED_COLLECTION_STEPS = 32_768
LOOKBACK = 96


def build_config(context: RunContext) -> PPOConfig:
    return _build_config(
        context,
        CONDITION,
        model_config=CONTEXT32_L3_MODEL_CONFIG,
        learning_rate=LEARNING_RATE,
        num_env_runners=NUM_ENV_RUNNERS,
        num_envs_per_env_runner=NUM_ENVS_PER_ENV_RUNNER,
    )


def run(context: RunContext):
    return run_condition(
        context,
        CONDITION,
        config_builder=build_config,
        model_config=CONTEXT32_L3_MODEL_CONFIG,
        recipe_overrides={
            "experiment_arm": "reward_factor_1_context32_l3_small_batch",
            "learning_rate": LEARNING_RATE,
            "collection_geometry": {
                "num_env_runners": NUM_ENV_RUNNERS,
                "num_envs_per_env_runner": NUM_ENVS_PER_ENV_RUNNER,
                "episode_length": 1024,
                "expected_collection_steps": EXPECTED_COLLECTION_STEPS,
            },
            "architecture_rationale": (
                "Same context32/l3 transformer as reward_factor_1_context32_l3 "
                "with 4x8 env runners for more frequent 32k gradient updates."
            ),
        },
    )
