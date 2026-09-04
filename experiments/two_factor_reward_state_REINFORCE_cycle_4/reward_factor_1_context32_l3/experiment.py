"""One-rewarding REINFORCE with context_len=32 and n_layers=3."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.two_factor_reward_state_REINFORCE_cycle_4.shared import (
    CONTEXT32_L3_MODEL_CONFIG,
    LEARNING_RATE,
    TRAIN_BATCH_SIZE as DEFAULT_TRAIN_BATCH,
    build_config as _build_config,
    run_condition,
)
from harness.context import RunContext

CONDITION = "reward_factor_1"
LOOKBACK = 96
# 16 env runners OOM on RTX 4090 at lookback 96 (~22GB during rollout sync).
NUM_ENV_RUNNERS = 4
NUM_ENVS_PER_ENV_RUNNER = 8


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
            "experiment_arm": "reward_factor_1_context32_l3",
            "learning_rate": LEARNING_RATE,
            "collection_geometry": {
                "num_env_runners": NUM_ENV_RUNNERS,
                "num_envs_per_env_runner": NUM_ENVS_PER_ENV_RUNNER,
                "episode_length": 1024,
                "expected_collection_steps": DEFAULT_TRAIN_BATCH,
            },
            "architecture_rationale": (
                "Cycle-4 one-rewarding task with context_len=32 and n_layers=3 "
                "(lookback 96). Uses 4x8 env runners (not 16) because 16-runner "
                "rollout sync OOMs on RTX 4090 at this lookback; standard lr 4.2e-4."
            ),
        },
    )
