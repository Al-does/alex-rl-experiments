"""One-rewarding REINFORCE with context_len=32 and n_layers=3."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.two_factor_reward_state_REINFORCE_cycle_4.shared import (
    CONTEXT32_L3_MODEL_CONFIG,
    build_config as _build_config,
    run_condition,
)
from harness.context import RunContext

CONDITION = "reward_factor_1"
LOOKBACK = 96
# 32k learner batches OOM on RTX 4090 with lookback 96; 16k fits with 16 runners.
TRAIN_BATCH_SIZE = 16_384


def build_config(context: RunContext) -> PPOConfig:
    return _build_config(
        context,
        CONDITION,
        model_config=CONTEXT32_L3_MODEL_CONFIG,
        train_batch_size=TRAIN_BATCH_SIZE,
    )


def run(context: RunContext):
    return run_condition(
        context,
        CONDITION,
        config_builder=build_config,
        model_config=CONTEXT32_L3_MODEL_CONFIG,
        recipe_overrides={
            "experiment_arm": "reward_factor_1_context32_l3",
            "architecture_rationale": (
                "Cycle-4 one-rewarding task with context_len=32 and n_layers=3 "
                "(lookback 96 vs default 40); standard 16-runner collection. "
                "Learner batch 16k (not 32k) to fit RTX 4090 memory at lookback 96."
            ),
            "train_batch_size_per_learner": TRAIN_BATCH_SIZE,
        },
    )
