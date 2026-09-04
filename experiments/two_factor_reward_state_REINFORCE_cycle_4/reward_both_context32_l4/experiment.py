"""Both-rewarding REINFORCE with context_len=32 and n_layers=4 (30M steps)."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.two_factor_reward_state_REINFORCE_cycle_4.shared import (
    CONTEXT32_L4_MODEL_CONFIG,
    LEARNING_RATE,
    TRAIN_BATCH_SIZE as DEFAULT_TRAIN_BATCH,
    build_config as _build_config,
    run_condition,
    write_budget_spec,
)
from harness.context import RunContext

CONDITION = "reward_both"
TARGET_AGENT_STEPS = 30_000_000
LOOKBACK = 128
NUM_ENV_RUNNERS = 4
NUM_ENVS_PER_ENV_RUNNER = 8


def build_config(context: RunContext) -> PPOConfig:
    return _build_config(
        context,
        CONDITION,
        model_config=CONTEXT32_L4_MODEL_CONFIG,
        learning_rate=LEARNING_RATE,
        num_env_runners=NUM_ENV_RUNNERS,
        num_envs_per_env_runner=NUM_ENVS_PER_ENV_RUNNER,
        track_occupancy=True,
    )


def run(context: RunContext):
    if not context.smoke:
        write_budget_spec(context, TARGET_AGENT_STEPS)
    return run_condition(
        context,
        CONDITION,
        config_builder=build_config,
        model_config=CONTEXT32_L4_MODEL_CONFIG,
        recipe_overrides={
            "experiment_arm": "reward_both_context32_l4",
            "target_agent_steps": TARGET_AGENT_STEPS,
            "skip_checkpoint_probes": True,
            "track_occupancy": True,
            "learning_rate": LEARNING_RATE,
            "collection_geometry": {
                "num_env_runners": NUM_ENV_RUNNERS,
                "num_envs_per_env_runner": NUM_ENVS_PER_ENV_RUNNER,
                "episode_length": 1024,
                "expected_collection_steps": DEFAULT_TRAIN_BATCH,
            },
            "architecture_rationale": (
                "Cycle-4 both-rewarding task with context_len=32 and n_layers=4 "
                "trained to 30M env steps. Same 4x8 collection geometry as the "
                "context32/l3 arms. Tracks reward occupancy from training rollouts; "
                "no probe batteries."
            ),
        },
        skip_checkpoint_probes=True,
        track_occupancy=True,
    )
