"""Both-rewarding context32/l3 REINFORCE with higher sampling temperature (T=2.0)."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.two_factor_reward_state_REINFORCE_cycle_4.shared import (
    CONTEXT32_L3_MODEL_CONFIG,
    LEARNING_RATE,
    TRAIN_BATCH_SIZE as DEFAULT_TRAIN_BATCH,
    build_config as _build_config,
    run_condition,
    write_budget_spec,
)
from harness.context import RunContext

CONDITION = "reward_both"
TARGET_AGENT_STEPS = 30_000_000
LOOKBACK = 96
NUM_ENV_RUNNERS = 4
NUM_ENVS_PER_ENV_RUNNER = 8
SAMPLING_TEMPERATURE = 2.0
MODEL_CONFIG = {
    **CONTEXT32_L3_MODEL_CONFIG,
    "sampling_temperature": SAMPLING_TEMPERATURE,
}


def build_config(context: RunContext) -> PPOConfig:
    return _build_config(
        context,
        CONDITION,
        model_config=MODEL_CONFIG,
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
        model_config=MODEL_CONFIG,
        recipe_overrides={
            "experiment_arm": "reward_both_context32_l3_sampling_temp_2p0",
            "target_agent_steps": TARGET_AGENT_STEPS,
            "skip_checkpoint_probes": True,
            "track_occupancy": True,
            "learning_rate": LEARNING_RATE,
            "entropy_coeff": 0.0,
            "sampling_temperature": SAMPLING_TEMPERATURE,
            "sampling_temperature_rationale": (
                "Higher on-policy exploration than the T=1.5 arm: logits scaled "
                "by 1/T in rollout sampling and train-time log-prob evaluation."
            ),
            "collection_geometry": {
                "num_env_runners": NUM_ENV_RUNNERS,
                "num_envs_per_env_runner": NUM_ENVS_PER_ENV_RUNNER,
                "episode_length": 1024,
                "expected_collection_steps": DEFAULT_TRAIN_BATCH,
            },
            "architecture_rationale": (
                "Same context32/l3 both-rewarding setup as reward_both_context32_l3 "
                "with sampling_temperature=2.0. Tracks reward occupancy from "
                "training rollouts; no probe batteries."
            ),
        },
        skip_checkpoint_probes=True,
        track_occupancy=True,
    )
