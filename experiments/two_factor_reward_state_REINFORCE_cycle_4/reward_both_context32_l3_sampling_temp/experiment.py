"""Both-rewarding context32/l3 REINFORCE with temperature-scaled multinomial sampling."""

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
SAMPLING_TEMPERATURE = 1.5
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
            "experiment_arm": "reward_both_context32_l3_sampling_temp",
            "target_agent_steps": TARGET_AGENT_STEPS,
            "learning_rate": LEARNING_RATE,
            "entropy_coeff": 0.0,
            "sampling_temperature": SAMPLING_TEMPERATURE,
            "sampling_temperature_rationale": (
                "Scale policy logits by 1/T in rollout sampling and train-time "
                "log-prob evaluation so multinomial draws stay on-policy; "
                "T=1.5 adds moderate exploration without an entropy bonus."
            ),
            "collection_geometry": {
                "num_env_runners": NUM_ENV_RUNNERS,
                "num_envs_per_env_runner": NUM_ENVS_PER_ENV_RUNNER,
                "episode_length": 1024,
                "expected_collection_steps": DEFAULT_TRAIN_BATCH,
            },
            "architecture_rationale": (
                "Same context32/l3 both-rewarding setup as reward_both_context32_l3 "
                "with sampling_temperature=1.5 for on-policy multinomial exploration."
            ),
        },
    )
