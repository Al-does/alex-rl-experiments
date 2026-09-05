"""One-rewarding context32/l3 with token-only observations for 15M steps."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.two_factor_reward_state_PPO_cycle_2.process import (
    environment_config_without_last_action,
)
from experiments.two_factor_reward_state_REINFORCE_cycle_4.shared import (
    CONTEXT32_L3_MODEL_CONFIG,
    LEARNING_RATE,
    TRAIN_BATCH_SIZE as DEFAULT_TRAIN_BATCH,
    build_config as _build_config,
    run_condition,
    write_budget_spec,
)
from harness.context import RunContext

CONDITION = "reward_factor_1"
TARGET_AGENT_STEPS = 15_000_000
LOOKBACK = 96
NUM_ENV_RUNNERS = 4
NUM_ENVS_PER_ENV_RUNNER = 8
ENV_CONFIG = environment_config_without_last_action(CONDITION)
MODEL_CONFIG = {
    **CONTEXT32_L3_MODEL_CONFIG,
    "include_last_action_in_obs": False,
}


def build_config(context: RunContext) -> PPOConfig:
    return _build_config(
        context,
        CONDITION,
        model_config=MODEL_CONFIG,
        env_config=ENV_CONFIG,
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
            "experiment_arm": "reward_factor_1_context32_l3_no_action_15m",
            "target_agent_steps": TARGET_AGENT_STEPS,
            "learning_rate": LEARNING_RATE,
            "include_last_action_in_obs": False,
            "observation_rationale": (
                "Token-only observations (9-dim one-hot) without the previous-action "
                "block. Extends reward_factor_1_context32_l3 from 8M to 15M steps."
            ),
            "collection_geometry": {
                "num_env_runners": NUM_ENV_RUNNERS,
                "num_envs_per_env_runner": NUM_ENVS_PER_ENV_RUNNER,
                "episode_length": 1024,
                "expected_collection_steps": DEFAULT_TRAIN_BATCH,
            },
            "architecture_rationale": (
                "Cycle-4 one-rewarding task with context_len=32 and n_layers=3 "
                "(lookback 96), lr 4.2e-4, token-only observations, 15M env steps."
            ),
        },
    )
