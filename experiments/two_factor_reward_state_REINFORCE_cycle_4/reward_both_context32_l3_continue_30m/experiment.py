"""Continue reward_both context32/l3 from a prior checkpoint for another 30M steps."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.two_factor_reward_state_REINFORCE_cycle_4.shared import (
    CONTEXT32_L3_MODEL_CONFIG,
    LEARNING_RATE,
    STEP_CHECKPOINT_INTERVAL_5M,
    TRAIN_BATCH_SIZE as DEFAULT_TRAIN_BATCH,
    build_config as _build_config,
    run_condition,
    write_continuation_spec,
)
from harness.context import RunContext

CONDITION = "reward_both"
PRIOR_RUN_ID = "20260904T203930Z-165d2800"
PRIOR_AGENT_STEPS = 30_000_000
TARGET_AGENT_STEPS = 60_000_000
LOOKBACK = 96
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
    if not context.smoke and context.resume_from is None:
        raise ValueError(
            "reward_both_context32_l3_continue_30m requires --resume-from "
            "pointing at the prior run's latest step checkpoint"
        )
    if not context.smoke:
        write_continuation_spec(
            context,
            target_agent_steps=TARGET_AGENT_STEPS,
            step_checkpoint_interval=STEP_CHECKPOINT_INTERVAL_5M,
            prior_run_id=PRIOR_RUN_ID,
            prior_agent_steps=PRIOR_AGENT_STEPS,
        )
    return run_condition(
        context,
        CONDITION,
        config_builder=build_config,
        model_config=CONTEXT32_L3_MODEL_CONFIG,
        recipe_overrides={
            "experiment_arm": "reward_both_context32_l3_continue_30m",
            "target_agent_steps": TARGET_AGENT_STEPS,
            "prior_run_id": PRIOR_RUN_ID,
            "prior_agent_steps": PRIOR_AGENT_STEPS,
            "step_checkpoint_interval": STEP_CHECKPOINT_INTERVAL_5M,
            "learning_rate": LEARNING_RATE,
            "collection_geometry": {
                "num_env_runners": NUM_ENV_RUNNERS,
                "num_envs_per_env_runner": NUM_ENVS_PER_ENV_RUNNER,
                "episode_length": 1024,
                "expected_collection_steps": DEFAULT_TRAIN_BATCH,
            },
            "architecture_rationale": (
                "Continue the completed 30M reward_both context32/l3 run for "
                "another 30M env steps from its latest step checkpoint. "
                "Step checkpoints every 5M env steps during continuation."
            ),
        },
    )
