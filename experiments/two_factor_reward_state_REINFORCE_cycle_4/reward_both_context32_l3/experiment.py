"""Both-rewarding REINFORCE with context_len=32 and n_layers=3 (30M steps)."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.two_factor_reward_state_REINFORCE_cycle_4.shared import (
    CONTEXT32_L3_MODEL_CONFIG,
    build_config as _build_config,
    run_condition,
    write_budget_spec,
)
from harness.context import RunContext

CONDITION = "reward_both"
TARGET_AGENT_STEPS = 30_000_000
LOOKBACK = 96


def build_config(context: RunContext) -> PPOConfig:
    return _build_config(context, CONDITION, model_config=CONTEXT32_L3_MODEL_CONFIG)


def run(context: RunContext):
    if not context.smoke:
        write_budget_spec(context, TARGET_AGENT_STEPS)
    return run_condition(
        context,
        CONDITION,
        config_builder=build_config,
        model_config=CONTEXT32_L3_MODEL_CONFIG,
        recipe_overrides={
            "experiment_arm": "reward_both_context32_l3",
            "target_agent_steps": TARGET_AGENT_STEPS,
            "architecture_rationale": (
                "Cycle-4 both-rewarding task with context_len=32 and n_layers=3 "
                "trained to 30M env steps."
            ),
        },
    )
