"""250M-step targeted PPO with best critic, BPTT 64, and visible previous reward."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.shared import (
    build_config as build_recipe_config,
    run_recipe,
)
from harness.context import RunContext


ACTION_SCOPE = "targeted"
CONDITION = "best_critic_bptt64_250m_targeted_previous_reward"


def build_config(context: RunContext) -> PPOConfig:
    return build_recipe_config(
        context,
        action_scope=ACTION_SCOPE,
        previous_reward_visible=True,
    )


def run(context: RunContext):
    return run_recipe(
        context,
        action_scope=ACTION_SCOPE,
        condition=CONDITION,
        previous_reward_visible=True,
    )


__all__ = ["ACTION_SCOPE", "CONDITION", "build_config", "run"]
