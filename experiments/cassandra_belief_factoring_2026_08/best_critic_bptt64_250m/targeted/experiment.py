"""250M-step targeted PPO with best critic and BPTT 64."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.shared import (
    build_config as build_recipe_config,
    run_recipe,
)
from harness.context import RunContext


ACTION_SCOPE = "targeted"
CONDITION = "best_critic_bptt64_250m_targeted"


def build_config(context: RunContext) -> PPOConfig:
    return build_recipe_config(context, action_scope=ACTION_SCOPE)


def run(context: RunContext):
    return run_recipe(
        context,
        action_scope=ACTION_SCOPE,
        condition=CONDITION,
    )


__all__ = ["ACTION_SCOPE", "CONDITION", "build_config", "run"]
