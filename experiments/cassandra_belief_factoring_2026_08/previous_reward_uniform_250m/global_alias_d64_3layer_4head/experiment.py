"""250M-step PPO: previous reward, uniform starts, global-alias actions, dim-64."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.cassandra_belief_factoring_2026_08.previous_reward_uniform_250m.shared import (
    build_config as build_recipe_config,
    run_recipe,
)
from harness.context import RunContext


D_MODEL = 64
ACTION_SCOPE = "global_aliases"
CONDITION = "previous_reward_uniform_250m_global_alias_d64_3layer_4head"


def build_config(context: RunContext) -> PPOConfig:
    return build_recipe_config(
        context,
        d_model=D_MODEL,
        action_scope=ACTION_SCOPE,
    )


def run(context: RunContext):
    return run_recipe(
        context,
        d_model=D_MODEL,
        condition=CONDITION,
        action_scope=ACTION_SCOPE,
    )


__all__ = ["ACTION_SCOPE", "CONDITION", "D_MODEL", "build_config", "run"]
