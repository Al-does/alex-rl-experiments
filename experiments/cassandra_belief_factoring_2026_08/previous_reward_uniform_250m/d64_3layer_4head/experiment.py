"""250M-step targeted PPO: previous reward, uniform starts, dim-64 transformer."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.cassandra_belief_factoring_2026_08.previous_reward_uniform_250m.shared import (
    build_config as build_recipe_config,
    run_recipe,
)
from harness.context import RunContext


D_MODEL = 64
CONDITION = "previous_reward_uniform_250m_d64_3layer_4head"


def build_config(context: RunContext) -> PPOConfig:
    return build_recipe_config(context, d_model=D_MODEL)


def run(context: RunContext):
    return run_recipe(context, d_model=D_MODEL, condition=CONDITION)


__all__ = ["CONDITION", "D_MODEL", "build_config", "run"]
