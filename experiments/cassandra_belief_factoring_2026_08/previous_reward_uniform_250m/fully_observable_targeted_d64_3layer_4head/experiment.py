"""250M-step fully observable targeted PPO with previous reward and uniform starts."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.cassandra_belief_factoring_2026_08.previous_reward_uniform_250m.shared import (
    build_config as build_recipe_config,
    run_recipe,
)
from harness.context import RunContext


D_MODEL = 64
ACTION_SCOPE = "targeted"
OBSERVATION_VARIANT = "state"
CONDITION = "previous_reward_uniform_250m_fully_observable_targeted_d64_3layer_4head"


def build_config(context: RunContext) -> PPOConfig:
    return build_recipe_config(
        context,
        d_model=D_MODEL,
        action_scope=ACTION_SCOPE,
        observation_variant=OBSERVATION_VARIANT,
    )


def run(context: RunContext):
    return run_recipe(
        context,
        d_model=D_MODEL,
        condition=CONDITION,
        action_scope=ACTION_SCOPE,
        observation_variant=OBSERVATION_VARIANT,
    )


__all__ = [
    "ACTION_SCOPE",
    "CONDITION",
    "D_MODEL",
    "OBSERVATION_VARIANT",
    "build_config",
    "run",
]
