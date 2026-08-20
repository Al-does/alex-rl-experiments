"""Train and probe a transformer with component-targeted maintenance actions."""

from ray.rllib.algorithms.ppo import PPOConfig

from harness.context import RunContext

from experiments.cassandra_belief_factoring_2026_08.shared import (
    build_config as build_shared_config,
)
from experiments.cassandra_belief_factoring_2026_08.shared import run_condition


ACTION_SCOPE = "targeted"
CONDITION = "targeted_actions_transformer_ppo"
HYPOTHESIS = (
    "Component-addressable repair and replacement create pressure to preserve "
    "labeled component beliefs beyond permutation-invariant machine health."
)


def build_config(context: RunContext) -> PPOConfig:
    """Build the component-targeted PPO recipe."""

    return build_shared_config(context, action_scope=ACTION_SCOPE)


def run(context: RunContext):
    """Train and probe the component-targeted condition."""

    return run_condition(
        context,
        action_scope=ACTION_SCOPE,
        condition=CONDITION,
        hypothesis=HYPOTHESIS,
    )


__all__ = ["build_config", "run"]
