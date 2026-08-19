"""Train and probe cardinality-matched aliases of global maintenance actions."""

from ray.rllib.algorithms.ppo import PPOConfig

from harness.context import RunContext

from experiments.cassandra_belief_factoring_2026_08.shared import (
    build_config as build_shared_config,
)
from experiments.cassandra_belief_factoring_2026_08.shared import run_condition


ACTION_SCOPE = "global_aliases"
CONDITION = "global_alias_actions_transformer_ppo"
HYPOTHESIS = (
    "Matching the targeted condition's ten-action vocabulary without changing "
    "global maintenance semantics should preserve the pressure for a coarse, "
    "permutation-invariant machine representation."
)


def build_config(context: RunContext) -> PPOConfig:
    """Build the cardinality-matched global-action PPO recipe."""

    return build_shared_config(context, action_scope=ACTION_SCOPE)


def run(context: RunContext):
    """Train and probe the global-alias cardinality control."""

    return run_condition(
        context,
        action_scope=ACTION_SCOPE,
        condition=CONDITION,
        hypothesis=HYPOTHESIS,
    )


__all__ = ["build_config", "run"]
