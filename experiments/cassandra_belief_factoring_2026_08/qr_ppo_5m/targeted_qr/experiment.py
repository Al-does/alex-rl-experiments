"""Train fixed-quantile PPO with component-targeted maintenance actions."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.cassandra_belief_factoring_2026_08.qr_ppo_5m.shared import (
    build_config as build_shared_config,
)
from experiments.cassandra_belief_factoring_2026_08.qr_ppo_5m.shared import (
    run_condition,
)
from harness.context import RunContext


ACTION_SCOPE = "targeted"


def build_config(context: RunContext) -> PPOConfig:
    """Build the component-targeted QR-PPO recipe."""

    return build_shared_config(context, action_scope=ACTION_SCOPE)


def run(context: RunContext):
    """Train the component-targeted condition."""

    return run_condition(context, action_scope=ACTION_SCOPE)


__all__ = ["ACTION_SCOPE", "build_config", "run"]
