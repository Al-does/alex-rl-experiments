"""Train PPG with component-targeted maintenance actions."""

from harness.context import RunContext
from learners import PPGConfig

from experiments.cassandra_belief_factoring_2026_08.ppg_5m.shared import (
    build_config as build_shared_config,
)
from experiments.cassandra_belief_factoring_2026_08.ppg_5m.shared import (
    run_condition,
)


ACTION_SCOPE = "targeted"
CONDITION = "targeted_actions_transformer_ppg"


def build_config(context: RunContext) -> PPGConfig:
    return build_shared_config(context, action_scope=ACTION_SCOPE)


def run(context: RunContext):
    return run_condition(
        context,
        action_scope=ACTION_SCOPE,
        condition=CONDITION,
    )


__all__ = ["build_config", "run"]
