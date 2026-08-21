"""Train targeted PPO with GAE lambda increased from 0.95 to 0.98."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_interventions_5m.shared import (
    build_config as build_intervention_config,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_interventions_5m.shared import (
    run_intervention,
)
from harness.context import RunContext


INTERVENTION = "lambda_098"
HYPOTHESIS = (
    "Longer-horizon advantage estimation improves credit assignment for "
    "maintenance actions whose benefits arrive after delayed degradation."
)


def build_config(context: RunContext) -> PPOConfig:
    return build_intervention_config(context, intervention=INTERVENTION)


def run(context: RunContext):
    return run_intervention(
        context,
        intervention=INTERVENTION,
        hypothesis=HYPOTHESIS,
    )


__all__ = ["build_config", "run"]
