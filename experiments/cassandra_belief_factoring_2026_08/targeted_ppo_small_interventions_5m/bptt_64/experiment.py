"""Train targeted PPO with BPTT and transformer context reduced to 64."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_interventions_5m.shared import (
    build_config as build_intervention_config,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_interventions_5m.shared import (
    run_intervention,
)
from harness.context import RunContext


INTERVENTION = "bptt_64"
HYPOTHESIS = (
    "A 64-step credit-assignment and attention window is sufficient for the "
    "task while reducing recurrent learner padding and attention cost."
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
