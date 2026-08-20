"""Train targeted PPO with the previous reward in the policy input."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_interventions_5m.shared import (
    build_config as build_intervention_config,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_interventions_5m.shared import (
    run_intervention,
)
from harness.context import RunContext


INTERVENTION = "previous_reward"
HYPOTHESIS = (
    "The state-informative operate reward helps the recurrent policy update "
    "its latent component-health belief between explicit inspections."
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
