"""Train targeted PPO with value clipping expanded from 10 to 100."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_interventions_5m.shared import (
    build_config as build_intervention_config,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_interventions_5m.shared import (
    run_intervention,
)
from harness.context import RunContext


INTERVENTION = "vf_clip_100"
HYPOTHESIS = (
    "A wider value-function clipping range reduces critic underfitting when "
    "returns move farther than the baseline clip of 10."
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
