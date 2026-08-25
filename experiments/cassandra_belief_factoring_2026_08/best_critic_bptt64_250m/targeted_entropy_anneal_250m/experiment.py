"""Anneal entropy from 0.03 to 0.008 over the next 250M targeted steps."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.continuation_shared import (
    ANNEAL_DURATION_ENV_STEPS,
    ANNEAL_FINAL_ENTROPY,
    ANNEAL_LIFETIME_ENV_STEPS,
    ENTROPY_ANNEAL_SCHEDULE,
    build_anneal_config,
    run_continuation,
)
from harness.context import RunContext


CONDITION = "best_critic_bptt64_targeted_entropy_anneal_250m"
HYPOTHESIS = (
    "Resume the completed targeted best-critic BPTT-64 agents and anneal "
    f"entropy over {ANNEAL_DURATION_ENV_STEPS:,} steps down to "
    f"{ANNEAL_FINAL_ENTROPY}."
)


def build_config(context: RunContext) -> PPOConfig:
    return build_anneal_config(context)


def run(context: RunContext):
    return run_continuation(
        context,
        condition=CONDITION,
        hypothesis=HYPOTHESIS,
        lifetime_env_steps=ANNEAL_LIFETIME_ENV_STEPS,
        config_builder=build_anneal_config,
        apply_entropy_anneal=True,
        extra_recipe={
            "entropy_anneal_schedule": ENTROPY_ANNEAL_SCHEDULE,
            "entropy_anneal_duration_env_steps": ANNEAL_DURATION_ENV_STEPS,
        },
    )


__all__ = ["CONDITION", "HYPOTHESIS", "build_config", "run"]
