"""Rerun visible previous reward with the best vf-clip critic settings."""

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_previous_reward_best_critic_5m.shared import (
    build_config as build_condition_config,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_previous_reward_best_critic_5m.shared import (
    run_condition,
)
from harness.context import RunContext


def build_config(context: RunContext) -> PPOConfig:
    return build_condition_config(context)


def run(context: RunContext):
    return run_condition(context)


__all__ = ["build_config", "run"]
