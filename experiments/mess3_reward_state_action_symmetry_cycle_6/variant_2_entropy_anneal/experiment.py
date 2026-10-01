"""Variant 2 REINFORCE with entropy coeff 0.03 annealed to 0 over the last 1M steps."""

from __future__ import annotations

from ray.rllib.algorithms.ppo import PPOConfig

from experiments.mess3_reward_state_action_symmetry_cycle_6.shared import (
    BASE_MODEL_CONFIG,
    _resolve_step_target,
    build_config as _build_config,
    entropy_coeff_schedule,
    run_condition,
)
from harness.context import RunContext

VARIANT = 2
ENTROPY_COEFF = 0.03
ENTROPY_ANNEAL_TAIL_STEPS = 1_000_000
MODEL_CONFIG = BASE_MODEL_CONFIG


def build_config(context: RunContext) -> PPOConfig:
    return _build_config(
        context,
        VARIANT,
        model_config=MODEL_CONFIG,
        entropy_coeff=entropy_coeff_schedule(
            ENTROPY_COEFF,
            _resolve_step_target(context),
            ENTROPY_ANNEAL_TAIL_STEPS,
        ),
    )


def run(context: RunContext):
    return run_condition(
        context,
        VARIANT,
        model_config=MODEL_CONFIG,
        entropy_coeff=ENTROPY_COEFF,
        entropy_anneal_tail_steps=ENTROPY_ANNEAL_TAIL_STEPS,
        recipe_overrides={
            "experiment_arm": "variant_2_entropy_anneal",
            "sampling_temperature": 1.0,
            "entropy_schedule_semantics": (
                "entropy coefficient held at 0.03, then linearly annealed to "
                "0.0 over the final 1M env steps; schedule advances on "
                "num_env_steps_sampled_lifetime"
            ),
        },
    )
