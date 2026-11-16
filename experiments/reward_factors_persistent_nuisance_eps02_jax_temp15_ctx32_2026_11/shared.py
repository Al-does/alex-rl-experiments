"""Persistent-nuisance v2 factors, PR #213 training, 32-frame transformer context.

CONTEXT_LEN only affects the transformer; memoryless and GRU specs ignore it.
"""
from experiments.reward_factors_persistent_nuisance_jax_2026_11 import shared as base
from experiments.belief_quotient_persistent_nuisance_jax_temp15_2026_11.shared import (
    ENTROPY_SCHEDULE, TEMPERATURE, TOTAL_ENV_STEPS,
)

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment

STUDY = "reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11"
ARCHITECTURE = "transformer"
ARCHITECTURES = ("transformer", "memoryless", "gru128", "gru92")
CONTEXT_LEN = 32
EXPLORATION_EPSILON = 0.0


def run_condition(context, *, condition, architecture=ARCHITECTURE, study=STUDY,
                  exploration_epsilon=EXPLORATION_EPSILON):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    return base.run_condition(
        context, condition=condition, architecture=architecture, study=study,
        total_env_steps=TOTAL_ENV_STEPS, entropy_schedule_points=ENTROPY_SCHEDULE,
        temperature=TEMPERATURE, environment=environment,
        exploration_epsilon=exploration_epsilon, context_len=CONTEXT_LEN,
    )
