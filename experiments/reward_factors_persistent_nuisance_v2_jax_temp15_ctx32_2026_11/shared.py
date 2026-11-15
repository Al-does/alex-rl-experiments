"""Persistent-nuisance v2 factors, PR #213 training, transformer with 32-frame context."""
from experiments.reward_factors_persistent_nuisance_jax_2026_11 import shared as base
from experiments.reward_factors_persistent_nuisance_v2_jax_temp15_2026_11.shared import (
    ENTROPY_SCHEDULE, TEMPERATURE, TOTAL_ENV_STEPS, environment,
)

STUDY = "reward_factors_persistent_nuisance_v2_jax_temp15_ctx32_2026_11"
ARCHITECTURE = "transformer"
CONTEXT_LEN = 32
EXPLORATION_EPSILON = 0.0


def run_condition(context, *, condition, architecture=ARCHITECTURE, study=STUDY,
                  exploration_epsilon=EXPLORATION_EPSILON):
    if architecture != ARCHITECTURE:
        raise ValueError(f"architecture must be {ARCHITECTURE!r}")
    return base.run_condition(
        context, condition=condition, architecture=architecture, study=study,
        total_env_steps=TOTAL_ENV_STEPS, entropy_schedule_points=ENTROPY_SCHEDULE,
        temperature=TEMPERATURE, environment=environment,
        exploration_epsilon=exploration_epsilon, context_len=CONTEXT_LEN,
    )
