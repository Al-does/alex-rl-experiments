"""Persistent-nuisance v2 temp15 ladder with epsilon-greedy exploration (epsilon 0.05)."""
from experiments.belief_quotient_persistent_nuisance_jax_2026_11 import shared as base
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import environment
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.shared import (
    ENTROPY_SCHEDULE, TEMPERATURE, TOTAL_ENV_STEPS,
)

STUDY = "belief_quotient_persistent_nuisance_v2_jax_temp15_eps05_2026_11"
EXPLORATION_EPSILON = 0.05


def run_condition(context, *, variant, architecture):
    return base.run_condition(
        context, variant=variant, architecture=architecture, study=STUDY,
        total_env_steps=TOTAL_ENV_STEPS, entropy_schedule_points=ENTROPY_SCHEDULE,
        temperature=TEMPERATURE, environment=environment,
        exploration_epsilon=EXPLORATION_EPSILON,
    )
