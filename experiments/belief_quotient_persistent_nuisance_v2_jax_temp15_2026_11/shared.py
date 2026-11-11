"""Persistent-nuisance v2 with the PR #213 training settings."""
from experiments.belief_quotient_persistent_nuisance_jax_2026_11 import shared as base
from experiments.belief_quotient_persistent_nuisance_jax_temp15_2026_11.shared import (
    ENTROPY_SCHEDULE, TEMPERATURE, TOTAL_ENV_STEPS,
)
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import environment

STUDY = "belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11"


def run_condition(context, *, variant, architecture):
    return base.run_condition(
        context, variant=variant, architecture=architecture, study=STUDY,
        total_env_steps=TOTAL_ENV_STEPS, entropy_schedule_points=ENTROPY_SCHEDULE,
        temperature=TEMPERATURE, environment=environment,
    )
