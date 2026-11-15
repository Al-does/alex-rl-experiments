"""Persistent-nuisance ladder transformer arms rerun from scratch for 2B steps with a
constant (never annealed) entropy bonus of 0.01 or 0.005, sampling temperature 1.0 and
no epsilon-greedy exploration; otherwise identical to the 2B baseline (entropy 0.015)."""

from experiments.belief_quotient_persistent_nuisance_jax_2026_11 import shared as base
from experiments.belief_quotient_persistent_nuisance_jax_2b_2026_11 import shared as budget

STUDY = "belief_quotient_persistent_nuisance_jax_entropy_2026_11"
TOTAL_ENV_STEPS = budget.TOTAL_ENV_STEPS
TEMPERATURE = 1.0
EPSILON = 0.0
ENTROPY_COEFFS = {"ent0p01": 0.01, "ent0p005": 0.005}


def run_condition(context, *, variant, architecture, entropy):
    return base.run_condition(
        context,
        variant=variant,
        architecture=architecture,
        total_env_steps=TOTAL_ENV_STEPS,
        study=STUDY,
        entropy_schedule_points=((0, ENTROPY_COEFFS[entropy]),),
        temperature=TEMPERATURE,
        epsilon=EPSILON,
    )
