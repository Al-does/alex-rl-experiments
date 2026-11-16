"""Entropy-regularised SAC extension of the eps02 transformer study."""
import math

from experiments.belief_quotient_persistent_nuisance_jax_2026_11 import shared as base
from experiments.belief_quotient_persistent_nuisance_jax_temp15_2026_11.shared import (
    ENTROPY_SCHEDULE, TEMPERATURE, TOTAL_ENV_STEPS,
)
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_sac_2026_11 import sac

STUDY = "belief_quotient_persistent_nuisance_eps02_jax_temp15_sac_autoalpha_2026_11"
HYPERPARAMETERS = {
    "auto_entropy_tuning": True,
    "alpha_init": 1.0,
    "alpha_lr": 3e-4,
    "target_entropy": 0.3 * math.log(3),
    "lr": 3e-5,
    "critic_lr": 3e-4,
    "q_head_init": "zeros",
}


def training_backend(context):
    return sac.for_context(context, **HYPERPARAMETERS)


def run_condition(context, *, variant, architecture="transformer"):
    if architecture != "transformer":
        raise ValueError("this SAC study supports only transformer")
    return base.run_condition(
        context, variant=variant, architecture=architecture, study=STUDY,
        total_env_steps=TOTAL_ENV_STEPS, entropy_schedule_points=ENTROPY_SCHEDULE,
        temperature=TEMPERATURE, environment=environment,
        training_backend=training_backend(context),
    )
