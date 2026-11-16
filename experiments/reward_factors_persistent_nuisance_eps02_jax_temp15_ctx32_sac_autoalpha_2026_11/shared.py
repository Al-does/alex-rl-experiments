"""Entropy-regularised SAC extension of the eps02 ctx32 transformer study."""
from experiments.reward_factors_persistent_nuisance_jax_2026_11 import shared as base
from experiments.belief_quotient_persistent_nuisance_jax_temp15_2026_11.shared import (
    ENTROPY_SCHEDULE, TEMPERATURE, TOTAL_ENV_STEPS,
)
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_sac_autoalpha_2026_11.shared import training_backend

STUDY = "reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_sac_autoalpha_2026_11"


def run_condition(context, *, condition, architecture="transformer"):
    if architecture != "transformer":
        raise ValueError("this SAC study supports only transformer")
    return base.run_condition(
        context, condition=condition, architecture=architecture, study=STUDY,
        total_env_steps=TOTAL_ENV_STEPS, entropy_schedule_points=ENTROPY_SCHEDULE,
        temperature=TEMPERATURE, environment=environment,
        training_backend=training_backend(context), context_len=32,
    )
