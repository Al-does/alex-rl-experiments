"""PPO extension of the eps02 temperature-1.5 transformer study."""
from experiments.belief_quotient_persistent_nuisance_jax_2026_11 import shared as base
from experiments.belief_quotient_persistent_nuisance_jax_temp15_2026_11.shared import (
    ENTROPY_SCHEDULE, TEMPERATURE, TOTAL_ENV_STEPS,
)
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_ppo_2026_11 import ppo

STUDY = "belief_quotient_persistent_nuisance_eps02_jax_temp15_ppo_2026_11"


def run_condition(context, *, variant, architecture="transformer"):
    if architecture != "transformer":
        raise ValueError("this PPO study supports only transformer")
    return base.run_condition(
        context, variant=variant, architecture=architecture, study=STUDY,
        total_env_steps=TOTAL_ENV_STEPS, entropy_schedule_points=ENTROPY_SCHEDULE,
        temperature=TEMPERATURE, environment=environment, training_backend=ppo,
    )
