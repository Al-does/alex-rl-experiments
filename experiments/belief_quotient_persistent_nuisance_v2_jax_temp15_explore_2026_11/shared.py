"""Exploration-pressure sweep on the persistent-nuisance v2 temp15 transformer arms.

Every arm is the ``belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11`` recipe (v2 env,
2B steps, T=1.5, seed 42) with one exploration knob changed: either epsilon-greedy mixing with
no entropy bonus, or a constant (non-annealed) entropy bonus with no epsilon.
"""
from experiments.belief_quotient_persistent_nuisance_jax_2026_11 import shared as base
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import environment
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.shared import (
    TEMPERATURE, TOTAL_ENV_STEPS,
)

STUDY = "belief_quotient_persistent_nuisance_v2_jax_temp15_explore_2026_11"
ARMS = {
    "eps075": {"exploration_epsilon": 0.075, "entropy_coeff": 0.0},
    "eps10": {"exploration_epsilon": 0.1, "entropy_coeff": 0.0},
    "ent002": {"exploration_epsilon": 0.0, "entropy_coeff": 0.002},
    "ent004": {"exploration_epsilon": 0.0, "entropy_coeff": 0.004},
    "ent008": {"exploration_epsilon": 0.0, "entropy_coeff": 0.008},
}
VARIANTS = (2, 3)


def run_condition(context, *, variant, arm):
    settings = ARMS[arm]
    return base.run_condition(
        context, variant=variant, architecture="transformer", study=STUDY,
        total_env_steps=TOTAL_ENV_STEPS, entropy_schedule_points=((0, settings["entropy_coeff"]),),
        temperature=TEMPERATURE, environment=environment,
        exploration_epsilon=settings["exploration_epsilon"],
    )
