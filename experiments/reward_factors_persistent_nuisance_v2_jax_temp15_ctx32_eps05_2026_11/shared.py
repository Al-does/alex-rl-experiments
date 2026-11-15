"""ctx32 persistent-nuisance v2 factors with epsilon-greedy exploration (epsilon 0.05)."""
from experiments.reward_factors_persistent_nuisance_v2_jax_temp15_ctx32_2026_11 import shared as parent
from experiments.reward_factors_persistent_nuisance_v2_jax_temp15_ctx32_2026_11.shared import (
    ARCHITECTURE, CONTEXT_LEN, ENTROPY_SCHEDULE, TEMPERATURE, TOTAL_ENV_STEPS, base,
)

STUDY = "reward_factors_persistent_nuisance_v2_jax_temp15_ctx32_eps05_2026_11"
EXPLORATION_EPSILON = 0.05


def run_condition(context, *, condition, architecture=ARCHITECTURE):
    return parent.run_condition(context, condition=condition, architecture=architecture,
                                study=STUDY, exploration_epsilon=EXPLORATION_EPSILON)
