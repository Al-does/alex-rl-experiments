"""10x budget (2B env steps) companion of ``reward_factors_history_jax_2026_11``; recipe otherwise identical."""

from experiments.reward_factors_history_jax_2026_11 import shared as parent

STUDY = "reward_factors_history_jax_2b_2026_11"
BUDGET_MULTIPLIER = 10
TOTAL_ENV_STEPS = parent.TOTAL_ENV_STEPS * BUDGET_MULTIPLIER


def run_condition(context, *, condition, architecture):
    return parent.run_condition(
        context, condition=condition, architecture=architecture,
        total_env_steps=TOTAL_ENV_STEPS, study=STUDY,
    )
