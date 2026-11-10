"""10x budget (2B env steps) companion of ``belief_quotient_history_jax_2026_11``; recipe otherwise identical."""

from experiments.belief_quotient_history_jax_2026_11 import shared as parent

STUDY = "belief_quotient_history_jax_2b_2026_11"
BUDGET_MULTIPLIER = 10
TOTAL_ENV_STEPS = parent.TOTAL_ENV_STEPS * BUDGET_MULTIPLIER


def run_condition(context, *, variant, architecture):
    return parent.run_condition(
        context, variant=variant, architecture=architecture,
        total_env_steps=TOTAL_ENV_STEPS, study=STUDY,
    )
