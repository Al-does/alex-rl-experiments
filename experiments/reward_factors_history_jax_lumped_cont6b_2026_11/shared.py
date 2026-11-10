"""Second continuation of the lumped-baseline study: restores each arm's completed
``reward_factors_history_jax_lumped_cont4b_2026_11`` final checkpoint (~4.02B steps) and trains
another ``ADDITIONAL_ENV_STEPS`` (to ~6.04B absolute); recipe otherwise identical."""

from pathlib import Path

from experiments.belief_quotient_history_jax_2026_11.continuation import load_parent_checkpoint
from experiments.reward_factors_history_jax_lumped_2026_11 import shared as base
from experiments.reward_factors_history_jax_lumped_cont4b_2026_11 import shared as parent

STUDY = "reward_factors_history_jax_lumped_cont6b_2026_11"
ADDITIONAL_ENV_STEPS = base.TOTAL_ENV_STEPS
PARENT_ROOT = Path(parent.__file__).resolve().parent


def run_condition(context, *, condition, architecture):
    leaf = f"{condition}_{architecture}"
    process = base.factor_process(condition)
    continue_from = load_parent_checkpoint(
        context, parent_leaf_dir=PARENT_ROOT / leaf, study=parent.STUDY,
        spec=base.model_spec(architecture, process),
    )
    return base.run_condition(
        context, condition=condition, architecture=architecture,
        total_env_steps=ADDITIONAL_ENV_STEPS, study=STUDY, continue_from=continue_from,
    )
