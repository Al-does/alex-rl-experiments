"""Second continuation of the lumped-baseline study: restores each arm's completed
``belief_quotient_history_jax_lumped_cont4b_2026_11`` final checkpoint (~4.02B steps) and trains
another ``ADDITIONAL_ENV_STEPS`` (to ~6.04B absolute); recipe otherwise identical."""

from pathlib import Path

from experiments.belief_quotient_history_jax_2026_11.continuation import load_parent_checkpoint
from experiments.belief_quotient_history_jax_lumped_2026_11 import shared as base
from experiments.belief_quotient_history_jax_lumped_cont4b_2026_11 import shared as parent

STUDY = "belief_quotient_history_jax_lumped_cont6b_2026_11"
ADDITIONAL_ENV_STEPS = base.TOTAL_ENV_STEPS
PARENT_ROOT = Path(parent.__file__).resolve().parent


def run_condition(context, *, variant, architecture):
    leaf = f"variant_{variant}_{architecture}"
    process = base.ladder_process(variant)
    continue_from = load_parent_checkpoint(
        context, parent_leaf_dir=PARENT_ROOT / leaf, study=parent.STUDY,
        spec=base.model_spec(architecture, process),
    )
    return base.run_condition(
        context, variant=variant, architecture=architecture,
        total_env_steps=ADDITIONAL_ENV_STEPS, study=STUDY, continue_from=continue_from,
    )
