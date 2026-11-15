"""Continuation of ``belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11``:
restores each transformer arm's completed seed-matched final checkpoint (2B steps)
and trains for another ``ADDITIONAL_ENV_STEPS`` (to ~4B absolute steps); recipe
otherwise identical (epsilon-0.02 emissions, entropy 0.0, temperature 1.5)."""

from pathlib import Path

from experiments.belief_quotient_history_jax_2026_11.continuation import load_parent_checkpoint
from experiments.belief_quotient_persistent_nuisance_jax_2026_11 import shared as base
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import (
    environment, shared as parent,
)

STUDY = "belief_quotient_persistent_nuisance_eps02_jax_temp15_cont4b_2026_11"
ADDITIONAL_ENV_STEPS = parent.TOTAL_ENV_STEPS
PARENT_ROOT = Path(parent.__file__).resolve().parent


def run_condition(context, *, variant, architecture):
    continue_from = load_parent_checkpoint(
        context, parent_leaf_dir=PARENT_ROOT / f"variant_{variant}_{architecture}",
        study=parent.STUDY, spec=base.model_spec(architecture, environment.ladder_process(variant)),
    )
    return base.run_condition(
        context, variant=variant, architecture=architecture, study=STUDY,
        total_env_steps=ADDITIONAL_ENV_STEPS, entropy_schedule_points=parent.ENTROPY_SCHEDULE,
        temperature=parent.TEMPERATURE, environment=environment, continue_from=continue_from,
    )
