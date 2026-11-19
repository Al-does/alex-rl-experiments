"""Continuation of ``reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11``:
restores each transformer arm's completed seed-matched final checkpoint (2B steps)
and trains for another ``ADDITIONAL_ENV_STEPS`` (to ~4B absolute steps); recipe
otherwise identical (epsilon-0.02 emissions, entropy 0.0, temperature 1.5, context 32)."""

from dataclasses import replace
from pathlib import Path

from experiments.belief_quotient_history_jax_2026_11.continuation import load_parent_checkpoint
from experiments.reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11 import (
    shared as parent,
)

STUDY = "reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_cont4b_2026_11"
ADDITIONAL_ENV_STEPS = parent.TOTAL_ENV_STEPS
PARENT_ROOT = Path(parent.__file__).resolve().parent
base = parent.base


def run_condition(context, *, condition, architecture="transformer"):
    process = base.factor_process(condition, parent.environment)
    spec = replace(base.model_spec(architecture, process), context_len=parent.CONTEXT_LEN)
    continue_from = load_parent_checkpoint(
        context, parent_leaf_dir=PARENT_ROOT / f"{condition}_{architecture}",
        study=parent.STUDY, spec=spec,
    )
    return parent.run_condition(
        context, condition=condition, architecture=architecture, study=STUDY,
        total_env_steps=ADDITIONAL_ENV_STEPS, continue_from=continue_from,
    )
