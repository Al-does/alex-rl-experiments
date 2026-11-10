"""Continuation of the small-batch temp15 study: restores each arm's completed
``reward_factors_history_jax_lumped_temp15_b2m_2026_11`` final checkpoint (501,350,400 steps, from
local ``artifacts/`` or B2) and trains another ``ADDITIONAL_ENV_STEPS`` with the
identical recipe (1,920 envs, 64 minibatches, entropy 0, temperature 1.5):
763 rollouts -> 2,001,469,440 absolute steps. Parameter-only restore: Adam
moments restart at the join (see ``continuation.py``)."""

from pathlib import Path

from experiments.belief_quotient_history_jax_2026_11.continuation import load_parent_checkpoint
from experiments.reward_factors_history_jax_lumped_2026_11 import shared as base
from experiments.reward_factors_history_jax_lumped_temp15_b2m_2026_11 import shared as parent

STUDY = "reward_factors_history_jax_lumped_temp15_b2m_cont2b_2026_11"
ADDITIONAL_ENV_STEPS = 1_500_000_000
PARENT_ROOT = Path(parent.__file__).resolve().parent


def run_condition(context, *, condition, architecture):
    leaf = f"{condition}_{architecture}"
    process = base.factor_process(condition)
    continue_from = load_parent_checkpoint(
        context, parent_leaf_dir=PARENT_ROOT / leaf, study=parent.STUDY,
        spec=base.model_spec(architecture, process),
    )
    return base.run_condition(
        context, condition=condition, architecture=architecture, total_env_steps=ADDITIONAL_ENV_STEPS,
        study=STUDY, continue_from=continue_from, entropy_schedule_points=parent.ENTROPY_SCHEDULE,
        temperature=parent.TEMPERATURE, num_envs=parent.NUM_ENVS,
    )
