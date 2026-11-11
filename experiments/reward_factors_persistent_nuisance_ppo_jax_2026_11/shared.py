"""Persistent-nuisance reward factors (PR #208) trained with standard PPO + critic."""

from experiments.belief_quotient_persistent_nuisance_jax_2026_11.shared import environment_metadata
from experiments.belief_quotient_persistent_nuisance_ppo_jax_2026_11.shared import train_config
from experiments.belief_quotient_persistent_nuisance_ppo_jax_2026_11.training import run_study
from experiments.reward_factors_history_jax_2026_11 import shared as factors_base
from experiments.reward_factors_history_jax_2026_11.process import ACTION_PAIRS, factor_targets
from experiments.reward_factors_persistent_nuisance_jax_2026_11 import shared as source

STUDY = "reward_factors_persistent_nuisance_ppo_jax_2026_11"
SOURCE_STUDY = source.STUDY
ARCHITECTURES = source.ARCHITECTURES
TOTAL_ENV_STEPS = source.TOTAL_ENV_STEPS
CONDITIONS = tuple(factors_base.CONDITION_NAMES)
factor_process = source.factor_process
model_spec = source.model_spec


def run_condition(context, *, condition, architecture):
    if condition not in CONDITIONS or architecture not in ARCHITECTURES:
        raise ValueError("unknown condition or architecture")
    process = factor_process(condition)
    return run_study(
        context, study=STUDY, condition=condition, architecture=architecture,
        process=process, spec=model_spec(architecture, process),
        config=train_config(context.smoke), targets=factor_targets, representation="residual",
        recipe_extra={**environment_metadata(), "source_study": SOURCE_STUDY,
                      "action_pairs": ACTION_PAIRS,
                      "condition_name": factors_base.CONDITION_NAMES[condition],
                      "reward": "factor-1 state-2 occupancy, plus factor 2 for reward_both",
                      "representation": "pre_final_layer_norm"},
    )
