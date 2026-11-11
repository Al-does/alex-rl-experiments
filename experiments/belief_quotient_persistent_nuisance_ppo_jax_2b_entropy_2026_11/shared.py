"""Transformer arms of ``belief_quotient_persistent_nuisance_ppo_jax_2026_11`` (standard PPO
with critic + GAE) at a 10x budget (2B env steps) with a lower constant entropy bonus
(0.005 or 0.0, never annealed); everything else is inherited unchanged."""

from dataclasses import replace

from experiments.belief_quotient_history_jax_2026_11.shared import belief_targets
from experiments.belief_quotient_persistent_nuisance_ppo_jax_2026_11 import shared as ppo

STUDY = "belief_quotient_persistent_nuisance_ppo_jax_2b_entropy_2026_11"
SOURCE_STUDY = ppo.STUDY
BUDGET_MULTIPLIER = 10
TOTAL_ENV_STEPS = ppo.TOTAL_ENV_STEPS * BUDGET_MULTIPLIER
ENTROPY_COEFFS = {"ent0p005": 0.005, "ent0": 0.0}


def train_config(smoke, entropy):
    return replace(ppo.train_config(smoke, TOTAL_ENV_STEPS), entropy_coeff=ENTROPY_COEFFS[entropy])


def run_condition(context, *, variant, architecture, entropy):
    if architecture not in ppo.ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ppo.ARCHITECTURES}")
    process = ppo.ladder_process(variant)
    return ppo.run_study(
        context, study=STUDY, condition=f"variant_{variant}", architecture=architecture,
        process=process, spec=ppo.model_spec(architecture, process),
        config=train_config(context.smoke, entropy), targets=belief_targets,
        representation="embedding",
        recipe_extra={**ppo.environment_metadata(), "source_study": SOURCE_STUDY,
                      "variant": variant, "condition_name": ppo.CONDITION_NAMES[variant],
                      "reward": "pre-transition state-2 occupancy",
                      "representation": "post_final_layer_norm"},
    )
