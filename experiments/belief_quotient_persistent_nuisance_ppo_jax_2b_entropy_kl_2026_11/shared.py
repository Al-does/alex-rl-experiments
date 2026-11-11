"""``belief_quotient_persistent_nuisance_ppo_jax_2b_entropy_2026_11`` (standard PPO, transformer,
2B env steps, constant entropy 0.005 or 0.0) with the RLlib-style adaptive KL penalty turned on
(``use_kl_loss=True``; initial ``kl_coeff`` 0.2, ``kl_target`` 0.01); otherwise identical."""

from dataclasses import replace

from experiments.belief_quotient_history_jax_2026_11.shared import belief_targets
from experiments.belief_quotient_persistent_nuisance_ppo_jax_2026_11 import shared as ppo
from experiments.belief_quotient_persistent_nuisance_ppo_jax_2b_entropy_2026_11 import shared as source

STUDY = "belief_quotient_persistent_nuisance_ppo_jax_2b_entropy_kl_2026_11"
SOURCE_STUDY = source.STUDY
TOTAL_ENV_STEPS = source.TOTAL_ENV_STEPS
ENTROPY_COEFFS = source.ENTROPY_COEFFS


def train_config(smoke, entropy):
    return replace(source.train_config(smoke, entropy), use_kl_loss=True)


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
