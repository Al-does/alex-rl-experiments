"""Persistent-nuisance ladder (PR #208) trained with standard PPO + critic."""

from experiments.belief_quotient_history_jax_2026_11 import shared as base
from experiments.belief_quotient_history_jax_2026_11.shared import belief_targets
from experiments.belief_quotient_persistent_nuisance_jax_2026_11 import shared as source
from experiments.belief_quotient_persistent_nuisance_ppo_jax_2026_11.ppo import PPOConfig
from experiments.belief_quotient_persistent_nuisance_ppo_jax_2026_11.training import run_study

STUDY = "belief_quotient_persistent_nuisance_ppo_jax_2026_11"
SOURCE_STUDY = source.STUDY
ARCHITECTURES = source.ARCHITECTURES
CONDITION_NAMES = source.CONDITION_NAMES
TOTAL_ENV_STEPS = source.TOTAL_ENV_STEPS
ladder_process = source.ladder_process
model_spec = source.model_spec
environment_metadata = source.environment_metadata


def train_config(smoke: bool, total_env_steps: int = TOTAL_ENV_STEPS) -> PPOConfig:
    """PR #208 batch geometry and entropy bonus; PPO coefficients from ``PPOConfig``."""

    return PPOConfig(total_env_steps=base.SMOKE_ENV_STEPS if smoke else total_env_steps,
                     num_envs=base.NUM_ENVS, episode_length=base.EPISODE_LENGTH,
                     num_minibatches=base.NUM_MINIBATCHES, entropy_coeff=base.ENTROPY_COEFF)


def run_condition(context, *, variant, architecture):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    process = ladder_process(variant)
    return run_study(
        context, study=STUDY, condition=f"variant_{variant}", architecture=architecture,
        process=process, spec=model_spec(architecture, process),
        config=train_config(context.smoke), targets=belief_targets, representation="embedding",
        recipe_extra={**environment_metadata(), "source_study": SOURCE_STUDY, "variant": variant,
                      "condition_name": CONDITION_NAMES[variant],
                      "reward": "pre-transition state-2 occupancy",
                      "representation": "post_final_layer_norm"},
    )
