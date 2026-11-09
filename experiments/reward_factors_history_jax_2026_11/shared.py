"""JAX reward-factor recipe: two reward conditions x four architectures."""

from __future__ import annotations

from experiments.belief_quotient_history_jax_2026_11.models import architecture_spec
from experiments.belief_quotient_history_jax_2026_11.reinforce import TrainConfig
from experiments.belief_quotient_history_jax_2026_11.shared import (
    ARCHITECTURES,
    EPISODE_LENGTH,
    NUM_ENVS,
    NUM_MINIBATCHES,
    entropy_schedule,
)
from experiments.belief_quotient_history_jax_2026_11.training import run_study
from experiments.reward_factors_history_jax_2026_11.process import (
    CONDITIONS,
    factor_process,
    factor_targets,
)

STUDY = "reward_factors_history_jax_2026_11"
SOURCE_STUDY = "reward_factors_history_REINFORCE_2026_11"
CONDITION_NAMES = {"reward_both": "Joint Reward", "reward_factor_1": "Unrewarded Factor 2"}
TOTAL_ENV_STEPS = 200_000_000
SMOKE_ENV_STEPS = 4_096
CONTEXT_LEN = 10


def train_config(smoke: bool) -> TrainConfig:
    return TrainConfig(total_env_steps=SMOKE_ENV_STEPS if smoke else TOTAL_ENV_STEPS,
                       episode_length=EPISODE_LENGTH, num_envs=NUM_ENVS, num_minibatches=NUM_MINIBATCHES,
                       entropy_schedule=entropy_schedule())


def model_spec(architecture, process):
    return architecture_spec(architecture, obs_dim=process.obs_dim, n_actions=process.n_actions,
                             n_tokens=process.n_tokens, context_len=CONTEXT_LEN)


def run_condition(context, *, condition, architecture):
    if condition not in CONDITIONS or architecture not in ARCHITECTURES:
        raise ValueError("unknown condition or architecture")
    process = factor_process(condition)
    return run_study(
        context, study=STUDY, condition=condition, architecture=architecture,
        process=process, spec=model_spec(architecture, process),
        config=train_config(context.smoke), targets=factor_targets,
        representation="residual",
        recipe_extra={
            "source_study": SOURCE_STUDY,
            "condition_name": CONDITION_NAMES[condition],
            "reward": "factor-1 occupancy of state 2, plus factor 2 for reward_both",
            "representation": "pre_final_layer_norm",
        },
    )
