"""Shared mechanics for the isolated four-arm targeted PPG campaign."""

from __future__ import annotations

from typing import Any

from ray import tune
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.cassandra_machine import action_names
from experiments.cassandra_belief_factoring_2026_08.environment import (
    CassandraActionObservationEnv,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune
from learners import PPGConfig
from learners.models import (
    PPGAuxiliaryValueHead,
    TransformerModel,
    TransformerModelConfig,
)


TOTAL_ENV_STEPS = 5_000_000
SMOKE_ENV_STEPS = 4_096
TRAIN_BATCH_SIZE = 32_768
SMOKE_BATCH_SIZE = 2_048
MINIBATCH_SIZE = 8_192
SMOKE_MINIBATCH_SIZE = 256
TRAIN_ENVS_PER_ENV_RUNNER = 4
ACTION_SCOPE = "targeted"
EPISODE_LENGTH = 1_000
GAMMA = 0.990
GAE_LAMBDA = 0.95
POLICY_LR = 3e-4
AUX_LR = 3e-4
AUX_EPOCHS = 6
BETA_CLONE = 1.0
ENTROPY_COEFF = [[0, 0.03], [2_500_000, 0.008]]

# Targeted rewards lie in [-3.75, 0.9985**4]. At gamma=0.99, the largest
# possible absolute discounted value is 3.75 / (1 - gamma) = 375, so this
# squared-loss ceiling avoids clipping any in-range, zero-centered target.
VF_CLIP_PARAM = 375.0**2
REWARD_RANGE = (-3.75, 0.9985**4)
DISCOUNTED_VALUE_RANGE = (
    REWARD_RANGE[0] / (1.0 - GAMMA),
    REWARD_RANGE[1] / (1.0 - GAMMA),
)

MODEL_CONFIG = TransformerModelConfig(
    d_model=64,
    n_layers=4,
    n_heads=1,
    context_len=256,
    max_seq_len=256,
).to_dict()


class CassandraPPGTransformer(PPGAuxiliaryValueHead, TransformerModel):
    """Stable Cassandra transformer plus PPG's auxiliary value head."""


def environment_config() -> dict[str, Any]:
    return {
        "episode_length": EPISODE_LENGTH,
        "action_scope": ACTION_SCOPE,
        "initial_state_distribution": "all_good",
        "diagnostics": False,
    }


def build_config(
    context: RunContext,
    *,
    policy_iterations_per_aux: int,
    aux_value_loss_coeff: float,
) -> PPGConfig:
    """Build one fresh targeted PPG intervention configuration."""

    if context.seed is None:
        raise ValueError("Cassandra PPG requires a resolved seed")
    profile = context.hardware or PROFILES["cpu"]
    smoke = context.smoke
    config = (
        PPGConfig()
        .environment(
            CassandraActionObservationEnv,
            env_config=environment_config(),
        )
        .framework(
            "torch",
            torch_compile_learner=False,
            torch_compile_worker=False,
        )
        .training(
            lr=POLICY_LR,
            gamma=GAMMA,
            lambda_=GAE_LAMBDA,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            vf_clip_param=VF_CLIP_PARAM,
            entropy_coeff=ENTROPY_COEFF,
            use_kl_loss=False,
            kl_coeff=0.0,
            train_batch_size_per_learner=(
                SMOKE_BATCH_SIZE if smoke else TRAIN_BATCH_SIZE
            ),
            minibatch_size=(
                SMOKE_MINIBATCH_SIZE if smoke else MINIBATCH_SIZE
            ),
            num_epochs=4,
            policy_iterations_per_aux=(
                2 if smoke else policy_iterations_per_aux
            ),
            aux_epochs=1 if smoke else AUX_EPOCHS,
            aux_minibatch_size=(
                SMOKE_MINIBATCH_SIZE if smoke else MINIBATCH_SIZE
            ),
            aux_lr=AUX_LR,
            beta_clone=BETA_CLONE,
            aux_value_loss_coeff=aux_value_loss_coeff,
            aux_true_value_loss_coeff=aux_value_loss_coeff,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=CassandraPPGTransformer,
                model_config=dict(MODEL_CONFIG),
            )
        )
        .debugging(seed=context.seed)
        .env_runners(
            num_env_runners=(
                0 if smoke else resolve_env_runners(profile, default=8)
            ),
            num_envs_per_env_runner=(
                1 if smoke else TRAIN_ENVS_PER_ENV_RUNNER
            ),
            num_gpus_per_env_runner=(
                0 if smoke else profile.num_gpus_per_env_runner
            ),
            sample_timeout_s=600.0,
        )
        .learners(
            num_gpus_per_learner=(
                1 if profile.learner_device == "cuda" else 0
            )
        )
    )
    return config


def run_intervention(
    context: RunContext,
    *,
    condition: str,
    policy_iterations_per_aux: int,
    aux_value_loss_coeff: float,
):
    """Run one 5M-step targeted PPG intervention (or two-phase smoke)."""

    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        {
            "condition": condition,
            "algorithm": "single_network_detach_ppg",
            "seed": context.seed,
            "smoke": context.smoke,
            "total_env_steps": target_steps,
            "environment": environment_config(),
            "action_names": list(action_names(ACTION_SCOPE)),
            "reward_range": list(REWARD_RANGE),
            "discounted_value_range_bound": list(DISCOUNTED_VALUE_RANGE),
            "model_config": MODEL_CONFIG,
            "entropy_coeff_schedule": ENTROPY_COEFF,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "vf_clip_param_squared_loss_ceiling": VF_CLIP_PARAM,
            "policy_iterations_per_aux": (
                2 if context.smoke else policy_iterations_per_aux
            ),
            "aux_epochs": 1 if context.smoke else AUX_EPOCHS,
            "beta_clone": BETA_CLONE,
            "aux_value_loss_coeff": aux_value_loss_coeff,
            "aux_true_value_loss_coeff": aux_value_loss_coeff,
            "hyperparameter_rationale": (
                "Raw PPG half-MSE sees discounted targets as large as roughly "
                "-375 to 99 in this environment. Coefficients 0.003 and 0.01 "
                "bound a 100-point value error's auxiliary representation "
                "gradient scale to about 0.3 and 1.0 while beta_clone remains "
                "at the paper default 1.0. N_pi 16 and 32 compare more frequent "
                "distillation with the canonical cadence."
            ),
        },
    )
    return run_tune(
        build_config(
            context,
            policy_iterations_per_aux=policy_iterations_per_aux,
            aux_value_loss_coeff=aux_value_loss_coeff,
        ),
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": target_steps},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=True,
            )
        },
    )


__all__ = [
    "ACTION_SCOPE",
    "AUX_EPOCHS",
    "BETA_CLONE",
    "CassandraPPGTransformer",
    "DISCOUNTED_VALUE_RANGE",
    "ENTROPY_COEFF",
    "MODEL_CONFIG",
    "REWARD_RANGE",
    "SMOKE_ENV_STEPS",
    "TOTAL_ENV_STEPS",
    "VF_CLIP_PARAM",
    "build_config",
    "environment_config",
    "run_intervention",
]
