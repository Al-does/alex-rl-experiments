"""Matched mechanics for the global-alias and targeted PPG runs."""

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
EPISODE_LENGTH = 1_000
GAMMA = 0.990
GAE_LAMBDA = 0.95
POLICY_LR = 3e-4
AUX_LR = 3e-4
POLICY_ITERATIONS_PER_AUX = 32
AUX_EPOCHS = 6
BETA_CLONE = 1.0
AUX_VALUE_LOSS_COEFF = 0.003
ENTROPY_COEFF = [[0, 0.03], [2_500_000, 0.008]]

REWARD_RANGES = {
    "global_aliases": (-15.0, 0.9985**4),
    "targeted": (-3.75, 0.9985**4),
}
# Use one common ceiling based on the larger global-alias reward magnitude so
# action scope remains the only algorithmic difference between the two runs.
VF_CLIP_PARAM = (15.0 / (1.0 - GAMMA)) ** 2

MODEL_CONFIG = TransformerModelConfig(
    d_model=64,
    n_layers=4,
    n_heads=1,
    context_len=256,
    max_seq_len=256,
).to_dict()


class CassandraPPGTransformer(PPGAuxiliaryValueHead, TransformerModel):
    """Stable Cassandra transformer plus PPG's auxiliary value head."""


def environment_config(action_scope: str) -> dict[str, Any]:
    if action_scope not in REWARD_RANGES:
        raise ValueError(f"unsupported PPG action scope: {action_scope}")
    return {
        "episode_length": EPISODE_LENGTH,
        "action_scope": action_scope,
        "initial_state_distribution": "all_good",
        "diagnostics": False,
    }


def build_config(
    context: RunContext,
    *,
    action_scope: str,
) -> PPGConfig:
    """Build one fresh, action-scope-matched PPG configuration."""

    if context.seed is None:
        raise ValueError("Cassandra PPG requires a resolved seed")
    profile = context.hardware or PROFILES["cpu"]
    smoke = context.smoke
    return (
        PPGConfig()
        .environment(
            CassandraActionObservationEnv,
            env_config=environment_config(action_scope),
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
                2 if smoke else POLICY_ITERATIONS_PER_AUX
            ),
            aux_epochs=1 if smoke else AUX_EPOCHS,
            aux_minibatch_size=(
                SMOKE_MINIBATCH_SIZE if smoke else MINIBATCH_SIZE
            ),
            aux_lr=AUX_LR,
            beta_clone=BETA_CLONE,
            aux_value_loss_coeff=AUX_VALUE_LOSS_COEFF,
            aux_true_value_loss_coeff=AUX_VALUE_LOSS_COEFF,
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


def run_condition(
    context: RunContext,
    *,
    action_scope: str,
    condition: str,
):
    """Run one matched 5M-step Cassandra PPG condition."""

    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    reward_range = REWARD_RANGES[action_scope]
    discounted_value_range = [
        reward / (1.0 - GAMMA) for reward in reward_range
    ]
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
            "environment": environment_config(action_scope),
            "action_names": list(action_names(action_scope)),
            "reward_range": list(reward_range),
            "discounted_value_range_bound": discounted_value_range,
            "model_config": MODEL_CONFIG,
            "entropy_coeff_schedule": ENTROPY_COEFF,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "vf_clip_param_squared_loss_ceiling": VF_CLIP_PARAM,
            "policy_iterations_per_aux": (
                2 if context.smoke else POLICY_ITERATIONS_PER_AUX
            ),
            "aux_epochs": 1 if context.smoke else AUX_EPOCHS,
            "beta_clone": BETA_CLONE,
            "aux_value_loss_coeff": AUX_VALUE_LOSS_COEFF,
            "aux_true_value_loss_coeff": AUX_VALUE_LOSS_COEFF,
            "primary_comparison": (
                "global aliases versus component-targeted actions with every "
                "model and PPG hyperparameter held fixed"
            ),
        },
    )
    return run_tune(
        build_config(context, action_scope=action_scope),
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
    "AUX_EPOCHS",
    "AUX_VALUE_LOSS_COEFF",
    "BETA_CLONE",
    "CassandraPPGTransformer",
    "ENTROPY_COEFF",
    "MODEL_CONFIG",
    "POLICY_ITERATIONS_PER_AUX",
    "REWARD_RANGES",
    "SMOKE_ENV_STEPS",
    "TOTAL_ENV_STEPS",
    "VF_CLIP_PARAM",
    "build_config",
    "environment_config",
    "run_condition",
]
