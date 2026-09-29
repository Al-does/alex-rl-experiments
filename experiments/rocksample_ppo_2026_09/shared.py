"""Common PPO settings for the six RockSample conditions."""

from __future__ import annotations

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.rocksample import RockSampleEnv
from experiments.storage.training_curves import write_training_curves
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune
from learners.models.transformer import TransformerModel, TransformerModelConfig


TOTAL_ENV_STEPS = 1_000_000
SMOKE_ENV_STEPS = 2_048
TRAIN_BATCH_SIZE = 8_192
SMOKE_BATCH_SIZE = 1_024
POLICY_CLIP = 0.2
VALUE_LOSS_CLIP = 1_000_000.0


def build_config(
    context: RunContext, *, env_config: dict[str, int], d_model: int
) -> PPOConfig:
    profile = context.hardware or PROFILES["cpu"]
    return (
        PPOConfig()
        .environment(RockSampleEnv, env_config=dict(env_config))
        .framework("torch", torch_compile_learner=False, torch_compile_worker=False)
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=POLICY_CLIP,
            vf_loss_coeff=0.5,
            vf_clip_param=VALUE_LOSS_CLIP,
            entropy_coeff=0.05,
            grad_clip=0.5,
            grad_clip_by="global_norm",
            train_batch_size_per_learner=(
                SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
            ),
            minibatch_size=256 if context.smoke else 1_024,
            num_epochs=4,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TransformerModel,
                model_config=TransformerModelConfig(
                    d_model=d_model,
                    n_layers=3,
                    n_heads=4,
                    context_len=32,
                ).to_dict(),
            )
        )
        .debugging(seed=context.seed)
        .env_runners(
            batch_mode="truncate_episodes",
            num_env_runners=(
                0 if context.smoke else resolve_env_runners(profile, default=4)
            ),
            num_envs_per_env_runner=(
                1 if context.smoke else profile.num_envs_per_env_runner
            ),
            num_gpus_per_env_runner=(
                0 if context.smoke else profile.num_gpus_per_env_runner
            ),
            sample_timeout_s=600.0,
        )
        .learners(num_gpus_per_learner=1 if profile.learner_device == "cuda" else 0)
    )


def run_condition(context: RunContext, *, env_config: dict[str, int], d_model: int):
    if context.seed is None:
        raise ValueError("RockSample PPO requires a resolved seed")

    config = build_config(context, env_config=env_config, d_model=d_model)
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        {
            "instance": [env_config.get("n", 5), env_config.get("k", 7)],
            "environment": dict(env_config),
            "model": config.rl_module_spec.model_config,
            "algorithm": "PPO",
            "gamma": config.gamma,
            "entropy_coeff": config.entropy_coeff,
            "policy_clip": config.clip_param,
            "value_loss_clip": config.vf_clip_param,
            "seed": context.seed,
            "total_env_steps": SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS,
        },
    )
    result_grid = run_tune(
        config,
        context,
        stop={
            "env_runners/num_env_steps_sampled_lifetime": (
                SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
            )
        },
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=1, checkpoint_at_end=True
            ),
        },
    )
    results = list(result_grid)
    if len(results) != 1 or results[0].error is not None:
        raise RuntimeError("RockSample PPO trial did not complete successfully")
    write_training_curves(context)
    return result_grid
