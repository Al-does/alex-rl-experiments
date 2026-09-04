"""Learning validation for Invariant Decoupled Advantage Actor-Critic.

CartPole is intentionally small enough for a CPU check while exercising the
real RLlib sampling, GAE, policy/advantage, value, and adversarial update paths.
The paper's algorithm-wide loss coefficients and alternating update schedule
are retained; only task-dependent PPO scale and budget settings are reduced.
"""

from __future__ import annotations

from ray import tune
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from experiments.storage.training_curves import write_training_curves
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune
from learners import IDAACConfig
from learners.models import IDAACModel


TOTAL_ENV_STEPS = 50_000
SMOKE_ENV_STEPS = 2_048
TRAIN_BATCH_SIZE = 1_024
SMOKE_BATCH_SIZE = 512
MINIBATCH_SIZE = 256


def build_config(context: RunContext) -> IDAACConfig:
    """Build a fresh IDAAC configuration for CartPole."""

    if context.seed is None:
        raise ValueError("the IDAAC CartPole validation requires a resolved seed")
    profile = context.hardware or PROFILES["cpu"]
    batch_size = SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
    return (
        IDAACConfig()
        .environment("CartPole-v1")
        .framework("torch", torch_compile_learner=False, torch_compile_worker=False)
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
            use_kl_loss=False,
            train_batch_size_per_learner=batch_size,
            minibatch_size=MINIBATCH_SIZE,
            num_epochs=1,
            value_num_epochs=3 if context.smoke else 9,
            value_update_frequency=1,
            value_minibatch_size=MINIBATCH_SIZE,
            advantage_loss_coeff=0.25,
            invariance_loss_coeff=0.001,
            adam_epsilon=1e-5,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=IDAACModel,
                model_config={
                    "encoder_type": "transformer",
                    "d_model": 64,
                    "n_layers": 4,
                    "n_heads": 1,
                    "context_len": 10,
                    "max_seq_len": 32,
                    "order_hidden_dims": (),
                },
            )
        )
        .debugging(seed=context.seed)
        .env_runners(
            num_env_runners=(
                0 if context.smoke else resolve_env_runners(profile, default=2)
            ),
            num_envs_per_env_runner=1,
            num_gpus_per_env_runner=0,
            sample_timeout_s=600.0,
        )
        .learners(
            num_gpus_per_learner=(
                1 if profile.learner_device == "cuda" else 0
            ),
        )
    )


def run(context: RunContext):
    """Train IDAAC and retain compact learning curves."""

    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    config = build_config(context)
    outputs.write_json(
        "resolved_recipe.json",
        {
            "algorithm": "IDAAC",
            "paper": "https://arxiv.org/abs/2102.10330",
            "environment": "CartPole-v1",
            "seed": context.seed,
            "smoke": context.smoke,
            "total_env_steps": target_steps,
            "train_batch_size_per_learner": config.train_batch_size_per_learner,
            "minibatch_size": config.minibatch_size,
            "policy_epochs": config.num_epochs,
            "value_epochs": config.value_num_epochs,
            "value_update_frequency": config.value_update_frequency,
            "advantage_loss_coeff": config.advantage_loss_coeff,
            "invariance_loss_coeff": config.invariance_loss_coeff,
        },
    )
    result_grid = run_tune(
        config,
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": target_steps},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=True,
            ),
        },
    )
    results = list(result_grid)
    if len(results) != 1:
        raise RuntimeError(f"expected one IDAAC trial, found {len(results)}")
    if results[0].error is not None:
        raise RuntimeError("IDAAC CartPole training failed") from results[0].error
    write_training_curves(context)
    return result_grid
