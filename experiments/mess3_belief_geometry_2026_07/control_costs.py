"""Shared recipe mechanics for the continuous-control cost comparison."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.checkpoint_probe import (
    experiment as checkpoint_probe,
)
from experiments.mess3_belief_geometry_2026_07.shared import (
    SMOKE_ENV_STEPS,
    apply_runtime_resources,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.runners import run_tune
from learners.models import TransformerModel, TransformerModelConfig


TOTAL_ENV_STEPS = 30_000_000
ACTION_LIMIT = 5.0
TRAIN_BATCH_SIZE = 65_536
MINIBATCH_SIZE = 8_192
LEARNING_RATE = 4.2e-4
MODEL_CONFIG = TransformerModelConfig(
    d_model=96,
    n_layers=3,
    n_heads=4,
    context_len=64,
).to_dict()


def environment_config(task_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build one independent controlled-MESS3 environment config."""

    return {
        "model": {
            "factory": "envs.mess3.model:control_model",
            "kwargs": {"alpha": 0.85},
        },
        "task": {
            "class": (
                "envs.mess3.tasks.occupancy_control:"
                "OccupancyControlTask"
            ),
            "kwargs": {
                "action_limit": ACTION_LIMIT,
                **task_kwargs,
            },
        },
        "delay": 1,
        "episode_length": 1024,
        "randomize_first_episode_length": True,
    }


def build_config(
    context: RunContext,
    *,
    task_kwargs: dict[str, Any],
) -> PPOConfig:
    """Build a fresh PPO config for one control-cost condition."""

    profile = context.hardware or PROFILES["cpu"]
    compile_learner = (
        not context.smoke and profile.learner_device == "cuda"
    )
    config = (
        PPOConfig()
        .environment(
            HMMEnv,
            env_config=environment_config(task_kwargs),
        )
        .framework(
            "torch",
            torch_compile_learner=compile_learner,
            torch_compile_learner_what_to_compile="forward_train",
            torch_compile_learner_dynamo_backend="inductor",
            torch_compile_learner_dynamo_mode="reduce-overhead",
            torch_compile_worker=False,
        )
        .training(
            lr=3e-4 if context.smoke else LEARNING_RATE,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.003,
            train_batch_size_per_learner=(
                2048 if context.smoke else TRAIN_BATCH_SIZE
            ),
            minibatch_size=256 if context.smoke else MINIBATCH_SIZE,
            num_epochs=6,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TransformerModel,
                model_config=MODEL_CONFIG,
            )
        )
        .debugging(seed=context.seed)
    )
    return apply_runtime_resources(
        config,
        context,
        default_env_runners=16,
    )


def run_condition(
    context: RunContext,
    *,
    condition: str,
    task_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Train one condition, then probe its final checkpoint before teardown."""

    condition_spec = {
        "condition": condition,
        "task_kwargs": {
            "action_limit": ACTION_LIMIT,
            **task_kwargs,
        },
        "total_env_steps": (
            SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
        ),
        "train_batch_size": (
            2048 if context.smoke else TRAIN_BATCH_SIZE
        ),
        "minibatch_size": (
            256 if context.smoke else MINIBATCH_SIZE
        ),
        "learning_rate": 3e-4 if context.smoke else LEARNING_RATE,
        "model_config": MODEL_CONFIG,
    }
    (context.results_dir / "condition_spec.json").write_text(
        json.dumps(condition_spec, indent=2) + "\n"
    )

    result_grid = run_tune(
        build_config(context, task_kwargs=task_kwargs),
        context,
        stop={
            "env_runners/num_env_steps_sampled_lifetime": (
                SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
            ),
        },
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=2,
                checkpoint_frequency=1 if context.smoke else 10,
                checkpoint_at_end=True,
            ),
        },
    )
    results = list(result_grid)
    if len(results) != 1:
        raise RuntimeError(
            f"{condition} expected one Tune trial, got {len(results)}"
        )
    result = results[0]
    if result.error is not None:
        raise RuntimeError(f"{condition} training failed") from result.error
    if result.checkpoint is None:
        raise RuntimeError(f"{condition} produced no final checkpoint")

    probe_metrics = checkpoint_probe.run(
        replace(
            context,
            resume_from=Path(result.checkpoint.path),
        )
    )
    summary = {
        "condition": condition,
        "checkpoint": str(result.checkpoint.path),
        "probe": probe_metrics,
    }
    (context.results_dir / "condition_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary
