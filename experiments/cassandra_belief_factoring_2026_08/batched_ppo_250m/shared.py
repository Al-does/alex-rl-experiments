"""Shared recipe for high-throughput fully observable Cassandra PPO."""

from __future__ import annotations

from typing import Any

import torch

from harness.artifacts import RunArtifacts
from harness.context import RunContext

from .environment import ActionScope, OBSERVATION_DIM
from .trainer import BatchedPPOTrainer, TrainerConfig


TOTAL_ENV_STEPS = 250_000_000
SMOKE_ENV_STEPS = 128
CHECKPOINT_STEP_INTERVAL = 50_000_000
EPISODE_LENGTH = 1_000
NUM_ENVS = 128
ROLLOUT_STEPS = 256
MINIBATCH_SIZE = 8_192
NUM_EPOCHS = 4
LEARNING_RATE = 3e-4
GAMMA = 0.990
GAE_LAMBDA = 0.95
CLIP_PARAM = 0.2
VF_CLIP_PARAM = 100.0
VF_LOSS_COEFF = 0.01
ENTROPY_COEFF = 0.03
D_MODEL = 64
N_LAYERS = 3
N_HEADS = 4
CONTEXT_LEN = 256

HYPOTHESIS = (
    "Keeping recurrent state and rollout tensors device-resident removes "
    "RLlib connector and Ray object-store overhead while preserving the "
    "fully observable long-context PPO recipe."
)
PRIMARY_COMPARISON = (
    "global-alias versus targeted maintenance actions under matched batched "
    "single-process PPO"
)


def build_config(
    context: RunContext, *, action_scope: ActionScope
) -> TrainerConfig:
    """Return the complete scientific configuration for one condition."""

    if context.seed is None:
        raise ValueError("batched Cassandra PPO requires a resolved seed")
    return TrainerConfig(
        action_scope=action_scope,
        total_env_steps=(
            SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
        ),
        num_envs=4 if context.smoke else NUM_ENVS,
        rollout_steps=32 if context.smoke else ROLLOUT_STEPS,
        minibatch_size=128 if context.smoke else MINIBATCH_SIZE,
        num_epochs=1 if context.smoke else NUM_EPOCHS,
        episode_length=EPISODE_LENGTH,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_param=CLIP_PARAM,
        vf_clip_param=VF_CLIP_PARAM,
        vf_loss_coeff=VF_LOSS_COEFF,
        entropy_coeff=ENTROPY_COEFF,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        context_len=CONTEXT_LEN,
        checkpoint_interval=CHECKPOINT_STEP_INTERVAL,
    )


def _device(context: RunContext) -> torch.device:
    requested = (
        context.hardware.learner_device
        if context.hardware is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA hardware profile selected, but torch cannot use CUDA"
        )
    if requested not in {"cpu", "cuda"}:
        raise RuntimeError(
            "batched Cassandra PPO currently supports CPU and CUDA devices"
        )
    return torch.device(requested)


def run_recipe(
    context: RunContext,
    *,
    action_scope: ActionScope,
    condition: str,
) -> dict[str, Any]:
    config = build_config(context, action_scope=action_scope)
    device = _device(context)
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        {
            "condition": condition,
            "hypothesis": HYPOTHESIS,
            "primary_comparison": PRIMARY_COMPARISON,
            "seed": context.seed,
            "algorithm": "experiment-local batched PPO",
            "environment": {
                "domain": "Cassandra machine maintenance",
                "action_scope": action_scope,
                "initial_state_distribution": "uniform",
                "episode_length": EPISODE_LENGTH,
                "observation": (
                    "four 4-way component-state one-hots plus preceding reward"
                ),
                "observation_dim": OBSERVATION_DIM,
                "belief_tracking": False,
                "device": str(device),
            },
            "transformer": {
                "d_model": D_MODEL,
                "n_layers": N_LAYERS,
                "n_heads": N_HEADS,
                "context_len": CONTEXT_LEN,
                "max_seq_len": ROLLOUT_STEPS,
            },
            "ppo": {
                "learning_rate": LEARNING_RATE,
                "gamma": GAMMA,
                "lambda": GAE_LAMBDA,
                "clip_param": CLIP_PARAM,
                "vf_clip_param": VF_CLIP_PARAM,
                "vf_loss_coeff": VF_LOSS_COEFF,
                "entropy_coeff": ENTROPY_COEFF,
                "num_epochs": config.num_epochs,
                "train_batch_size": config.train_batch_size,
                "minibatch_size": config.minibatch_size,
            },
            "runtime": {
                "num_envs": config.num_envs,
                "rollout_steps": config.rollout_steps,
                "total_env_steps": config.total_env_steps,
                "checkpoint_interval": CHECKPOINT_STEP_INTERVAL,
                "resume_from": (
                    str(context.resume_from)
                    if context.resume_from is not None
                    else None
                ),
            },
        },
    )
    summary = BatchedPPOTrainer(
        config=config,
        context=context,
        device=device,
    ).train()
    summary.update(
        {
            "condition": condition,
            "action_scope": action_scope,
            "seed": context.seed,
            "smoke": context.smoke,
        }
    )
    outputs.write_json("condition_summary.json", summary)
    return summary


__all__ = [
    "CHECKPOINT_STEP_INTERVAL",
    "CONTEXT_LEN",
    "D_MODEL",
    "ENTROPY_COEFF",
    "GAE_LAMBDA",
    "HYPOTHESIS",
    "MINIBATCH_SIZE",
    "N_HEADS",
    "N_LAYERS",
    "NUM_ENVS",
    "PRIMARY_COMPARISON",
    "ROLLOUT_STEPS",
    "SMOKE_ENV_STEPS",
    "TOTAL_ENV_STEPS",
    "VF_CLIP_PARAM",
    "VF_LOSS_COEFF",
    "build_config",
    "run_recipe",
]
