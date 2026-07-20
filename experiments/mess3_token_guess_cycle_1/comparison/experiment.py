"""Compare reward, predictive-loss, and max-entropy token-guess agents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import torch
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.mess3_token_guess_cycle_1.analysis import (
    ProbeResult,
    plot_comparison,
    probe_checkpoint,
)
from experiments.mess3_token_guess_cycle_1.entropy_reward import (
    COEFFICIENT_KEY,
    EntropyRewardPPOTorchLearner,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune
from learners.models.next_token import NextTokenAuxHead
from learners.models.transformer import (
    TransformerModel,
    TransformerModelConfig,
)
from losses.next_token import NextTokenAuxLossMixin


class PredictiveModule(NextTokenAuxHead, TransformerModel):
    """Transformer actor-critic with a training-only token head."""


class PredictiveLearner(NextTokenAuxLossMixin, PPOTorchLearner):
    """PPO plus the controlled next-token auxiliary objective."""


@dataclass(frozen=True, slots=True)
class Arm:
    name: str
    module_class: type
    learner_class: type | None
    learner_config: Mapping[str, Any]
    predictive_loss_weight: float
    entropy_reward_coefficient: float


TOTAL_ENV_STEPS = 2_500_000
SMOKE_ENV_STEPS = 4_096
PREDICTIVE_LOSS_WEIGHT = 0.1
ENTROPY_REWARD_COEFFICIENT = 0.05
ENV_CONFIG = {
    "model": {
        "factory": "envs.mess3.model:passive_model",
        "kwargs": {"alpha": 0.85},
    },
    "task": {
        "class": (
            "experiments.mess3_token_guess_cycle_1.task:"
            "NextTokenGuessTask"
        ),
    },
    "observation": {"action": None},
    "delay": 1,
    "episode_length": 512,
    "randomize_first_episode_length": True,
}
BASE_MODEL_CONFIG = TransformerModelConfig(
    d_model=96,
    n_layers=3,
    n_heads=4,
    context_len=64,
).to_dict()


def next_token_targets(
    batch: Mapping[str, Any],
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align each residual activation with the following visible token."""

    observations = batch[Columns.OBS]
    if observations.ndim != 3 or logits.ndim != 3:
        raise ValueError("next-token training expects (B, T, D) tensors")
    num_classes = logits.shape[-1]
    next_tokens = observations[:, 1:, :num_classes]
    targets = next_tokens.argmax(dim=-1)
    populated = next_tokens.sum(dim=-1) > 0.5
    mask = batch.get(Columns.LOSS_MASK)
    if mask is None:
        mask = torch.ones(
            observations.shape[:2],
            dtype=torch.bool,
            device=observations.device,
        )
    else:
        mask = mask.to(dtype=torch.bool)
    valid = mask[:, :-1] & mask[:, 1:] & populated
    return logits[:, :-1, :], targets, valid


ARMS = (
    Arm(
        name="reward_only",
        module_class=TransformerModel,
        learner_class=None,
        learner_config={},
        predictive_loss_weight=0.0,
        entropy_reward_coefficient=0.0,
    ),
    Arm(
        name="predictive_loss",
        module_class=PredictiveModule,
        learner_class=PredictiveLearner,
        learner_config={
            "next_token_aux/lambda": PREDICTIVE_LOSS_WEIGHT,
            "next_token_aux/target_extractor": next_token_targets,
        },
        predictive_loss_weight=PREDICTIVE_LOSS_WEIGHT,
        entropy_reward_coefficient=0.0,
    ),
    Arm(
        name="max_entropy",
        module_class=TransformerModel,
        learner_class=EntropyRewardPPOTorchLearner,
        learner_config={
            COEFFICIENT_KEY: ENTROPY_REWARD_COEFFICIENT,
        },
        predictive_loss_weight=0.0,
        entropy_reward_coefficient=ENTROPY_REWARD_COEFFICIENT,
    ),
)


def _arm(name: str) -> Arm:
    try:
        return next(arm for arm in ARMS if arm.name == name)
    except StopIteration as error:
        raise ValueError(f"unknown comparison arm {name!r}") from error


def _apply_runtime_resources(config: PPOConfig, context: RunContext) -> PPOConfig:
    profile = context.hardware or PROFILES["cpu"]
    return config.env_runners(
        num_env_runners=(
            0
            if context.smoke
            else resolve_env_runners(profile, default=16)
        ),
        num_envs_per_env_runner=(
            1 if context.smoke else profile.num_envs_per_env_runner
        ),
        num_gpus_per_env_runner=(
            0 if context.smoke else profile.num_gpus_per_env_runner
        ),
        sample_timeout_s=600.0,
    ).learners(
        num_gpus_per_learner=(
            1 if profile.learner_device == "cuda" else 0
        )
    )


def build_config(
    context: RunContext,
    arm_name: str = "reward_only",
) -> PPOConfig:
    """Build a fresh controlled PPO configuration for one comparison arm."""

    arm = _arm(arm_name)
    profile = context.hardware or PROFILES["cpu"]
    model_config = dict(BASE_MODEL_CONFIG)
    if arm.predictive_loss_weight > 0.0:
        model_config["next_token_aux"] = {"num_classes": 3}

    config = (
        PPOConfig()
        .environment(HMMEnv, env_config=ENV_CONFIG)
        .framework(
            "torch",
            torch_compile_learner=(
                not context.smoke and profile.learner_device == "cuda"
            ),
            torch_compile_learner_what_to_compile="forward_train",
            torch_compile_learner_dynamo_backend="inductor",
            torch_compile_learner_dynamo_mode="reduce-overhead",
            torch_compile_worker=False,
        )
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.0,
            train_batch_size_per_learner=(
                2_048 if context.smoke else 32_768
            ),
            minibatch_size=256 if context.smoke else 4_096,
            num_epochs=6,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=arm.module_class,
                model_config=model_config,
            )
        )
        .debugging(seed=context.seed)
    )
    if arm.learner_class is not None:
        config = config.learners(
            learner_class=arm.learner_class,
            learner_config_dict=dict(arm.learner_config),
        )
    return _apply_runtime_resources(config, context)


def _subcontext(context: RunContext, arm: Arm) -> RunContext:
    return replace(
        context,
        results_dir=context.results_dir / arm.name,
        artifacts_dir=context.artifacts_dir / arm.name,
        resume_from=None,
    )


def _run_arm(context: RunContext, arm: Arm) -> ProbeResult:
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        {
            "condition": arm.name,
            "environment": ENV_CONFIG,
            "model": BASE_MODEL_CONFIG,
            "algorithm": "PPO",
            "total_env_steps": (
                SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
            ),
            "reward": "1 iff action equals the next emitted token, else 0",
            "predictive_loss_weight": arm.predictive_loss_weight,
            "entropy_reward_coefficient": arm.entropy_reward_coefficient,
            "rlib_entropy_loss_coefficient": 0.0,
        },
    )
    result_grid = run_tune(
        build_config(context, arm.name),
        context,
        stop={
            "env_runners/num_env_steps_sampled_lifetime": (
                SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
            ),
        },
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=True,
            ),
        },
    )
    results = list(result_grid)
    if len(results) != 1:
        raise RuntimeError(f"{arm.name} expected one trial, got {len(results)}")
    result = results[0]
    if result.error is not None:
        raise RuntimeError(f"{arm.name} training failed") from result.error
    if result.checkpoint is None:
        raise RuntimeError(f"{arm.name} produced no final checkpoint")

    probed = probe_checkpoint(
        context,
        checkpoint=Path(result.checkpoint.path),
        condition=arm.name,
    )
    outputs.write_json(
        "condition_summary.json",
        {
            "condition": arm.name,
            "predictive_loss_weight": arm.predictive_loss_weight,
            "entropy_reward_coefficient": arm.entropy_reward_coefficient,
            "probe": probed.metrics,
        },
    )
    return probed


def _findings(summary: Mapping[str, Any]) -> str:
    lines = [
        "# MESS3 token-guess belief comparison",
        "",
        "All conditions use the same transformer, PPO recipe, seed, environment, "
        "and binary next-token reward.",
        "",
        "| condition | held-out R² | greedy token accuracy | predictive λ | entropy α |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition, values in summary["conditions"].items():
        lines.append(
            f"| {condition} | {values['r_squared']:.4f} | "
            f"{values['token_accuracy_greedy']:.4f} | "
            f"{values['predictive_loss_weight']:.3f} | "
            f"{values['entropy_reward_coefficient']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The affine probe is fit on one rollout seed and evaluated on a "
            "disjoint seed. Belief labels are never used during reward-only or "
            "max-entropy training.",
            "",
        ]
    )
    return "\n".join(lines)


def run(context: RunContext):
    if context.seed is None:
        raise ValueError("the comparison requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    probes = {
        arm.name: _run_arm(_subcontext(context, arm), arm)
        for arm in ARMS
    }
    plot_comparison(
        probes,
        path=context.results_dir / "belief_simplex_comparison.png",
    )
    conditions = {
        arm.name: {
            **probes[arm.name].metrics,
            "predictive_loss_weight": arm.predictive_loss_weight,
            "entropy_reward_coefficient": arm.entropy_reward_coefficient,
        }
        for arm in ARMS
    }
    summary = {
        "paper": "arXiv:2405.15943",
        "seed": context.seed,
        "smoke": context.smoke,
        "conditions": conditions,
        "comparison_figure": str(
            context.results_dir / "belief_simplex_comparison.png"
        ),
    }
    outputs.write_json("comparison_summary.json", summary)
    (context.results_dir / "findings.md").write_text(_findings(summary))
    return summary
