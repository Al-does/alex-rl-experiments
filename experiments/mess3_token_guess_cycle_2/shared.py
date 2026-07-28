"""Shared gamma-zero recipe and checkpoint analysis for token-guess cycle 2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.mess3_token_guess_cycle_2.analysis import (
    BAYESIAN_OPTIMAL_ACCURACY,
    ProbeResult,
    plot_init_final,
    plot_probe_pair,
    plot_probe_trajectory,
    probe_checkpoint,
)
from experiments.mess3_token_guess_cycle_2.learning import (
    KELLY_LOSS_COEFFICIENT_KEY,
    A2CTorchLearner,
    DecoupledKellyHead,
    KellyPPOTorchLearner,
)
from experiments.mess3_token_guess_cycle_2.model import (
    PaperActorCriticConfig,
    PaperActorCriticModel,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune
from learners import (
    HUBER_KAPPA_KEY,
    LOSS_COEFFICIENT_KEY,
    IQNPPOTorchLearner,
)
from learners.models import IQNValueMixin
from learners.models.iqn_value import NAMESPACE as IQN_NAMESPACE
from learners.models.next_token import NextTokenAuxHead
from losses.next_token import NextTokenAuxLossMixin


@dataclass(frozen=True, slots=True)
class Condition:
    name: str
    algorithm: str
    objective: str


CONDITIONS = (
    Condition("a2c", "A2C", "on_policy_advantage_actor_critic"),
    Condition("ppo", "PPO", "clipped_correctness"),
    Condition("predictive_loss", "PPO", "correctness_plus_next_token_ce"),
    Condition("decoupled_kelly", "PPO", "correctness_plus_direct_kelly"),
    Condition("iqn", "PPO-IQN", "clipped_correctness_distributional_value"),
)
TOTAL_ENV_STEPS = 2_500_000
SMOKE_ENV_STEPS = 4_096
TRAIN_BATCH_SIZE = 32_768
SMOKE_BATCH_SIZE = 2_048
MINIBATCH_SIZE = 4_096
SMOKE_MINIBATCH_SIZE = 256
CHECKPOINT_FREQUENCY = 10
VALIDATION_ENV_STEPS = 131_072
PREDICTIVE_LOSS_WEIGHT = 0.1
DIRECT_KELLY_LOSS_WEIGHT = 1.0
IQN_LOSS_COEFFICIENT = 0.5
IQN_HUBER_KAPPA = 1.0
IQN_CONFIG = {
    "train_quantiles": 32,
    "value_quantiles": 64,
    "n_cosines": 64,
}
BASE_MODEL_CONFIG = PaperActorCriticConfig().to_dict()
ENV_CONFIG = {
    "model": {
        "factory": "envs.mess3.model:passive_model",
        "kwargs": {"alpha": 0.85},
    },
    "task": {
        "class": (
            "experiments.mess3_token_guess_cycle_2.task:"
            "NextTokenGuessTask"
        ),
    },
    "observation": {"action": None},
    "delay": 1,
    "episode_length": 512,
    "randomize_first_episode_length": True,
}


class PredictiveModel(NextTokenAuxHead, PaperActorCriticModel):
    """Paper transformer with a training-only predictive head."""


class PredictiveLearner(NextTokenAuxLossMixin, PPOTorchLearner):
    """Correctness PPO plus next-emission cross entropy."""


class KellyModel(DecoupledKellyHead, PaperActorCriticModel):
    """Paper transformer with a separate three-logit Kelly head."""


class IQNModel(IQNValueMixin, PaperActorCriticModel):
    """Paper transformer with a promoted implicit-quantile value head."""


def next_emission_targets(
    batch: Mapping[str, Any],
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align action-time activations with the emission scored by the task."""

    observations = batch[Columns.OBS]
    if observations.ndim != 3 or logits.ndim != 3:
        raise ValueError("next-emission training expects (B, T, D) tensors")
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
        mask = mask.to(device=observations.device, dtype=torch.bool)
    valid = mask[:, :-1] & mask[:, 1:] & populated
    return logits[:, :-1, :], targets, valid


def condition_by_name(name: str) -> Condition:
    try:
        return next(condition for condition in CONDITIONS if condition.name == name)
    except StopIteration as error:
        raise ValueError(f"unknown token-guess condition {name!r}") from error


def _module_class(condition: Condition):
    if condition.name == "predictive_loss":
        return PredictiveModel
    if condition.name == "decoupled_kelly":
        return KellyModel
    if condition.name == "iqn":
        return IQNModel
    return PaperActorCriticModel


def _learner_class(condition: Condition):
    if condition.name == "a2c":
        return A2CTorchLearner
    if condition.name == "predictive_loss":
        return PredictiveLearner
    if condition.name == "decoupled_kelly":
        return KellyPPOTorchLearner
    if condition.name == "iqn":
        return IQNPPOTorchLearner
    return None


def _model_config(condition: Condition) -> dict[str, Any]:
    config = dict(BASE_MODEL_CONFIG)
    if condition.name == "predictive_loss":
        config["next_token_aux"] = {"num_classes": 3}
    if condition.name == "iqn":
        config[IQN_NAMESPACE] = dict(IQN_CONFIG)
    return config


def _learner_config(condition: Condition) -> dict[str, Any]:
    if condition.name == "predictive_loss":
        return {
            "next_token_aux/lambda": PREDICTIVE_LOSS_WEIGHT,
            "next_token_aux/target_extractor": next_emission_targets,
        }
    if condition.name == "decoupled_kelly":
        return {KELLY_LOSS_COEFFICIENT_KEY: DIRECT_KELLY_LOSS_WEIGHT}
    if condition.name == "iqn":
        return {
            LOSS_COEFFICIENT_KEY: IQN_LOSS_COEFFICIENT,
            HUBER_KAPPA_KEY: IQN_HUBER_KAPPA,
        }
    return {}


def _apply_runtime_resources(config: PPOConfig, context: RunContext) -> PPOConfig:
    profile = context.hardware or PROFILES["cpu"]
    return config.env_runners(
        num_env_runners=(
            0 if context.smoke else resolve_env_runners(profile, default=16)
        ),
        num_envs_per_env_runner=(
            1 if context.smoke else profile.num_envs_per_env_runner
        ),
        # Keep rollout inference on CPU so one-GPU workers reserve the device
        # for the learner's forward/backward hot path.
        num_gpus_per_env_runner=0,
        sample_timeout_s=600.0,
    ).learners(
        num_gpus_per_learner=1 if profile.learner_device == "cuda" else 0,
    )


def build_config(
    context: RunContext,
    condition_name: str = "ppo",
) -> PPOConfig:
    """Build one fresh controlled gamma-zero condition."""

    condition = condition_by_name(condition_name)
    profile = context.hardware or PROFILES["cpu"]
    is_a2c = condition.name == "a2c"
    batch_size = SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
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
            gamma=0.0,
            lambda_=0.0,
            clip_param=0.2,
            use_kl_loss=False,
            vf_loss_coeff=0.0 if condition.name == "iqn" else 0.5,
            entropy_coeff=0.0,
            train_batch_size_per_learner=batch_size,
            minibatch_size=(
                None
                if is_a2c
                else (SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE)
            ),
            num_epochs=1 if is_a2c else 6,
            shuffle_batch_per_epoch=not is_a2c,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=_module_class(condition),
                model_config=_model_config(condition),
            )
        )
        .debugging(seed=context.seed)
    )
    learner_class = _learner_class(condition)
    if learner_class is not None:
        config = config.learners(
            learner_class=learner_class,
            learner_config_dict=_learner_config(condition),
        )
    return _apply_runtime_resources(config, context)


def _metric(metrics: Mapping[str, Any], path: str) -> float | None:
    direct = metrics.get(path)
    if isinstance(direct, Real):
        return float(direct)
    value: Any = metrics
    for part in path.split("/"):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return float(value) if isinstance(value, Real) else None


def checkpoint_records(result: Any) -> list[dict[str, Any]]:
    """Return every retained checkpoint in sampled-step order."""

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    candidates = list(result.best_checkpoints or [])
    if result.checkpoint is not None:
        candidates.append((result.checkpoint, result.metrics or {}))
    for checkpoint, metrics in candidates:
        path = str(checkpoint.path)
        if path in seen:
            continue
        steps = _metric(metrics, "env_runners/num_env_steps_sampled_lifetime")
        iteration = _metric(metrics, "training_iteration")
        if steps is None or iteration is None:
            continue
        seen.add(path)
        records.append(
            {
                "checkpoint": checkpoint,
                "checkpoint_name": Path(path).name,
                "training_iteration": int(iteration),
                "agent_steps": int(steps),
            }
        )
    return sorted(records, key=lambda record: record["agent_steps"])


def _save_initial_checkpoint(config: PPOConfig, path: Path) -> Path:
    """Materialize the deterministic pre-training module for N-init probing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    algorithm = config.build_algo()
    try:
        saved = algorithm.save_to_path(str(path))
    finally:
        algorithm.stop()
    return Path(saved)


def _probe_at(
    context: RunContext,
    *,
    checkpoint: Path,
    condition: str,
    agent_steps: int,
) -> tuple[ProbeResult, dict[str, Any]]:
    probe_dir = context.results_dir / "checkpoint_probes" / (
        f"steps_{agent_steps:09d}"
    )
    result = probe_checkpoint(
        replace(context, results_dir=probe_dir, resume_from=checkpoint),
        checkpoint=checkpoint,
        condition=condition,
        agent_steps=agent_steps,
    )
    point = {
        "agent_steps": agent_steps,
        "mse": float(result.metrics["mse"]),
        "target_variance": float(result.metrics["target_variance"]),
        "global_mse_ratio": float(result.metrics["global_mse_ratio"]),
        "fine_mse_ratio": float(result.metrics["fine_mse_ratio"]),
        "branch_baseline_mse": float(
            result.metrics["branch_baseline_mse"]
        ),
        "r_squared": float(result.metrics["r_squared"]),
        "token_accuracy_greedy": float(
            result.metrics["token_accuracy_greedy"]
        ),
        "bayesian_optimal_accuracy": BAYESIAN_OPTIMAL_ACCURACY,
        "probe": result.metrics,
    }
    return result, point


def run_condition(
    context: RunContext,
    condition_name: str,
    *,
    target_steps_override: int | None = None,
) -> dict[str, Any]:
    """Train one condition and probe init plus every retained checkpoint."""

    if context.seed is None:
        raise ValueError("token-guess cycle 2 requires a resolved seed")
    condition = condition_by_name(condition_name)
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    target_steps = (
        target_steps_override
        if target_steps_override is not None
        else (SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS)
    )
    if target_steps <= 0:
        raise ValueError("target steps must be positive")
    checkpoint_frequency = (
        1
        if context.smoke or target_steps_override is not None
        else CHECKPOINT_FREQUENCY
    )
    recipe = {
        "condition": condition.name,
        "algorithm": condition.algorithm,
        "objective": condition.objective,
        "gamma": 0.0,
        "lambda": 0.0,
        "environment": ENV_CONFIG,
        "model": _model_config(condition),
        "total_env_steps": target_steps,
        "validation_budget": target_steps_override is not None,
        "checkpoint_frequency_iterations": checkpoint_frequency,
        "predictive_loss_weight": (
            PREDICTIVE_LOSS_WEIGHT
            if condition.name == "predictive_loss"
            else 0.0
        ),
        "direct_kelly_loss_weight": (
            DIRECT_KELLY_LOSS_WEIGHT
            if condition.name == "decoupled_kelly"
            else 0.0
        ),
        "kelly_head_logits": 3 if condition.name == "decoupled_kelly" else 0,
        "kelly_reward_decoupled_from_ppo": (
            condition.name == "decoupled_kelly"
        ),
        "iqn_config": IQN_CONFIG if condition.name == "iqn" else None,
        "bayesian_optimal_accuracy_context_10": BAYESIAN_OPTIMAL_ACCURACY,
    }
    outputs.write_json("resolved_recipe.json", recipe)
    config = build_config(context, condition.name)
    initial_checkpoint = _save_initial_checkpoint(
        config,
        context.artifacts_dir / "initial_checkpoint",
    )
    initial_probe, initial_point = _probe_at(
        context,
        checkpoint=initial_checkpoint,
        condition=f"{condition.name}_init",
        agent_steps=0,
    )
    plot_probe_pair(
        initial_probe,
        title=f"{condition.name} — init",
        path=context.results_dir / "belief_simplex_init.png",
    )

    result_grid = run_tune(
        config,
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": target_steps},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=None,
                checkpoint_frequency=checkpoint_frequency,
                checkpoint_at_end=True,
            )
        },
    )
    results = list(result_grid)
    if len(results) != 1:
        raise RuntimeError(f"{condition.name} expected one trial, got {len(results)}")
    result = results[0]
    if result.error is not None:
        raise RuntimeError(f"{condition.name} training failed") from result.error
    checkpoints = checkpoint_records(result)
    if not checkpoints:
        raise RuntimeError(f"{condition.name} retained no checkpoints")

    trajectory = [initial_point]
    checkpoint_probes: list[ProbeResult] = []
    for record in checkpoints:
        probed, point = _probe_at(
            context,
            checkpoint=Path(record["checkpoint"].path),
            condition=condition.name,
            agent_steps=record["agent_steps"],
        )
        checkpoint_probes.append(probed)
        trajectory.append(
            {
                **point,
                "training_iteration": record["training_iteration"],
                "checkpoint_name": record["checkpoint_name"],
            }
        )
    final_probe = checkpoint_probes[-1]
    plot_probe_pair(
        final_probe,
        title=f"{condition.name} — final",
        path=context.results_dir / "belief_simplex_final.png",
    )
    plot_init_final(
        initial_probe,
        final_probe,
        condition=condition.name,
        path=context.results_dir / "belief_simplex_init_vs_final.png",
    )
    plot_probe_trajectory(
        trajectory,
        condition=condition.name,
        path=context.results_dir / "probe_and_success_trajectory.png",
    )
    outputs.write_json(
        "checkpoint_probe_curve.json",
        {"condition": condition.name, "checkpoints": trajectory},
    )
    summary = {
        "condition": condition.name,
        "seed": context.seed,
        "smoke": context.smoke,
        "gamma": 0.0,
        "algorithm": condition.algorithm,
        "bayesian_optimal_accuracy_context_10": BAYESIAN_OPTIMAL_ACCURACY,
        "initial_probe": initial_probe.metrics,
        "final_probe": final_probe.metrics,
        "training_change": {
            "mse_delta": (
                float(final_probe.metrics["mse"])
                - float(initial_probe.metrics["mse"])
            ),
            "global_mse_ratio_delta": (
                float(final_probe.metrics["global_mse_ratio"])
                - float(initial_probe.metrics["global_mse_ratio"])
            ),
            "task_success_delta": (
                float(final_probe.metrics["token_accuracy_greedy"])
                - float(initial_probe.metrics["token_accuracy_greedy"])
            ),
            "task_success_improved": (
                float(final_probe.metrics["token_accuracy_greedy"])
                > float(initial_probe.metrics["token_accuracy_greedy"])
            ),
        },
        "checkpoint_probes": trajectory,
        "figures": {
            "init": str(context.results_dir / "belief_simplex_init.png"),
            "final": str(context.results_dir / "belief_simplex_final.png"),
            "init_vs_final": str(
                context.results_dir / "belief_simplex_init_vs_final.png"
            ),
            "trajectory": str(
                context.results_dir / "probe_and_success_trajectory.png"
            ),
        },
    }
    outputs.write_json("condition_summary.json", summary)
    return summary


def run_battery(context: RunContext) -> dict[str, Any]:
    """Run each controlled condition once; intended for smoke validation."""

    summaries = {}
    for condition in CONDITIONS:
        condition_context = replace(
            context,
            results_dir=context.results_dir / condition.name,
            artifacts_dir=context.artifacts_dir / condition.name,
            resume_from=None,
        )
        summaries[condition.name] = run_condition(
            condition_context,
            condition.name,
        )
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    summary = {
        "seed": context.seed,
        "smoke": context.smoke,
        "gamma": 0.0,
        "conditions": summaries,
        "bayesian_optimal_accuracy_context_10": BAYESIAN_OPTIMAL_ACCURACY,
    }
    outputs.write_json("battery_summary.json", summary)
    return summary
