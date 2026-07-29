"""Sticky-state PPO recipe and longitudinal probes for three action variants."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from numbers import Real
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.shared import (
    apply_runtime_resources,
)
from experiments.mess3_reward_state_action_symmetry_cycle_4.analysis import (
    ProbeResult,
    plot_probe,
    probe_checkpoint,
)
from experiments.mess3_reward_state_action_symmetry_cycle_4.design import (
    CYCLE_4_TRANSITION_MATRIX,
    EFFECT_SIZE,
    analytic_design_summary,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.runners import run_tune
from learners.models.transformer import TransformerModel, TransformerModelConfig


TOTAL_ENV_STEPS = 700_000
SMOKE_ENV_STEPS = 4_096
TRAIN_BATCH_SIZE = 32_768
SMOKE_BATCH_SIZE = 2_048
MINIBATCH_SIZE = 2_048
SMOKE_MINIBATCH_SIZE = 256
DEFAULT_ENTROPY_COEFF = 0.003
BASE_MODEL_CONFIG = TransformerModelConfig(
    d_model=96,
    n_layers=3,
    n_heads=4,
    context_len=64,
).to_dict()


def _single_gpu_context(context: RunContext) -> RunContext:
    """Use cuda4090 on 1-GPU boxes; gpuinfer needs 1.8 GPUs to schedule."""

    profile = context.hardware
    if (
        not context.smoke
        and profile is not None
        and profile.name == "cuda4090_gpuinfer"
    ):
        return replace(context, hardware=PROFILES["cuda4090"])
    return context


def environment_config(variant: int) -> dict[str, Any]:
    """Build the sticky-state HMM configuration for one action variant."""

    if variant not in (1, 2, 3):
        raise ValueError("variant must be one of 1, 2, or 3")
    return {
        "model": {
            "factory": "envs.mess3.model:control_model",
            "kwargs": {
                "alpha": 0.85,
                "transition_matrix": [
                    list(row) for row in CYCLE_4_TRANSITION_MATRIX
                ],
            },
        },
        "task": {
            "class": (
                "experiments.mess3_reward_state_action_symmetry_cycle_4.task:"
                "ActionSymmetryTask"
            ),
            "kwargs": {
                "variant": variant,
                "effect_size": EFFECT_SIZE,
            },
        },
        "delay": 0,
        "episode_length": 1024,
        "randomize_first_episode_length": True,
    }


def build_config(context: RunContext, variant: int) -> PPOConfig:
    """Build one fresh transformer PPO configuration."""

    batch_size = SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
    config = (
        PPOConfig()
        .environment(HMMEnv, env_config=environment_config(variant))
        .framework(
            "torch",
            torch_compile_learner=False,
            torch_compile_worker=False,
        )
        .training(
            lr=3e-4 if context.smoke else 4.2e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=DEFAULT_ENTROPY_COEFF,
            train_batch_size_per_learner=batch_size,
            minibatch_size=(
                SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE
            ),
            num_epochs=6,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TransformerModel,
                model_config=dict(BASE_MODEL_CONFIG),
            )
        )
        .debugging(seed=context.seed)
    )
    return apply_runtime_resources(
        config,
        _single_gpu_context(context),
        default_env_runners=16,
    )


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
    """Return retained checkpoints in sampled-step order."""

    candidates = list(result.best_checkpoints or [])
    if result.checkpoint is not None:
        candidates.append((result.checkpoint, result.metrics or {}))
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
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
    return sorted(records, key=lambda item: item["training_iteration"])


def _log_spaced_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select iterations 1, 2, 4, ... and always include the final one."""

    if not records:
        return []
    selected = [
        record
        for record in records
        if record["training_iteration"] > 0
        and record["training_iteration"].bit_count() == 1
    ]
    if selected[-1]["checkpoint_name"] != records[-1]["checkpoint_name"]:
        selected.append(records[-1])
    return selected


def _save_initial_checkpoint(config: PPOConfig, path: Path) -> Path:
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
    probe_dir = (
        context.results_dir
        / "checkpoint_probes"
        / f"steps_{agent_steps:09d}"
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
        "branch_baseline_mse": float(
            result.metrics["branch_baseline_mse"]
        ),
        "fine_mse_ratio": float(result.metrics["fine_mse_ratio"]),
        "reward_state_2_fraction_greedy": float(
            result.metrics["reward_state_2_fraction_greedy"]
        ),
        "greedy_action_fractions": result.metrics["greedy_action_fractions"],
        "probe": result.metrics,
    }
    return result, point


def run_condition(context: RunContext, variant: int) -> dict[str, Any]:
    """Train one PPO variant and probe init plus log-spaced checkpoints."""

    if context.seed is None:
        raise ValueError("action-symmetry cycle requires a resolved seed")
    condition = f"variant_{variant}"
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    recipe = {
        "condition": condition,
        "algorithm": "PPO",
        "gamma": 0.99,
        "lambda": 0.95,
        "entropy_coeff": DEFAULT_ENTROPY_COEFF,
        "environment": environment_config(variant),
        "analytic_design": analytic_design_summary(),
        "total_env_steps": target_steps,
        "checkpoint_schedule": "init_then_iterations_1_2_4_8_and_final",
        "checkpoint_storage": (
            "every_iteration_unpruned_pending_generic_log_schedule"
        ),
        "model_config": BASE_MODEL_CONFIG,
        "probe_target": "exact_predictive_bayesian_belief",
        "probe_sampling_distribution": "process_weighted_rollout",
    }
    outputs.write_json("resolved_recipe.json", recipe)
    config = build_config(context, variant)
    initial_checkpoint = _save_initial_checkpoint(
        config,
        context.artifacts_dir / "initial_checkpoint",
    )
    initial_probe, initial_point = _probe_at(
        context,
        checkpoint=initial_checkpoint,
        condition=f"{condition}_init",
        agent_steps=0,
    )

    result_grid = run_tune(
        config,
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": target_steps},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=None,
                checkpoint_frequency=1,
                checkpoint_at_end=True,
            )
        },
    )
    results = list(result_grid)
    if len(results) != 1:
        raise RuntimeError(f"{condition} expected one trial, got {len(results)}")
    result = results[0]
    if result.error is not None:
        raise RuntimeError(f"{condition} training failed") from result.error
    records = checkpoint_records(result)
    selected = _log_spaced_records(records)
    if not selected:
        raise RuntimeError(f"{condition} retained no checkpoints")

    trajectory = [initial_point]
    checkpoint_probes: list[ProbeResult] = []
    for record in selected:
        probe, point = _probe_at(
            context,
            checkpoint=Path(record["checkpoint"].path),
            condition=condition,
            agent_steps=record["agent_steps"],
        )
        checkpoint_probes.append(probe)
        trajectory.append(
            {
                **point,
                "training_iteration": record["training_iteration"],
                "checkpoint_name": record["checkpoint_name"],
            }
        )
    final_probe = checkpoint_probes[-1]
    plot_probe(
        initial_probe,
        title=f"{condition} — init",
        path=context.results_dir / "belief_simplex_init.png",
    )
    plot_probe(
        final_probe,
        title=f"{condition} — final",
        path=context.results_dir / "belief_simplex_final.png",
    )
    outputs.write_json(
        "checkpoint_probe_curve.json",
        {"condition": condition, "checkpoints": trajectory},
    )
    summary = {
        "condition": condition,
        "seed": context.seed,
        "smoke": context.smoke,
        "algorithm": "PPO",
        "initial_probe": initial_probe.metrics,
        "final_probe": final_probe.metrics,
        "checkpoint_probes": trajectory,
    }
    outputs.write_json("condition_summary.json", summary)
    return summary
