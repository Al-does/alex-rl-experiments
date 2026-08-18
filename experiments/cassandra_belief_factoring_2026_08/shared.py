"""Transformer PPO recipe and longitudinal probes for Cassandra maintenance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from numbers import Real
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.cassandra_machine import DISCOUNT, action_names
from experiments.cassandra_belief_factoring_2026_08.analysis import (
    ProbeResult,
    plot_probe_trajectory,
    probe_checkpoint,
)
from experiments.cassandra_belief_factoring_2026_08.environment import (
    CassandraActionObservationEnv,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune
from learners.models.transformer import TransformerModel, TransformerModelConfig


TOTAL_ENV_STEPS = 5_000_000
SMOKE_ENV_STEPS = 4_096
TRAIN_BATCH_SIZE = 32_768
SMOKE_BATCH_SIZE = 2_048
MINIBATCH_SIZE = 2_048
SMOKE_MINIBATCH_SIZE = 256
ENTROPY_COEFF = 0.005
EPISODE_LENGTH = 1_000
MODEL_CONFIG = TransformerModelConfig(
    d_model=96,
    n_layers=3,
    n_heads=4,
    context_len=64,
).to_dict()


def environment_config(
    *,
    action_scope: str = "global",
) -> dict[str, Any]:
    """Return the hidden-symbol task with public diagnostics off."""

    return {
        "episode_length": EPISODE_LENGTH,
        "action_scope": action_scope,
        "diagnostics": False,
    }


def _apply_runtime_resources(config: PPOConfig, context: RunContext) -> PPOConfig:
    profile = context.hardware or PROFILES["cpu"]
    return config.env_runners(
        num_env_runners=(
            0
            if context.smoke
            else resolve_env_runners(profile, default=8)
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
    *,
    action_scope: str = "global",
) -> PPOConfig:
    """Build a fresh transformer PPO configuration."""

    batch_size = SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
    config = (
        PPOConfig()
        .environment(
            CassandraActionObservationEnv,
            env_config=environment_config(action_scope=action_scope),
        )
        .framework(
            "torch",
            torch_compile_learner=False,
            torch_compile_worker=False,
        )
        .training(
            lr=3e-4,
            gamma=DISCOUNT,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=ENTROPY_COEFF,
            train_batch_size_per_learner=batch_size,
            minibatch_size=(
                SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE
            ),
            num_epochs=4,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TransformerModel,
                model_config=dict(MODEL_CONFIG),
            )
        )
        .debugging(seed=context.seed)
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
    """Return retained checkpoints in sampled-step order."""

    candidates = list(result.best_checkpoints or [])
    if result.checkpoint is not None:
        candidates.append((result.checkpoint, result.metrics or {}))
    records = []
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


def log_spaced_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select iterations 1, 2, 4, ... and always retain the final one."""

    if not records:
        return []
    selected = [
        record
        for record in records
        if record["training_iteration"] > 0
        and record["training_iteration"].bit_count() == 1
    ]
    if not selected or (
        selected[-1]["checkpoint_name"] != records[-1]["checkpoint_name"]
    ):
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
    return result, {
        "agent_steps": agent_steps,
        "targets": result.metrics["targets"],
        "factor_specific": result.metrics["factor_specific"],
        "geometry": result.metrics["geometry"],
        "hypothesis_diagnostics": result.metrics["hypothesis_diagnostics"],
        "behavior_reward_mean": result.metrics["behavior_reward_mean"],
    }


def _training_change(
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "target_r_squared_delta": {
            name: (
                float(final["targets"][name]["r_squared"])
                - float(initial["targets"][name]["r_squared"])
            )
            for name in initial["targets"]
        },
        "coarse_over_identity_r2_advantage_delta": (
            float(
                final["hypothesis_diagnostics"][
                    "coarse_over_identity_r2_advantage"
                ]
            )
            - float(
                initial["hypothesis_diagnostics"][
                    "coarse_over_identity_r2_advantage"
                ]
            )
        ),
        "component_subspace_overlap_delta": (
            float(
                final["hypothesis_diagnostics"][
                    "mean_component_subspace_overlap"
                ]
            )
            - float(
                initial["hypothesis_diagnostics"][
                    "mean_component_subspace_overlap"
                ]
            )
        ),
    }


def run_condition(
    context: RunContext,
    *,
    action_scope: str = "global",
    condition: str = "global_actions_transformer_ppo",
    hypothesis: str = (
        "Global actions create pressure for a coarse, permutation-invariant "
        "machine representation rather than separate labeled component "
        "beliefs."
    ),
) -> dict[str, Any]:
    """Train PPO and probe initialization plus log-spaced checkpoints."""

    if context.seed is None:
        raise ValueError("Cassandra belief factoring requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    action_count = len(action_names(action_scope))
    outputs.write_json(
        "resolved_recipe.json",
        {
            "condition": condition,
            "hypothesis": hypothesis,
            "primary_comparison": (
                "trained-minus-initialization decodability of aggregate belief "
                "versus identity-specific component residuals"
            ),
            "algorithm": "PPO",
            "gamma": DISCOUNT,
            "lambda": 0.95,
            "entropy_coeff": ENTROPY_COEFF,
            "environment": environment_config(action_scope=action_scope),
            "policy_observation": (
                f"16-way symbol one-hot plus previous {action_count}-way "
                "action one-hot; no belief, hidden state, or reward"
            ),
            "model_config": MODEL_CONFIG,
            "total_env_steps": target_steps,
            "checkpoint_schedule": (
                "initialization_then_iterations_1_2_4_8_and_final"
            ),
            "probe_sampling": (
                "fixed checkpoint-independent behavior policy with disjoint "
                "train/test seed streams"
            ),
            "probe_representation": "pre_final_layer_norm_decision_token",
            "paper": "https://arxiv.org/abs/2602.02385",
            "controls": [
                "step-zero untrained transformer",
                "current-observation-plus-previous-action linear baseline",
                "target PCA dimensions",
            ],
        },
    )

    config = build_config(context, action_scope=action_scope)
    initial_checkpoint = _save_initial_checkpoint(
        config,
        context.artifacts_dir / "initial_checkpoint",
    )
    initial_probe, initial_point = _probe_at(
        context,
        checkpoint=initial_checkpoint,
        condition=f"{condition}_initialization",
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
    selected = log_spaced_records(checkpoint_records(result))
    if not selected:
        raise RuntimeError(f"{condition} retained no checkpoints")

    trajectory = [initial_point]
    trained_probes: list[ProbeResult] = []
    for record in selected:
        probe, point = _probe_at(
            context,
            checkpoint=Path(record["checkpoint"].path),
            condition=condition,
            agent_steps=record["agent_steps"],
        )
        trained_probes.append(probe)
        trajectory.append(
            {
                **point,
                "training_iteration": record["training_iteration"],
                "checkpoint_name": record["checkpoint_name"],
            }
        )
    final_probe = trained_probes[-1]
    plot_probe_trajectory(
        trajectory,
        path=context.results_dir / "probe_trajectory.png",
    )
    outputs.write_json(
        "checkpoint_probe_curve.json",
        {"condition": condition, "checkpoints": trajectory},
    )

    training_metrics = {
        "episode_return_mean": _metric(
            result.metrics or {},
            "env_runners/episode_return_mean",
        ),
        "episode_len_mean": _metric(
            result.metrics or {},
            "env_runners/episode_len_mean",
        ),
        "sampled_env_steps": _metric(
            result.metrics or {},
            "env_runners/num_env_steps_sampled_lifetime",
        ),
    }
    summary = {
        "condition": condition,
        "seed": context.seed,
        "smoke": context.smoke,
        "algorithm": "PPO",
        "training_metrics": training_metrics,
        "initial_probe": initial_probe.metrics,
        "final_probe": final_probe.metrics,
        "training_change": _training_change(
            initial_probe.metrics,
            final_probe.metrics,
        ),
        "checkpoint_probes": trajectory,
        "conclusion_status": (
            "smoke_diagnostic_only"
            if context.smoke
            else "single_seed_exploratory"
        ),
    }
    outputs.write_json("condition_summary.json", summary)
    return summary
