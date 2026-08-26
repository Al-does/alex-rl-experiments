"""Two-arm PPO recipe and longitudinal factor-geometry analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_2026_08.analysis import (
    analyze_checkpoint,
    plot_probe_trajectory,
)
from experiments.factored_representations_reproduction_2026_08.learning import (
    AUXILIARY_COEFFICIENT,
    ActorCriticWithNextJointTokenAux,
    PPOWithNextJointTokenAux,
    next_joint_token_targets,
)
from experiments.factored_representations_reproduction_2026_08.model import (
    FactoredReproductionActorCritic,
    FactoredReproductionModelConfig,
)
from experiments.factored_representations_reproduction_2026_08.process import (
    FACTOR_COUNTS,
    environment_config,
    joint_token_count,
)
from experiments.storage.training_curves import write_training_curves
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune


TOTAL_ENV_STEPS = 5_000_000
SMOKE_ENV_STEPS = 1_024
TRAIN_BATCH_SIZE = 32_768
SMOKE_BATCH_SIZE = 1_024
# Live RTX 4090 profiling on the worst-case three-factor PPO+CE arm measured
# 5,420 env steps/s at 32,768 with 76.7% CUDA memory reserved. 65,536 OOMed.
MINIBATCH_SIZE = 32_768
SMOKE_MINIBATCH_SIZE = 256
MODEL_CONFIG = FactoredReproductionModelConfig().to_dict()
CONDITIONS = ("ppo", "ppo_aux_ce")


def _metric(metrics: Mapping[str, Any], path: str) -> float | None:
    direct = metrics.get(path)
    if isinstance(direct, Real):
        number = float(direct)
        return number if math.isfinite(number) else None
    value: Any = metrics
    for part in path.split("/"):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    if not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _save_initial_checkpoint(
    *,
    algorithm: Any,
    checkpoint_path: str,
    **_: Any,
) -> None:
    """Save step zero from the exact Algorithm instance Tune will optimize."""

    destination = Path(checkpoint_path)
    if (destination / "rllib_checkpoint.json").is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    algorithm.save_to_path(str(destination))


def _save_log_spaced_checkpoint(
    *,
    algorithm: Any,
    result: Mapping[str, Any],
    checkpoint_root: str,
    **_: Any,
) -> None:
    """Save public Algorithm checkpoints at power-of-two iterations."""

    iteration_value = _metric(result, "training_iteration")
    steps_value = _metric(
        result,
        "env_runners/num_env_steps_sampled_lifetime",
    )
    if iteration_value is None or steps_value is None:
        return
    iteration = int(iteration_value)
    if iteration <= 0 or iteration.bit_count() != 1:
        return
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    records = (
        json.loads(index_path.read_text()).get("checkpoints", [])
        if index_path.is_file()
        else []
    )
    if any(int(record["training_iteration"]) == iteration for record in records):
        return
    destination = root / (
        f"iteration_{iteration:06d}_steps_{int(steps_value):09d}"
    )
    saved = Path(algorithm.save_to_path(str(destination)))
    records.append(
        {
            "path": str(saved),
            "checkpoint_name": saved.name,
            "training_iteration": iteration,
            "agent_steps": int(steps_value),
        }
    )
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"checkpoints": records}, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(index_path)


def build_config(
    context: RunContext,
    *,
    factor_count: int,
    condition: str,
) -> PPOConfig:
    """Build one fresh objective-by-environment configuration."""

    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if factor_count not in FACTOR_COUNTS:
        raise ValueError(f"factor_count must be one of {FACTOR_COUNTS}")
    profile = context.hardware or PROFILES["cpu"]
    auxiliary = condition == "ppo_aux_ce"
    model_config = dict(MODEL_CONFIG)
    if auxiliary:
        model_config["next_token_aux"] = {
            "num_classes": joint_token_count(factor_count)
        }
    config = (
        PPOConfig()
        .environment(
            HMMEnv,
            env_config=environment_config(factor_count),
        )
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
            lr=1e-4,
            gamma=0.0,
            lambda_=0.0,
            clip_param=0.2,
            use_kl_loss=False,
            vf_loss_coeff=0.5,
            entropy_coeff=0.0,
            train_batch_size_per_learner=(
                SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
            ),
            minibatch_size=(
                SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE
            ),
            num_epochs=6,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=(
                    ActorCriticWithNextJointTokenAux
                    if auxiliary
                    else FactoredReproductionActorCritic
                ),
                model_config=model_config,
            )
        )
        .callbacks(
            on_algorithm_init=partial(
                _save_initial_checkpoint,
                checkpoint_path=str(
                    context.artifacts_dir / "initial_checkpoint"
                ),
            ),
            on_train_result=partial(
                _save_log_spaced_checkpoint,
                checkpoint_root=str(
                    context.artifacts_dir / "log_spaced_checkpoints"
                ),
            ),
        )
        .debugging(seed=context.seed)
    )
    if auxiliary:
        config = config.learners(
            learner_class=PPOWithNextJointTokenAux,
            learner_config_dict={
                "next_token_aux/lambda": AUXILIARY_COEFFICIENT,
                "next_token_aux/target_extractor": next_joint_token_targets,
            },
        )
    return config.env_runners(
        num_env_runners=(
            0 if context.smoke else resolve_env_runners(profile, default=16)
        ),
        num_envs_per_env_runner=(
            1 if context.smoke else profile.num_envs_per_env_runner
        ),
        num_gpus_per_env_runner=0,
        sample_timeout_s=600.0,
    ).learners(
        num_gpus_per_learner=(
            1 if profile.learner_device == "cuda" else 0
        ),
    )


def checkpoint_records(
    result: Any,
    *,
    checkpoint_root: Path,
) -> list[dict[str, Any]]:
    """Combine custom log-spaced checkpoints with Tune's final checkpoint."""

    by_iteration: dict[int, dict[str, Any]] = {}
    index_path = checkpoint_root / "index.json"
    if index_path.is_file():
        for record in json.loads(index_path.read_text()).get("checkpoints", []):
            iteration = int(record["training_iteration"])
            path = Path(record["path"])
            by_iteration[iteration] = {
                "checkpoint_path": path,
                "checkpoint_name": str(record.get("checkpoint_name", path.name)),
                "training_iteration": iteration,
                "agent_steps": int(record["agent_steps"]),
            }
    if result.checkpoint is not None:
        iteration = _metric(result.metrics or {}, "training_iteration")
        steps = _metric(
            result.metrics or {},
            "env_runners/num_env_steps_sampled_lifetime",
        )
        if iteration is not None and steps is not None:
            path = Path(result.checkpoint.path)
            by_iteration[int(iteration)] = {
                "checkpoint_path": path,
                "checkpoint_name": path.name,
                "training_iteration": int(iteration),
                "agent_steps": int(steps),
            }
    return [by_iteration[key] for key in sorted(by_iteration)]


def _resolved_recipe(
    *,
    factor_count: int,
    condition: str,
    context: RunContext,
) -> dict[str, Any]:
    return {
        "paper": "Transformers learn factored representations (arXiv:2602.02385)",
        "condition": condition,
        "factor_count": factor_count,
        "factors": "conditionally independent MESS3(alpha=0.6, x=0.15)",
        "joint_token_count": joint_token_count(factor_count),
        "environment": environment_config(factor_count),
        "algorithm": "PPO",
        "objective": (
            "PPO correctness reward plus next-joint-token CE"
            if condition == "ppo_aux_ce"
            else "PPO correctness reward"
        ),
        "next_token_aux_coefficient": (
            AUXILIARY_COEFFICIENT if condition == "ppo_aux_ce" else 0.0
        ),
        "gamma": 0.0,
        "lambda": 0.0,
        "learning_rate": 1e-4,
        "clip_param": 0.2,
        "num_epochs": 6,
        "train_batch_size_per_learner": TRAIN_BATCH_SIZE,
        "minibatch_size": MINIBATCH_SIZE,
        "model": MODEL_CONFIG,
        "head_choice_rationale": (
            "The paper uses three heads at d_model=120. Four 16-dimensional "
            "heads are the closest divisor-compatible choice at d_model=64."
        ),
        "activation_location": (
            "final transformer block residual, after MLP and before final LN"
        ),
        "total_env_steps": (
            SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
        ),
        "checkpoint_schedule": "initial, powers of two iterations, final",
        "analysis": [
            "held-out joint regression to concatenated factor beliefs",
            "activation/factored-target/joint-target CEV",
            "vary-one per-position-centered PCA",
            "effective-dimension additivity",
            "principal-angle overlap curves",
            "rank-2 projected belief regression",
            "token-embedding CEV, factor subspaces, and PC attribution",
        ],
    }


def run_factor_condition(
    context: RunContext,
    *,
    factor_count: int,
    condition: str,
) -> dict[str, Any]:
    """Train and analyze one cell of the 2x2 objective/environment design."""

    if context.seed is None:
        raise ValueError("the reproduction requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        _resolved_recipe(
            factor_count=factor_count,
            condition=condition,
            context=context,
        ),
    )
    config = build_config(
        context,
        factor_count=factor_count,
        condition=condition,
    )
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    result_grid = run_tune(
        config,
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": target_steps},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=True,
            )
        },
    )
    results = list(result_grid)
    if len(results) != 1:
        raise RuntimeError(f"expected one Tune trial, found {len(results)}")
    result = results[0]
    if result.error is not None:
        raise RuntimeError("PPO training failed") from result.error
    write_training_curves(context)

    records = [
        {
            "checkpoint_path": context.artifacts_dir / "initial_checkpoint",
            "checkpoint_name": "initial_checkpoint",
            "training_iteration": 0,
            "agent_steps": 0,
        },
        *checkpoint_records(
            result,
            checkpoint_root=context.artifacts_dir / "log_spaced_checkpoints",
        ),
    ]
    reports = []
    for record in records:
        probe_context = replace(
            context,
            results_dir=(
                context.results_dir
                / "checkpoint_probes"
                / f"steps_{record['agent_steps']:09d}"
            ),
            resume_from=record["checkpoint_path"],
        )
        reports.append(
            analyze_checkpoint(
                probe_context,
                checkpoint=record["checkpoint_path"],
                factor_count=factor_count,
                condition=condition,
                checkpoint_label=record["checkpoint_name"],
                agent_steps=record["agent_steps"],
                training_iteration=record["training_iteration"],
            )
        )
    plot_probe_trajectory(
        reports,
        condition=condition,
        factor_count=factor_count,
        path=context.results_dir / "probe_trajectory.png",
    )
    summary = {
        "condition": condition,
        "factor_count": factor_count,
        "seed": context.seed,
        "smoke": context.smoke,
        "checkpoints": reports,
        "initial": reports[0],
        "final": reports[-1],
        "trajectory_figure": str(context.results_dir / "probe_trajectory.png"),
    }
    outputs.write_json("condition_summary.json", summary)
    return summary


def run_arm(context: RunContext, condition: str) -> dict[str, Any]:
    """Run the two- and three-factor environments for one objective arm."""

    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    summaries = {}
    for factor_count in FACTOR_COUNTS:
        factor_context = replace(
            context,
            results_dir=context.results_dir / f"{factor_count}_factors",
            artifacts_dir=context.artifacts_dir / f"{factor_count}_factors",
            resume_from=None,
        )
        summaries[str(factor_count)] = run_factor_condition(
            factor_context,
            factor_count=factor_count,
            condition=condition,
        )
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    result = {
        "condition": condition,
        "seed": context.seed,
        "smoke": context.smoke,
        "factor_conditions": summaries,
    }
    outputs.write_json("arm_summary.json", result)
    return result
