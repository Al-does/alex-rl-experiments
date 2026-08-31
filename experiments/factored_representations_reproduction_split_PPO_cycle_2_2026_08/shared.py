"""Cycle-2 PPO recipe with fully separate actor and critic transformers."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_PPO_2026_08.analysis import (
    analyze_checkpoint,
    plot_probe_trajectory,
)
from experiments.factored_representations_reproduction_PPO_2026_08.model import (
    FactoredReproductionModelConfig,
)
from experiments.factored_representations_reproduction_PPO_2026_08.process import (
    FACTOR_COUNTS,
    environment_config,
    joint_token_count,
)
from experiments.factored_representations_reproduction_PPO_2026_08.shared import (
    SMOKE_BATCH_SIZE,
    SMOKE_ENV_STEPS,
    SMOKE_MINIBATCH_SIZE,
    TOTAL_ENV_STEPS,
    _save_initial_checkpoint,
    _save_log_spaced_checkpoint,
    checkpoint_records,
)
from experiments.factored_representations_reproduction_split_PPO_cycle_2_2026_08.learning import (
    AUXILIARY_COEFFICIENT,
    PPOWithNextJointTokenAux,
    next_joint_token_targets,
)
from experiments.factored_representations_reproduction_split_PPO_cycle_2_2026_08.model import (
    SplitActorCriticWithNextJointTokenAux,
    SplitFactoredReproductionActorCritic,
)
from experiments.storage.training_curves import write_training_curves
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune


MODEL_CONFIG = FactoredReproductionModelConfig().to_dict()
CONDITIONS = ("ppo", "ppo_aux_ce")
# Independent actor and critic transformers roughly double learner activation
# memory versus the shared-encoder PPO reproduction at the same batch size.
TRAIN_BATCH_SIZE = 16_384
MINIBATCH_SIZE = 16_384


def build_config(
    context: RunContext,
    *,
    factor_count: int,
    condition: str,
) -> PPOConfig:
    """Build one fresh split-network objective-by-environment configuration."""

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
        .environment(HMMEnv, env_config=environment_config(factor_count))
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
                    SplitActorCriticWithNextJointTokenAux
                    if auxiliary
                    else SplitFactoredReproductionActorCritic
                ),
                model_config=model_config,
            )
        )
        .callbacks(
            on_algorithm_init=partial(
                _save_initial_checkpoint,
                checkpoint_path=str(context.artifacts_dir / "initial_checkpoint"),
            ),
            on_train_result=partial(
                _save_log_spaced_checkpoint,
                checkpoint_root=str(
                    context.artifacts_dir / "log_spaced_checkpoints"
                ),
            ),
        )
        .debugging(seed=context.seed)
        .env_runners(
            num_env_runners=(
                0 if context.smoke else resolve_env_runners(profile, default=16)
            ),
            num_envs_per_env_runner=(
                1 if context.smoke else profile.num_envs_per_env_runner
            ),
            num_gpus_per_env_runner=0,
            sample_timeout_s=600.0,
        )
        .learners(
            num_gpus_per_learner=(
                1 if profile.learner_device == "cuda" else 0
            ),
        )
    )
    if auxiliary:
        config = config.learners(
            learner_class=PPOWithNextJointTokenAux,
            learner_config_dict={
                "next_token_aux/lambda": AUXILIARY_COEFFICIENT,
                "next_token_aux/target_extractor": next_joint_token_targets,
            },
        )
    return config


def _resolved_recipe(
    *,
    factor_count: int,
    condition: str,
    context: RunContext,
) -> dict[str, Any]:
    return {
        "cycle": 2,
        "source_recipe": "factored_representations_reproduction_PPO_2026_08",
        "paper": "Transformers learn factored representations (arXiv:2602.02385)",
        "condition": condition,
        "factor_count": factor_count,
        "factors": "conditionally independent MESS3(alpha=0.6, x=0.15)",
        "joint_token_count": joint_token_count(factor_count),
        "environment": environment_config(factor_count),
        "algorithm": "PPO",
        "objective": (
            "PPO correctness reward plus actor next-joint-token CE"
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
        "batch_size_rationale": (
            "Halved from the shared-encoder PPO recipe because separate actor "
            "and critic transformers OOM at 32,768 on RTX 4090."
        ),
        "model": MODEL_CONFIG,
        "architecture": (
            "independent actor and critic paper transformers with separate "
            "linear policy and value heads"
        ),
        "activation_location": (
            "actor final transformer block residual, after MLP and before final LN"
        ),
        "total_env_steps": (
            SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
        ),
        "checkpoint_schedule": "initial, powers of two iterations, final",
        "analysis": [
            "actor-only held-out joint regression to factor beliefs",
            "actor activation/factored-target/joint-target CEV",
            "actor vary-one per-position-centered PCA",
            "actor subspace additivity and principal-angle overlap",
            "actor token-embedding geometry",
        ],
    }


def run_factor_condition(
    context: RunContext,
    *,
    factor_count: int,
    condition: str,
) -> dict[str, Any]:
    """Train and probe one split-PPO design cell."""

    if context.seed is None:
        raise ValueError("the reproduction requires a resolved seed")
    if context.resume_from is not None:
        raise ValueError("split-PPO continuation is not defined for this recipe")
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
    result_grid = run_tune(
        build_config(context, factor_count=factor_count, condition=condition),
        context,
        stop={
            "env_runners/num_env_steps_sampled_lifetime": (
                SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
            )
        },
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
        raise RuntimeError("split PPO training failed") from result.error
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
            resume_from=Path(record["checkpoint_path"]),
        )
        reports.append(
            analyze_checkpoint(
                probe_context,
                checkpoint=Path(record["checkpoint_path"]),
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
    return {
        "condition": condition,
        "factor_count": factor_count,
        "seed": context.seed,
        "smoke": context.smoke,
        "checkpoint_reports": [
            {
                "agent_steps": report["agent_steps"],
                "checkpoint": report["checkpoint"],
                "training_iteration": report["training_iteration"],
                "path": str(
                    context.results_dir
                    / "checkpoint_probes"
                    / f"steps_{report['agent_steps']:09d}"
                    / "probe_battery.json"
                ),
            }
            for report in reports
        ],
        "trajectory_figure": str(context.results_dir / "probe_trajectory.png"),
    }


def run_arm(context: RunContext, condition: str) -> dict[str, Any]:
    """Run two- and three-factor cells for one objective arm."""

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
        "cycle": 2,
        "condition": condition,
        "seed": context.seed,
        "smoke": context.smoke,
        "factor_conditions": summaries,
    }
    outputs.write_json("arm_summary.json", result)
    return result

