"""Cycle-2 discrete-SAC recipe with focused entropy and auxiliary sweeps."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
import math
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_PPO_2026_08.model import (
    FactoredReproductionModelConfig,
)
from experiments.factored_representations_reproduction_PPO_2026_08.shared import (
    _save_initial_checkpoint,
    _save_log_spaced_checkpoint,
    checkpoint_records,
)
from experiments.factored_representations_reproduction_SAC_2026_08.analysis import (
    analyze_checkpoint,
    plot_probe_trajectory,
)
from experiments.factored_representations_reproduction_SAC_2026_08.learning import (
    SACWithNextJointTokenAux,
)
from experiments.factored_representations_reproduction_SAC_2026_08.model import (
    FactoredReproductionSAC,
    ReproductionSACCatalog,
)
from experiments.factored_representations_reproduction_SAC_2026_08.process import (
    FACTOR_COUNTS,
    environment_config,
    joint_token_count,
)
from experiments.storage.training_curves import write_training_curves
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune
from losses.next_token import LAMBDA_KEY

TOTAL_ENV_STEPS = 50_000_000
SMOKE_ENV_STEPS = 128
TRAIN_BATCH_SIZE = 256
SMOKE_BATCH_SIZE = 64
LEARNING_STARTS = 10_000
SMOKE_LEARNING_STARTS = 32
REPLAY_CAPACITY = 1_000_000
SMOKE_REPLAY_CAPACITY = 1_024
PRIORITIZED_REPLAY_ALPHA = 0.6
PRIORITIZED_REPLAY_BETA = 0.6
TRAINING_INTENSITY = 1.0
TARGET_ENTROPY_FRACTIONS = (0.3, 0.6)
AUXILIARY_COEFFICIENTS = (0.1, 0.3)
CONDITIONS = ("sac", "sac_aux_ce")
MODEL_CONFIG = {
    **FactoredReproductionModelConfig().to_dict(),
    "head_fcnet_hiddens": [],
}


def _validate_cell(
    *,
    factor_count: int,
    condition: str,
    target_entropy_fraction: float,
    auxiliary_coefficient: float | None,
) -> None:
    if factor_count not in FACTOR_COUNTS:
        raise ValueError(f"factor_count must be one of {FACTOR_COUNTS}")
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if target_entropy_fraction not in TARGET_ENTROPY_FRACTIONS:
        raise ValueError(
            f"target_entropy_fraction must be one of {TARGET_ENTROPY_FRACTIONS}"
        )
    if condition == "sac_aux_ce":
        if auxiliary_coefficient not in AUXILIARY_COEFFICIENTS:
            raise ValueError(
                f"auxiliary_coefficient must be one of {AUXILIARY_COEFFICIENTS}"
            )
    elif auxiliary_coefficient is not None:
        raise ValueError("reward-only SAC does not accept an auxiliary coefficient")


def target_entropy(factor_count: int, fraction: float) -> float:
    """Return a positive entropy target as a fraction of categorical maximum."""

    if factor_count not in FACTOR_COUNTS:
        raise ValueError(f"factor_count must be one of {FACTOR_COUNTS}")
    if fraction not in TARGET_ENTROPY_FRACTIONS:
        raise ValueError(f"fraction must be one of {TARGET_ENTROPY_FRACTIONS}")
    return fraction * math.log(joint_token_count(factor_count))


def build_config(
    context: RunContext,
    *,
    factor_count: int,
    condition: str,
    target_entropy_fraction: float,
    auxiliary_coefficient: float | None = None,
) -> SACConfig:
    """Build one cycle-2 objective-by-entropy-by-environment configuration."""

    _validate_cell(
        factor_count=factor_count,
        condition=condition,
        target_entropy_fraction=target_entropy_fraction,
        auxiliary_coefficient=auxiliary_coefficient,
    )
    profile = context.hardware or PROFILES["cpu"]
    auxiliary = condition == "sac_aux_ce"
    model_config = dict(MODEL_CONFIG)
    if auxiliary:
        model_config["next_token_aux"] = {
            "num_classes": joint_token_count(factor_count)
        }

    config = (
        SACConfig()
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
            gamma=0.0,
            n_step=1,
            twin_q=True,
            tau=0.005,
            actor_lr=3e-5,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            target_entropy=target_entropy(
                factor_count,
                target_entropy_fraction,
            ),
            train_batch_size_per_learner=(
                SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
            ),
            training_intensity=TRAINING_INTENSITY,
            num_steps_sampled_before_learning_starts=(
                SMOKE_LEARNING_STARTS if context.smoke else LEARNING_STARTS
            ),
            replay_buffer_config={
                "type": "PrioritizedEpisodeReplayBuffer",
                "capacity": (
                    SMOKE_REPLAY_CAPACITY if context.smoke else REPLAY_CAPACITY
                ),
                "alpha": PRIORITIZED_REPLAY_ALPHA,
                "beta": PRIORITIZED_REPLAY_BETA,
            },
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=FactoredReproductionSAC,
                catalog_class=ReproductionSACCatalog,
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
            rollout_fragment_length=1,
            sample_timeout_s=600.0,
        )
        .learners(
            num_gpus_per_learner=(
                1 if profile.learner_device == "cuda" else 0
            ),
        )
        .reporting(
            min_sample_timesteps_per_iteration=(64 if context.smoke else 100),
            min_time_s_per_iteration=0 if context.smoke else 1,
        )
    )
    if auxiliary:
        config = config.learners(
            learner_class=SACWithNextJointTokenAux,
            learner_config_dict={
                LAMBDA_KEY: auxiliary_coefficient,
            },
        )
    return config


def _resolved_recipe(
    *,
    factor_count: int,
    condition: str,
    target_entropy_fraction: float,
    auxiliary_coefficient: float | None,
    context: RunContext,
) -> dict[str, Any]:
    auxiliary = condition == "sac_aux_ce"
    return {
        "cycle": 2,
        "source_recipe": "PR #63",
        "paper": "Transformers learn factored representations (arXiv:2602.02385)",
        "condition": condition,
        "factor_count": factor_count,
        "factors": "conditionally independent MESS3(alpha=0.6, x=0.15)",
        "joint_token_count": joint_token_count(factor_count),
        "environment": environment_config(factor_count),
        "algorithm": "discrete SAC",
        "objective": (
            "discrete SAC correctness reward plus next-joint-token CE"
            if auxiliary
            else "discrete SAC correctness reward"
        ),
        "next_token_aux_coefficient": (
            auxiliary_coefficient if auxiliary else 0.0
        ),
        "target_entropy_fraction_of_categorical_maximum": target_entropy_fraction,
        "target_entropy": target_entropy(
            factor_count,
            target_entropy_fraction,
        ),
        "gamma": 0.0,
        "n_step": 1,
        "twin_q": True,
        "tau": 0.005,
        "actor_learning_rate": 3e-5,
        "critic_learning_rate": 3e-4,
        "alpha_learning_rate": 3e-4,
        "train_batch_size_per_learner": TRAIN_BATCH_SIZE,
        "training_intensity": TRAINING_INTENSITY,
        "learning_starts": LEARNING_STARTS,
        "replay_capacity": REPLAY_CAPACITY,
        "prioritized_replay_alpha": PRIORITIZED_REPLAY_ALPHA,
        "prioritized_replay_beta": PRIORITIZED_REPLAY_BETA,
        "model": MODEL_CONFIG,
        "architecture": (
            "independent actor, critic, and twin-critic paper transformers; "
            "one linear head per network"
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
    target_entropy_fraction: float,
    auxiliary_coefficient: float | None = None,
) -> dict[str, Any]:
    """Train and analyze one cycle-2 design cell."""

    _validate_cell(
        factor_count=factor_count,
        condition=condition,
        target_entropy_fraction=target_entropy_fraction,
        auxiliary_coefficient=auxiliary_coefficient,
    )
    if context.seed is None:
        raise ValueError("the reproduction requires a resolved seed")
    if context.resume_from is not None:
        raise ValueError("SAC continuation is not defined for this recipe")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        _resolved_recipe(
            factor_count=factor_count,
            condition=condition,
            target_entropy_fraction=target_entropy_fraction,
            auxiliary_coefficient=auxiliary_coefficient,
            context=context,
        ),
    )
    result_grid = run_tune(
        build_config(
            context,
            factor_count=factor_count,
            condition=condition,
            target_entropy_fraction=target_entropy_fraction,
            auxiliary_coefficient=auxiliary_coefficient,
        ),
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
        raise RuntimeError("cycle-2 discrete SAC training failed") from result.error
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
        "target_entropy_fraction": target_entropy_fraction,
        "target_entropy": target_entropy(
            factor_count,
            target_entropy_fraction,
        ),
        "auxiliary_coefficient": auxiliary_coefficient,
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


def run_arm(
    context: RunContext,
    *,
    condition: str,
    target_entropy_fraction: float,
    auxiliary_coefficient: float | None = None,
) -> dict[str, Any]:
    """Run both factor counts for one cycle-2 hyperparameter arm."""

    _validate_cell(
        factor_count=FACTOR_COUNTS[0],
        condition=condition,
        target_entropy_fraction=target_entropy_fraction,
        auxiliary_coefficient=auxiliary_coefficient,
    )
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
            target_entropy_fraction=target_entropy_fraction,
            auxiliary_coefficient=auxiliary_coefficient,
        )
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    result = {
        "cycle": 2,
        "condition": condition,
        "seed": context.seed,
        "smoke": context.smoke,
        "target_entropy_fraction": target_entropy_fraction,
        "auxiliary_coefficient": auxiliary_coefficient,
        "factor_conditions": summaries,
    }
    outputs.write_json("arm_summary.json", result)
    return result
