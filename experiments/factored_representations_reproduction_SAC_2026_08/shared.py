"""Discrete-SAC recipe and actor-representation checkpoint analysis."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
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
    AUXILIARY_COEFFICIENT,
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

TOTAL_ENV_STEPS = 50_000_000
SMOKE_ENV_STEPS = 128
TRAIN_BATCH_SIZE = 256
SMOKE_BATCH_SIZE = 64
LEARNING_STARTS = 1_500
SMOKE_LEARNING_STARTS = 32
REPLAY_CAPACITY = 1_000_000
SMOKE_REPLAY_CAPACITY = 1_024
MODEL_CONFIG = {
    **FactoredReproductionModelConfig().to_dict(),
    # Empty RLlib head stacks make both actor and Q heads single linear maps.
    "head_fcnet_hiddens": [],
}
CONDITIONS = ("sac", "sac_aux_ce")


def build_config(
    context: RunContext,
    *,
    factor_count: int,
    condition: str,
) -> SACConfig:
    """Build one fresh discrete-SAC objective-by-environment configuration."""

    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if factor_count not in FACTOR_COUNTS:
        raise ValueError(f"factor_count must be one of {FACTOR_COUNTS}")
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
            train_batch_size_per_learner=(
                SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
            ),
            num_steps_sampled_before_learning_starts=(
                SMOKE_LEARNING_STARTS if context.smoke else LEARNING_STARTS
            ),
            replay_buffer_config={
                "type": "PrioritizedEpisodeReplayBuffer",
                "capacity": (
                    SMOKE_REPLAY_CAPACITY if context.smoke else REPLAY_CAPACITY
                ),
                "alpha": 0.6,
                "beta": 0.4,
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
            min_sample_timesteps_per_iteration=(
                64 if context.smoke else 100
            ),
            min_time_s_per_iteration=0 if context.smoke else 1,
        )
    )
    if auxiliary:
        config = config.learners(
            learner_class=SACWithNextJointTokenAux,
            learner_config_dict={
                "next_token_aux/lambda": AUXILIARY_COEFFICIENT,
            },
        )
    return config


def _resolved_recipe(
    *,
    factor_count: int,
    condition: str,
    context: RunContext,
) -> dict[str, Any]:
    auxiliary = condition == "sac_aux_ce"
    return {
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
            AUXILIARY_COEFFICIENT if auxiliary else 0.0
        ),
        "gamma": 0.0,
        "n_step": 1,
        "twin_q": True,
        "tau": 0.005,
        "actor_learning_rate": 3e-5,
        "critic_learning_rate": 3e-4,
        "alpha_learning_rate": 3e-4,
        "train_batch_size_per_learner": TRAIN_BATCH_SIZE,
        "learning_starts": LEARNING_STARTS,
        "replay_capacity": REPLAY_CAPACITY,
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
) -> dict[str, Any]:
    """Train and analyze one cell of the SAC objective-by-factor design."""

    if context.seed is None:
        raise ValueError("the reproduction requires a resolved seed")
    if context.resume_from is not None:
        raise ValueError("SAC continuation is not defined for this new recipe")
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
        build_config(
            context,
            factor_count=factor_count,
            condition=condition,
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
        raise RuntimeError("discrete SAC training failed") from result.error
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
    """Run both factor counts for one SAC objective arm."""

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
