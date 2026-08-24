"""Scientific recipe for transformer PPO on two independent MESS3 factors."""

from __future__ import annotations

from collections.abc import Mapping
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
from experiments.factored_mess3_beliefs_2026_08.analysis import (
    probe_checkpoint,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune
from learners.models import TransformerModel, TransformerModelConfig


TOTAL_ENV_STEPS = 2_500_000
SMOKE_ENV_STEPS = 4_096
TRAIN_BATCH_SIZE = 32_768
SMOKE_BATCH_SIZE = 2_048
MINIBATCH_SIZE = 8_192
SMOKE_MINIBATCH_SIZE = 256
EPISODE_LENGTH = 512
MODEL_CONFIG = TransformerModelConfig(
    d_model=64,
    n_layers=2,
    n_heads=4,
    context_len=32,
    max_seq_len=32,
).to_dict()
ENV_CONFIG = {
    "model": {
        "factory": "envs.hmm:factored_model",
        "kwargs": {
            "factors": [
                {
                    "factory": "envs.mess3.model:passive_model",
                    "kwargs": {"alpha": 0.85},
                },
                {
                    "factory": "envs.mess3.model:passive_model",
                    "kwargs": {"alpha": 0.85},
                },
            ],
        },
    },
    "task": {
        "class": "envs.mess3.tasks.state_guess:StateGuessTask",
    },
    "observation": {
        "token": {"offset": 0, "depth": 1},
        "action": None,
    },
    "delay": 0,
    "episode_length": EPISODE_LENGTH,
    "randomize_first_episode_length": True,
}


def _save_initial_checkpoint(
    *,
    algorithm: Any,
    checkpoint_path: str,
    **_: Any,
) -> None:
    """Save the exact initialized Algorithm instance Tune will train."""

    destination = Path(checkpoint_path)
    if (destination / "rllib_checkpoint.json").is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    algorithm.save_to_path(str(destination))


def _apply_runtime_resources(
    config: PPOConfig,
    context: RunContext,
) -> PPOConfig:
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
        num_gpus_per_env_runner=0,
        sample_timeout_s=600.0,
    ).learners(
        num_gpus_per_learner=(
            1 if profile.learner_device == "cuda" else 0
        )
    )


def build_config(context: RunContext) -> PPOConfig:
    """Build a fresh gamma-zero transformer PPO configuration."""

    batch_size = SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
    profile = context.hardware or PROFILES["cpu"]
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
            vf_loss_coeff=0.5,
            entropy_coeff=0.002,
            train_batch_size_per_learner=batch_size,
            minibatch_size=(
                SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE
            ),
            num_epochs=4,
            shuffle_batch_per_epoch=True,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TransformerModel,
                model_config=dict(MODEL_CONFIG),
            )
        )
        .callbacks(
            on_algorithm_init=partial(
                _save_initial_checkpoint,
                checkpoint_path=str(
                    context.artifacts_dir / "initial_checkpoint"
                ),
            )
        )
        .debugging(seed=context.seed)
    )
    return _apply_runtime_resources(config, context)


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


def _probe_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "joint_belief_r_squared": metrics["targets"]["joint_belief"][
            "r_squared"
        ],
        "factor_r_squared": {
            name: payload["r_squared"]
            for name, payload in metrics["targets"].items()
            if name.startswith("factor_")
        },
        "observation_only_factor_r_squared": {
            name: payload["r_squared"]
            for name, payload in metrics["observation_only_baselines"].items()
            if name.startswith("factor_")
        },
        "cev95_dimension": metrics["geometry"]["activation_pca"][
            "cev95_dimension"
        ],
        "dimension_predictions": metrics["dimension_predictions"],
        "factor_subspace_overlap": metrics["geometry"][
            "mean_pairwise_subspace_overlap"
        ],
    }


def run_independent(context: RunContext) -> dict[str, Any]:
    """Train one independent two-factor condition, then probe init and final."""

    if context.seed is None:
        raise ValueError("factored MESS3 training requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    outputs.write_json(
        "resolved_recipe.json",
        {
            "hypothesis": (
                "PPO's 64-dimensional transformer will encode each independent "
                "MESS3 belief in a linearly decodable, approximately orthogonal "
                "two-dimensional subspace, using about four activation "
                "dimensions rather than the eight-dimensional joint simplex."
            ),
            "primary_comparison": "step_zero_initialization_vs_final_checkpoint",
            "generator": "two independent passive MESS3 HMM factors",
            "observed_token": (
                "one nine-way token in one-to-one correspondence with the "
                "Cartesian pair of three-way factor subtokens"
            ),
            "task": "guess the current joint hidden state (nine actions)",
            "algorithm": "PPO",
            "gamma": 0.0,
            "lambda": 0.0,
            "environment": ENV_CONFIG,
            "model": MODEL_CONFIG,
            "total_env_steps": target_steps,
            "train_batch_size_per_learner": (
                SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
            ),
            "minibatch_size": (
                SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE
            ),
            "probe_controls": [
                "exact step-zero checkpoint",
                "disjoint train/test rollout seeds",
                "current-token observation-only affine baseline",
                "joint-versus-factored dimension predictions",
            ],
            "paper": "https://arxiv.org/abs/2602.02385",
        },
    )

    config = build_config(context)
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
        raise RuntimeError(f"expected one Tune trial, got {len(results)}")
    result = results[0]
    if result.error is not None:
        raise RuntimeError("two-factor PPO training failed") from result.error
    if result.checkpoint is None:
        raise RuntimeError("two-factor PPO training produced no final checkpoint")
    initial_checkpoint = context.artifacts_dir / "initial_checkpoint"
    if not (initial_checkpoint / "rllib_checkpoint.json").is_file():
        raise RuntimeError("the exact initialization checkpoint was not saved")
    final_checkpoint = Path(result.checkpoint.path)
    final_steps = int(
        _metric(
            result.metrics or {},
            "env_runners/num_env_steps_sampled_lifetime",
        )
        or target_steps
    )

    initial_metrics = probe_checkpoint(
        context,
        checkpoint=initial_checkpoint,
        agent_steps=0,
    )
    initial_probe_path = context.results_dir / "probe_metrics.json"
    initial_cev_path = context.results_dir / "cev.png"
    initial_dir = context.results_dir / "initial"
    initial_dir.mkdir(parents=True, exist_ok=True)
    initial_probe_path.replace(initial_dir / "probe_metrics.json")
    initial_cev_path.replace(initial_dir / "cev.png")

    final_metrics = probe_checkpoint(
        context,
        checkpoint=final_checkpoint,
        agent_steps=final_steps,
    )
    final_probe_path = context.results_dir / "probe_metrics.json"
    final_cev_path = context.results_dir / "cev.png"
    final_dir = context.results_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_probe_path.replace(final_dir / "probe_metrics.json")
    final_cev_path.replace(final_dir / "cev.png")

    initial_summary = _probe_summary(initial_metrics)
    final_summary = _probe_summary(final_metrics)
    summary = {
        "seed": context.seed,
        "smoke": context.smoke,
        "sampled_env_steps": final_steps,
        "training_episode_return_mean": _metric(
            result.metrics or {},
            "env_runners/episode_return_mean",
        ),
        "initial_probe": initial_summary,
        "final_probe": final_summary,
        "training_change": {
            "joint_belief_r_squared_delta": (
                final_summary["joint_belief_r_squared"]
                - initial_summary["joint_belief_r_squared"]
            ),
            "factor_r_squared_delta": {
                name: (
                    final_summary["factor_r_squared"][name]
                    - initial_summary["factor_r_squared"][name]
                )
                for name in final_summary["factor_r_squared"]
            },
            "cev95_dimension_delta": (
                final_summary["cev95_dimension"]
                - initial_summary["cev95_dimension"]
            ),
            "factor_subspace_overlap_delta": (
                final_summary["factor_subspace_overlap"]
                - initial_summary["factor_subspace_overlap"]
            ),
        },
        "conclusion_status": (
            "smoke_diagnostic_only"
            if context.smoke
            else "single_seed_exploratory"
        ),
        "interpretation_rule": (
            "Evidence for native factoring requires both factor beliefs to be "
            "linearly decodable, CEV near the four-dimensional direct-sum "
            "prediction rather than the eight-dimensional joint prediction, "
            "and low factor-subspace overlap."
        ),
    }
    outputs.write_json("condition_summary.json", summary)
    required = [
        context.results_dir / "resolved_recipe.json",
        context.results_dir / "tune_summary.json",
        context.results_dir / "condition_summary.json",
        initial_dir / "probe_metrics.json",
        initial_dir / "cev.png",
        final_dir / "probe_metrics.json",
        final_dir / "cev.png",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("missing compact outputs: " + ", ".join(missing))
    (context.results_dir / "output_validation.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "required_files": [
                    str(path.relative_to(context.results_dir))
                    for path in required
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return summary
