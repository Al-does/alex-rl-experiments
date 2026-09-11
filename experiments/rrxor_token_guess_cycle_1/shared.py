"""Gamma-zero PPO recipe and checkpoint analysis for RRXOR token guessing."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_PPO_2026_08.shared import (
    _save_initial_checkpoint,
    _save_log_spaced_checkpoint,
    checkpoint_records,
)
from experiments.rrxor_token_guess_cycle_1.model import (
    MODEL_CONFIG,
    RRXORActorCritic,
)
from experiments.rrxor_token_guess_cycle_1.process import (
    CONTEXT_LENGTH,
    EPISODE_LENGTH,
    environment_config,
)
from experiments.storage.training_curves import write_training_curves
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune


TOTAL_ENV_STEPS = 2_500_000
SMOKE_ENV_STEPS = 4_096
TRAIN_BATCH_SIZE = 32_768
SMOKE_BATCH_SIZE = 2_048
MINIBATCH_SIZE = 4_096
SMOKE_MINIBATCH_SIZE = 256
LEARNING_RATE = 1e-4
NUM_EPOCHS = 6


def build_config(context: RunContext) -> PPOConfig:
    """Build a fresh paper-scale gamma-zero PPO configuration."""

    if context.seed is None:
        raise ValueError("RRXOR token-guess PPO requires a resolved seed")
    profile = context.hardware or PROFILES["cpu"]
    return (
        PPOConfig()
        .environment(HMMEnv, env_config=environment_config())
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
            lr=LEARNING_RATE,
            gamma=0.0,
            lambda_=0.0,
            clip_param=0.2,
            use_critic=True,
            use_gae=True,
            use_kl_loss=False,
            vf_loss_coeff=0.5,
            entropy_coeff=0.0,
            train_batch_size_per_learner=(
                SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
            ),
            minibatch_size=(
                SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE
            ),
            num_epochs=NUM_EPOCHS,
            shuffle_batch_per_epoch=True,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=RRXORActorCritic,
                model_config=dict(MODEL_CONFIG),
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
            )
        )
    )


def resolved_recipe(context: RunContext) -> dict[str, Any]:
    """Return the complete scientific and operational recipe."""

    profile = context.hardware or PROFILES["cpu"]
    return {
        "study": "rrxor_token_guess_cycle_1",
        "condition": "ppo",
        "paper": "Shai et al., arXiv:2405.15943",
        "seed": context.seed,
        "smoke": context.smoke,
        "hypothesis": (
            "Gamma-zero PPO can learn the pending-token predictor from delayed "
            "RRXOR token history while exposing the exact Bayesian source belief "
            "across transformer depth."
        ),
        "primary_comparison": (
            "initial versus trained token accuracy and held-out affine belief "
            "decoding from each layer versus all layers concatenated"
        ),
        "controls": [
            "train-mean belief predictor",
            "current-visible-token and two-token-history predictors",
            "exact next-token probability and log-probability affine predictors",
            "policy-probability nuisance predictor",
            "held-out shuffled-label probes",
            "training-covariance-matched Gaussian features",
            "two-token-history-matched feature resampling",
            "true step-zero initialization",
        ],
        "environment": environment_config(),
        "process": {
            "name": "Random-Random-XOR",
            "hidden_states": 5,
            "tokens": 2,
            "reachable_belief_states": 36,
            "initial_distribution": "stationary",
        },
        "observation_semantics": (
            "one delayed token one-hot; zero vector before the first token"
        ),
        "action_semantics": "two categorical token-guess logits",
        "reward": (
            "1 when the sampled action equals event.raw_token_before, else 0"
        ),
        "leakage": {
            "previous_reward": False,
            "previous_action": False,
            "belief": False,
            "hidden_state": False,
        },
        "sequence_design": {
            "context_length": CONTEXT_LENGTH,
            "episode_length": EPISODE_LENGTH,
            "rationale": (
                "eleven decisions retain the empty prior plus ten "
                "history-conditioned guesses, matching paper-length prefixes"
            ),
        },
        "algorithm": "clipped PPO",
        "optimizer": "Adam (RLlib default)",
        "learning_rate": LEARNING_RATE,
        "gamma": 0.0,
        "lambda": 0.0,
        "clip_param": 0.2,
        "use_critic": True,
        "use_gae": True,
        "use_kl_loss": False,
        "value_loss_coeff": 0.5,
        "entropy_coeff": 0.0,
        "train_batch_size_per_learner": (
            SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
        ),
        "minibatch_size": (
            SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE
        ),
        "num_epochs": NUM_EPOCHS,
        "model": dict(MODEL_CONFIG),
        "total_env_steps": (
            SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
        ),
        "stopping_metric": (
            "env_runners/num_env_steps_sampled_lifetime"
        ),
        "success_metrics": [
            "held-out stochastic token accuracy",
            "Bayes expected token accuracy",
            "layerwise and concatenated held-out belief MSE/R2",
            "next-token-invisible belief contrast MSE/R2",
        ],
        "checkpoint_schedule": "initial, powers of two iterations, final",
        "analysis": {
            "target": (
                "exact float64 filtered edge-source belief reconstructed from "
                "public visible-token diagnostics"
            ),
            "sampling": "independent process-weighted paper-length prefixes",
            "representations": (
                "four pre-final-LN block outputs, their concatenation, and the "
                "post-final-LN policy embedding control"
            ),
        },
        "hardware": {
            "profile": profile.name,
            "learner_device": profile.learner_device,
            "intended": (
                "CPU smoke; selected hardware-profile learner for a full run"
            ),
        },
        "artifact_policy": {
            "compact_reports": "results/",
            "tune_checkpoints_raw_rollouts": "artifacts/",
        },
    }


def _headline(report: dict[str, Any]) -> dict[str, Any]:
    representations = report["geometry"]["representations"]
    concatenated = representations["concatenated_layers"]["metrics"]
    layer_reports = {
        name: value["metrics"]
        for name, value in representations.items()
        if name.startswith("layer_")
    }
    return {
        "agent_steps": report["agent_steps"],
        "training_iteration": report["training_iteration"],
        "token_accuracy": report["policy"]["token_accuracy"],
        "bayes_expected_accuracy": report["policy"]["bayes_expected_accuracy"],
        "belief_mse_per_layer": {
            name: value["mse"]
            for name, value in layer_reports.items()
        },
        "belief_r2_per_layer": {
            name: value["r_squared"]
            for name, value in layer_reports.items()
        },
        "concatenated_belief_mse": concatenated["mse"],
        "concatenated_belief_r2": concatenated["r_squared"],
        "concatenated_prediction_null_r2": {
            name: value["r_squared"]
            for name, value in concatenated["contrasts"].items()
        },
        "reachable_beliefs_observed": report["coverage"][
            "test_beliefs_observed"
        ],
    }


def run_condition(context: RunContext) -> dict[str, Any]:
    """Train one smoke/full condition and probe initialization plus checkpoints."""

    from experiments.rrxor_token_guess_cycle_1.analysis import analyze_checkpoint

    if context.seed is None:
        raise ValueError("RRXOR token-guess PPO requires a resolved seed")
    if context.resume_from is not None:
        raise ValueError("continuation is not defined for this experiment")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json("resolved_recipe.json", resolved_recipe(context))
    result_grid = run_tune(
        build_config(context),
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
    if len(results) != 1 or results[0].error is not None:
        raise RuntimeError("RRXOR token-guess PPO training failed")
    write_training_curves(context)
    records = [
        {
            "checkpoint_path": context.artifacts_dir / "initial_checkpoint",
            "checkpoint_name": "initial_checkpoint",
            "training_iteration": 0,
            "agent_steps": 0,
        },
        *checkpoint_records(
            results[0],
            checkpoint_root=(
                context.artifacts_dir / "log_spaced_checkpoints"
            ),
        ),
    ]
    reports = []
    for record in records:
        reports.append(
            analyze_checkpoint(
                replace(
                    context,
                    results_dir=(
                        context.results_dir
                        / "checkpoint_probes"
                        / f"steps_{record['agent_steps']:09d}"
                    ),
                    resume_from=Path(record["checkpoint_path"]),
                ),
                checkpoint=Path(record["checkpoint_path"]),
                checkpoint_label=record["checkpoint_name"],
                agent_steps=record["agent_steps"],
                training_iteration=record["training_iteration"],
            )
        )
    summary = {
        "condition": "ppo",
        "seed": context.seed,
        "smoke": context.smoke,
        "checkpoint_headlines": [_headline(report) for report in reports],
        "checkpoint_reports": [
            {
                "agent_steps": record["agent_steps"],
                "training_iteration": record["training_iteration"],
                "path": (
                    "checkpoint_probes/"
                    f"steps_{record['agent_steps']:09d}/probe_battery.json"
                ),
            }
            for record in records
        ],
    }
    outputs.write_json("condition_summary.json", summary)
    return summary
