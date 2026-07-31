"""Plain-PPO recipe for the first MESS3 action-feedback study."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.mess3_feedback_cycle_1.analysis import (
    ProbeResult,
    plot_init_final,
    plot_probe_pair,
    plot_probe_trajectory,
    probe_checkpoint,
)
from experiments.mess3_token_guess_cycle_2.model import (
    PaperActorCriticConfig,
    PaperActorCriticModel,
)
from experiments.mess3_token_guess_cycle_2.shared import (
    _apply_runtime_resources,
    _save_initial_checkpoint,
    checkpoint_records,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.runners import run_tune


ETA = 0.10
TOTAL_ENV_STEPS = 2_000_000
SMOKE_ENV_STEPS = 4_096
TRAIN_BATCH_SIZE = 32_768
SMOKE_BATCH_SIZE = 2_048
MINIBATCH_SIZE = 4_096
SMOKE_MINIBATCH_SIZE = 256
CHECKPOINT_FREQUENCY = 10
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
CONDITION = "ppo_feedback_eta_0p10"
BASE_MODEL_CONFIG = PaperActorCriticConfig().to_dict()
ENV_CONFIG = {
    "model": {
        "factory": "envs.mess3.model:passive_model",
        "kwargs": {"alpha": 0.85},
    },
    "task": {
        "class": "experiments.mess3_feedback_cycle_1.task:FeedbackNextTokenTask",
        "kwargs": {"eta": ETA},
    },
    "observation": {
        "token": {"offset": 0, "depth": 1},
        "action": {"offset": 0, "depth": 1},
    },
    "delay": 1,
    "episode_length": 512,
    "randomize_first_episode_length": True,
}


def build_config(context: RunContext) -> PPOConfig:
    """Build the paper-scale, gamma-zero plain PPO configuration."""

    profile = context.hardware or PROFILES["cpu"]
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
            lr=1e-4,
            gamma=0.0,
            lambda_=0.0,
            clip_param=0.2,
            use_kl_loss=False,
            vf_loss_coeff=0.5,
            entropy_coeff=0.0,
            train_batch_size_per_learner=batch_size,
            minibatch_size=(
                SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE
            ),
            num_epochs=6,
            shuffle_batch_per_epoch=True,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=PaperActorCriticModel,
                model_config=dict(BASE_MODEL_CONFIG),
            )
        )
        .debugging(seed=context.seed)
    )
    return _apply_runtime_resources(config, context)


def _probe_at(
    context: RunContext,
    *,
    checkpoint: Path,
    agent_steps: int,
    final: bool,
) -> tuple[ProbeResult, dict[str, Any]]:
    probe_dir = context.results_dir / "checkpoint_probes" / (
        f"steps_{agent_steps:09d}"
    )
    result = probe_checkpoint(
        replace(context, results_dir=probe_dir, resume_from=checkpoint),
        checkpoint=checkpoint,
        condition=CONDITION if agent_steps else f"{CONDITION}_init",
        agent_steps=agent_steps,
        run_causal_evaluations=final,
    )
    point = {
        "agent_steps": agent_steps,
        "mse": float(result.metrics["mse"]),
        "target_variance": float(result.metrics["target_variance"]),
        "global_mse_ratio": float(result.metrics["global_mse_ratio"]),
        "fine_mse_ratio": float(result.metrics["fine_mse_ratio"]),
        "branch_baseline_mse": float(result.metrics["branch_baseline_mse"]),
        "r_squared": float(result.metrics["r_squared"]),
        "token_accuracy_greedy": float(
            result.metrics["token_accuracy_greedy"]
        ),
        "bayesian_optimal_accuracy_on_rollout": float(
            result.metrics["bayesian_optimal_accuracy_on_rollout"]
        ),
        "probe": result.metrics,
    }
    return result, point


def run_condition(context: RunContext) -> dict[str, Any]:
    """Train to 2M steps, probe every checkpoint, and causally test the final."""

    if context.seed is None:
        raise ValueError("feedback cycle 1 requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    checkpoint_frequency = 1 if context.smoke else CHECKPOINT_FREQUENCY
    config = build_config(context)
    outputs.write_json(
        "resolved_recipe.json",
        {
            "study": "mess3_feedback_cycle_1",
            "condition": CONDITION,
            "algorithm": "PPO",
            "objective": "clipped_next_token_correctness",
            "feedback_transition": (
                "U_a = (1 - eta) * T + eta * R_a; R_a sends every state to a"
            ),
            "eta": ETA,
            "lr": 1e-4,
            "gamma": 0.0,
            "lambda": 0.0,
            "environment": ENV_CONFIG,
            "model": BASE_MODEL_CONFIG,
            "total_env_steps": target_steps,
            "train_batch_size_per_learner": config.train_batch_size_per_learner,
            "minibatch_size": config.minibatch_size,
            "num_epochs": config.num_epochs,
            "vf_loss_coeff": config.vf_loss_coeff,
            "checkpoint_frequency_iterations": checkpoint_frequency,
            "causal_evaluations": [
                "closed_loop_previous_action_mask_and_shuffle",
                "counterfactual_previous_action_belief_shift",
            ],
        },
    )

    initial_checkpoint = _save_initial_checkpoint(
        config,
        context.artifacts_dir / "initial_checkpoint",
    )
    initial_probe, initial_point = _probe_at(
        context,
        checkpoint=initial_checkpoint,
        agent_steps=0,
        final=False,
    )
    plot_probe_pair(
        initial_probe,
        title=f"{CONDITION} — init",
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
        raise RuntimeError(f"{CONDITION} expected one trial, got {len(results)}")
    result = results[0]
    if result.error is not None:
        raise RuntimeError(f"{CONDITION} training failed") from result.error
    checkpoints = checkpoint_records(result)
    if not checkpoints:
        raise RuntimeError(f"{CONDITION} retained no checkpoints")

    trajectory = [initial_point]
    checkpoint_probes: list[ProbeResult] = []
    for index, record in enumerate(checkpoints):
        is_final = index == len(checkpoints) - 1
        probed, point = _probe_at(
            context,
            checkpoint=Path(record["checkpoint"].path),
            agent_steps=record["agent_steps"],
            final=is_final,
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
        title=f"{CONDITION} — final",
        path=context.results_dir / "belief_simplex_final.png",
    )
    plot_init_final(
        initial_probe,
        final_probe,
        condition=CONDITION,
        path=context.results_dir / "belief_simplex_init_vs_final.png",
    )
    plot_probe_trajectory(
        trajectory,
        path=context.results_dir / "probe_and_success_trajectory.png",
    )
    outputs.write_json(
        "checkpoint_probe_curve.json",
        {"condition": CONDITION, "checkpoints": trajectory},
    )
    summary = {
        "condition": CONDITION,
        "seed": context.seed,
        "smoke": context.smoke,
        "eta": ETA,
        "gamma": 0.0,
        "algorithm": "PPO",
        "initial_probe": initial_probe.metrics,
        "final_probe": final_probe.metrics,
        "causal_evaluations": final_probe.metrics["causal_evaluations"],
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
