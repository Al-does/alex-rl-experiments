"""Shared PPO recipe for the three two-factor reward arms."""

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
from experiments.storage.training_curves import write_training_curves
from experiments.two_factor_reward_state_PPO_cycle_1.analysis import (
    analyze_checkpoint,
    plot_probe_trajectory,
)
from experiments.two_factor_reward_state_PPO_cycle_1.design import (
    GAMMA,
    demand_audit,
)
from experiments.two_factor_reward_state_PPO_cycle_1.model import (
    FactoredReproductionModelConfig,
    TwoFactorRewardPPO,
)
from experiments.two_factor_reward_state_PPO_cycle_1.process import (
    CONTEXT_LENGTH,
    FACTOR_COUNT,
    JOINT_TOKEN_COUNT,
    MESS3_ALPHA,
    TRANSITION_MATRIX,
    environment_config,
)
from experiments.two_factor_reward_state_PPO_cycle_1.task import (
    ACTION_LABELS,
    ACTION_PAIRS,
    CONDITIONS,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune


TOTAL_ENV_STEPS = 10_000_000
SMOKE_ENV_STEPS = 1_024
# A 64-frame action-aware transformer OOMs at the factored-reproduction 32k
# batch on a single RTX 4090; these sizes match other long-context PPO arms.
TRAIN_BATCH_SIZE = 8_192
SMOKE_BATCH_SIZE = 1_024
MINIBATCH_SIZE = 2_048
SMOKE_MINIBATCH_SIZE = 256
LEARNING_RATE = 3e-4
GAE_LAMBDA = 0.95
NUM_EPOCHS = 6
MODEL_CONFIG = FactoredReproductionModelConfig(
    d_model=64,
    n_layers=4,
    n_heads=4,
    d_mlp=256,
    context_length=CONTEXT_LENGTH,
    max_seq_len=CONTEXT_LENGTH,
).to_dict()


def build_config(context: RunContext, condition: str) -> PPOConfig:
    """Build one fresh nine-action recurrent-PPO configuration."""

    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    profile = context.hardware or PROFILES["cpu"]
    return (
        PPOConfig()
        .environment(HMMEnv, env_config=environment_config(condition))
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
            gamma=GAMMA,
            lambda_=GAE_LAMBDA,
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
            num_epochs=NUM_EPOCHS,
            shuffle_batch_per_epoch=True,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TwoFactorRewardPPO,
                model_config=dict(MODEL_CONFIG),
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


def _resolved_recipe(
    context: RunContext,
    condition: str,
    audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "study": "two_factor_reward_state_PPO_cycle_1",
        "condition": condition,
        "hypothesis": (
            "rewarding one versus both controlled factors changes which exact "
            "factor beliefs become linearly accessible in PPO's shared encoder"
        ),
        "factor_count": FACTOR_COUNT,
        "factor_transition_matrix": TRANSITION_MATRIX.tolist(),
        "emission_alpha": MESS3_ALPHA,
        "joint_token_count": JOINT_TOKEN_COUNT,
        "action_pairs_by_flat_index": [list(pair) for pair in ACTION_PAIRS],
        "action_labels_by_flat_index": list(ACTION_LABELS),
        "reward_state": 2,
        "environment": environment_config(condition),
        "algorithm": "PPO",
        "gamma": GAMMA,
        "lambda": GAE_LAMBDA,
        "learning_rate": LEARNING_RATE,
        "clip_param": 0.2,
        "use_kl_loss": False,
        "vf_loss_coeff": 0.5,
        "entropy_coeff": 0.0,
        "train_batch_size_per_learner": TRAIN_BATCH_SIZE,
        "minibatch_size": MINIBATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "model": MODEL_CONFIG,
        "total_env_steps": (
            SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
        ),
        "checkpoint_schedule": "initial, powers of two iterations, final",
        "pretraining_demand_audit": audit,
        "analysis": {
            "representation": (
                "shared actor-critic final transformer block residual before "
                "final LayerNorm"
            ),
            "targets": [
                "exact action-conditioned joint predictive belief",
                "exact action-conditioned factor-1 predictive belief",
                "exact action-conditioned factor-2 predictive belief",
            ],
            "metrics": ["held-out MSE/RMSE/R2", "PCA dimensions at 95% CEV"],
        },
    }


def run_condition(context: RunContext, condition: str) -> dict[str, Any]:
    """Audit, train, and probe one reward condition."""

    if context.seed is None:
        raise ValueError("two-factor PPO requires a resolved seed")
    if context.resume_from is not None:
        raise ValueError("continuation is not defined for this new experiment")
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")

    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    audit = demand_audit(seed=context.seed)
    outputs.write_json("demand_audit.json", audit)
    outputs.write_json(
        "resolved_recipe.json",
        _resolved_recipe(context, condition, audit),
    )
    result_grid = run_tune(
        build_config(context, condition),
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
        raise RuntimeError(f"{condition} expected one Tune trial, found {len(results)}")
    result = results[0]
    if result.error is not None:
        raise RuntimeError(f"{condition} PPO training failed") from result.error
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
                condition=condition,
                checkpoint_label=record["checkpoint_name"],
                agent_steps=record["agent_steps"],
                training_iteration=record["training_iteration"],
            )
        )
    plot_probe_trajectory(
        reports,
        condition=condition,
        path=context.results_dir / "probe_trajectory.png",
    )
    summary = {
        "condition": condition,
        "seed": context.seed,
        "smoke": context.smoke,
        "demand_audit": audit,
        "checkpoint_reports": [
            {
                "agent_steps": report["agent_steps"],
                "training_iteration": report["training_iteration"],
                "checkpoint": report["checkpoint"],
                "probe_fits": report["probe_fits"],
                "actor_cev95_dimension": report["cev"]["actor_activation"][
                    "cev95_dimension"
                ],
            }
            for report in reports
        ],
    }
    outputs.write_json("condition_summary.json", summary)
    return summary
