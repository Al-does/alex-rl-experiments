"""Shared discrete-SAC recipe for the three two-factor reward arms."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
import math
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.sac import SAC, SACConfig
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
from experiments.storage.training_curves import write_training_curves
from experiments.two_factor_reward_state_SAC_cycle_1.analysis import (
    analyze_checkpoint,
    plot_probe_trajectory,
)
from experiments.two_factor_reward_state_SAC_cycle_1.design import (
    GAMMA,
    demand_audit,
)
from experiments.two_factor_reward_state_SAC_cycle_1.model import (
    TwoFactorRewardSAC,
    TwoFactorSACCatalog,
)
from experiments.two_factor_reward_state_SAC_cycle_1.process import (
    CONTEXT_LENGTH,
    FACTOR_COUNT,
    JOINT_TOKEN_COUNT,
    MESS3_ALPHA,
    TRANSITION_MATRIX,
    environment_config,
)
from experiments.two_factor_reward_state_SAC_cycle_1.task import (
    ACTION_LABELS,
    ACTION_PAIRS,
    CONDITIONS,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune


TOTAL_ENV_STEPS = 20_000_000
SMOKE_ENV_STEPS = 128
TRAIN_BATCH_SIZE = 8_192
LEARNER_MINIBATCH_COUNT = 8
LEARNER_MINIBATCH_SIZE = TRAIN_BATCH_SIZE // LEARNER_MINIBATCH_COUNT
SMOKE_BATCH_SIZE = 64
LEARNING_STARTS = 1_500
SMOKE_LEARNING_STARTS = 32
REPLAY_CAPACITY = 1_000_000
SMOKE_REPLAY_CAPACITY = 1_024
TRAINING_INTENSITY = 1.0
TARGET_ENTROPY_FRACTION = 0.6
TARGET_ENTROPY = TARGET_ENTROPY_FRACTION * math.log(JOINT_TOKEN_COUNT)
MODEL_CONFIG = {
    **FactoredReproductionModelConfig(
        d_model=64,
        n_layers=4,
        n_heads=4,
        d_mlp=256,
        context_length=CONTEXT_LENGTH,
        max_seq_len=CONTEXT_LENGTH,
    ).to_dict(),
    "head_fcnet_hiddens": [],
}


class EightMinibatchSAC(SAC):
    """Process each replay batch as eight GPU-sized learner minibatches."""

    def training_step(self) -> dict[str, Any]:
        original_update = self.learner_group.update
        minibatch_size = min(
            LEARNER_MINIBATCH_SIZE,
            self.config.total_train_batch_size,
        )

        def update_with_minibatches(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("num_epochs", 1)
            kwargs.setdefault("minibatch_size", minibatch_size)
            return original_update(*args, **kwargs)

        self.learner_group.update = update_with_minibatches
        try:
            return super().training_step()
        finally:
            self.learner_group.update = original_update


def build_config(context: RunContext, condition: str) -> SACConfig:
    """Build one fresh nine-action discrete-SAC configuration."""

    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    profile = context.hardware or PROFILES["cpu"]
    return (
        SACConfig(algo_class=EightMinibatchSAC)
        .environment(HMMEnv, env_config=environment_config(condition))
        .framework(
            "torch",
            torch_compile_learner=False,
            torch_compile_worker=False,
        )
        .training(
            gamma=GAMMA,
            n_step=1,
            twin_q=True,
            tau=0.005,
            actor_lr=3e-5,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            target_entropy=TARGET_ENTROPY,
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
                "alpha": 0.6,
                "beta": 0.4,
            },
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TwoFactorRewardSAC,
                catalog_class=TwoFactorSACCatalog,
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
            rollout_fragment_length=(1 if context.smoke else CONTEXT_LENGTH),
            sample_timeout_s=600.0,
        )
        .learners(
            num_gpus_per_learner=(
                1 if profile.learner_device == "cuda" else 0
            ),
        )
        .reporting(
            min_sample_timesteps_per_iteration=(
                64 if context.smoke else TRAIN_BATCH_SIZE
            ),
            min_time_s_per_iteration=0 if context.smoke else 1,
        )
    )


def _resolved_recipe(
    context: RunContext,
    condition: str,
    audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "study": "two_factor_reward_state_SAC_cycle_1",
        "condition": condition,
        "hypothesis": (
            "rewarding one versus both controlled factors changes which exact "
            "factor beliefs become linearly accessible in the SAC actor"
        ),
        "factor_count": FACTOR_COUNT,
        "factor_transition_matrix": TRANSITION_MATRIX.tolist(),
        "emission_alpha": MESS3_ALPHA,
        "joint_token_count": JOINT_TOKEN_COUNT,
        "action_pairs_by_flat_index": [list(pair) for pair in ACTION_PAIRS],
        "action_labels_by_flat_index": list(ACTION_LABELS),
        "reward_state": 2,
        "environment": environment_config(condition),
        "algorithm": "discrete SAC",
        "gamma": GAMMA,
        "n_step": 1,
        "twin_q": True,
        "tau": 0.005,
        "actor_learning_rate": 3e-5,
        "critic_learning_rate": 3e-4,
        "alpha_learning_rate": 3e-4,
        "train_batch_size_per_learner": TRAIN_BATCH_SIZE,
        "learner_minibatch_count": LEARNER_MINIBATCH_COUNT,
        "learner_minibatch_size": LEARNER_MINIBATCH_SIZE,
        "learner_num_epochs": 1,
        "training_intensity": TRAINING_INTENSITY,
        "target_entropy_fraction_of_categorical_maximum": TARGET_ENTROPY_FRACTION,
        "target_entropy": TARGET_ENTROPY,
        "learning_starts": LEARNING_STARTS,
        "rollout_fragment_length": CONTEXT_LENGTH,
        "torch_compile_learner": False,
        "min_sample_timesteps_per_iteration": TRAIN_BATCH_SIZE,
        "replay_capacity": REPLAY_CAPACITY,
        "model": MODEL_CONFIG,
        "total_env_steps": (
            SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
        ),
        "checkpoint_schedule": "initial, powers of two iterations, final",
        "pretraining_demand_audit": audit,
        "analysis": {
            "representation": (
                "actor final transformer block residual before final LayerNorm"
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
        raise ValueError("two-factor SAC requires a resolved seed")
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
        raise RuntimeError(f"{condition} SAC training failed") from result.error
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
