"""Discrete-SAC recipe for two cycle-2 reward conditions."""

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
from experiments.factored_representations_reproduction_PPO_2026_08.shared import (
    _save_initial_checkpoint,
    _save_log_spaced_checkpoint,
    checkpoint_records,
)
from experiments.storage.training_curves import write_training_curves
from experiments.two_factor_reward_state_SAC_cycle_2.analysis import (
    analyze_checkpoint,
    plot_probe_trajectory,
)
from experiments.two_factor_reward_state_SAC_cycle_2.design import (
    GAMMA,
    analytic_design_summary,
)
from experiments.two_factor_reward_state_SAC_cycle_2.model import (
    TwoFactorRewardSAC,
    TwoFactorRoPESACCatalog,
)
from experiments.two_factor_reward_state_SAC_cycle_2.process import (
    FACTOR_COUNT,
    JOINT_TOKEN_COUNT,
    LOCAL_CONTEXT_LENGTH,
    MESS3_ALPHA,
    SAC_HISTORY_LENGTH,
    TRANSFORMER_LAYERS,
    TRANSITION_MATRIX,
    environment_config,
)
from experiments.two_factor_reward_state_SAC_cycle_2.task import (
    ACTION_LABELS,
    ACTION_PAIRS,
    CONDITIONS,
    N_ACTIONS,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune
from learners.models.transformer import TransformerModelConfig


TOTAL_ENV_STEPS = 5_000_000
SMOKE_ENV_STEPS = 128
TRAIN_BATCH_SIZE = 8_192
LEARNER_MINIBATCH_COUNT = 8
LEARNER_MINIBATCH_SIZE = TRAIN_BATCH_SIZE // LEARNER_MINIBATCH_COUNT
SMOKE_BATCH_SIZE = 64
LEARNING_STARTS = 1_500
SMOKE_LEARNING_STARTS = 32
REPLAY_CAPACITY = 1_000_000
SMOKE_REPLAY_CAPACITY = 1_024
TARGET_ENTROPY_FRACTION = 0.6
TARGET_ENTROPY = TARGET_ENTROPY_FRACTION * math.log(N_ACTIONS)
MODEL_CONFIG = {
    **TransformerModelConfig(
        d_model=64,
        n_layers=TRANSFORMER_LAYERS,
        n_heads=1,
        context_len=LOCAL_CONTEXT_LENGTH,
    ).to_dict(),
    "head_fcnet_hiddens": [],
}


class EightMinibatchSAC(SAC):
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
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    profile = context.hardware or PROFILES["cpu"]
    return (
        SACConfig(algo_class=EightMinibatchSAC)
        .environment(HMMEnv, env_config=environment_config(condition))
        .framework("torch", torch_compile_learner=False, torch_compile_worker=False)
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
            training_intensity=1.0,
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
                catalog_class=TwoFactorRoPESACCatalog,
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
                checkpoint_root=str(context.artifacts_dir / "log_spaced_checkpoints"),
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
            rollout_fragment_length=1 if context.smoke else SAC_HISTORY_LENGTH,
            sample_timeout_s=600.0,
        )
        .learners(
            num_gpus_per_learner=(
                1 if profile.learner_device == "cuda" else 0
            )
        )
        .reporting(
            min_sample_timesteps_per_iteration=(
                64 if context.smoke else TRAIN_BATCH_SIZE
            ),
            min_time_s_per_iteration=0 if context.smoke else 1,
        )
    )


def _resolved_recipe(context: RunContext, condition: str) -> dict[str, Any]:
    return {
        "study": "two_factor_reward_state_SAC_cycle_2",
        "condition": condition,
        "hypothesis": (
            "selective reward makes the rewarded factor belief more linearly "
            "accessible under independent variant-3 control"
        ),
        "factor_count": FACTOR_COUNT,
        "factor_transition_matrix": TRANSITION_MATRIX.tolist(),
        "emission_alpha": MESS3_ALPHA,
        "joint_token_count": JOINT_TOKEN_COUNT,
        "action_pairs_by_flat_index": [list(pair) for pair in ACTION_PAIRS],
        "action_labels_by_flat_index": list(ACTION_LABELS),
        "environment": environment_config(condition),
        "analytic_design": analytic_design_summary(),
        "algorithm": "discrete SAC",
        "model": MODEL_CONFIG,
        "sac_fixed_history_frames": SAC_HISTORY_LENGTH,
        "total_env_steps": SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS,
        "checkpoint_schedule": "initial, powers of two iterations, final",
    }


def run_condition(context: RunContext, condition: str) -> dict[str, Any]:
    if context.seed is None:
        raise ValueError("two-factor SAC requires a resolved seed")
    if context.resume_from is not None:
        raise ValueError("continuation is not defined for this experiment")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json("resolved_recipe.json", _resolved_recipe(context, condition))
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
    if len(results) != 1 or results[0].error is not None:
        raise RuntimeError(f"{condition} SAC training failed")
    result = results[0]
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
        "checkpoint_reports": reports,
    }
    outputs.write_json("condition_summary.json", summary)
    return summary
