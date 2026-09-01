"""Cycle-2-matched two-factor study using REINFORCE only."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import torch
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo import (
    LEARNER_RESULTS_KL_KEY,
    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY,
    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
)
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import (
    ENTROPY_KEY,
    POLICY_LOSS_KEY,
    VF_LOSS_KEY,
)
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import explained_variance

from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_PPO_2026_08.shared import (
    _save_initial_checkpoint,
    _save_log_spaced_checkpoint,
    checkpoint_records,
)
from experiments.storage.training_curves import write_training_curves
from experiments.two_factor_reward_state_PPO_cycle_2.analysis import (
    analyze_checkpoint,
    plot_probe_trajectory,
)
from experiments.two_factor_reward_state_PPO_cycle_2.design import (
    GAMMA,
    analytic_design_summary,
)
from experiments.two_factor_reward_state_PPO_cycle_2.process import (
    FACTOR_COUNT,
    JOINT_TOKEN_COUNT,
    LOCAL_CONTEXT_LENGTH,
    MESS3_ALPHA,
    TRANSFORMER_LAYERS,
    TRANSFORMER_LOOKBACK,
    TRANSITION_MATRIX,
    environment_config,
)
from experiments.two_factor_reward_state_PPO_cycle_2.task import (
    ACTION_LABELS,
    ACTION_PAIRS,
    CONDITIONS,
)
from experiments.two_factor_reward_state_REINFORCE_cycle_3.model import (
    TwoFactorRewardReinforce,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune
from learners.models.transformer import TransformerModelConfig


TOTAL_ENV_STEPS = 5_000_000
SMOKE_ENV_STEPS = 4_096
TRAIN_BATCH_SIZE = 32_768
SMOKE_BATCH_SIZE = 2_048
MINIBATCH_SIZE = 2_048
SMOKE_MINIBATCH_SIZE = 256
LEARNING_RATE = 4.2e-4
SMOKE_LEARNING_RATE = 3e-4
ENTROPY_COEFF = 0.0
VALUE_LOSS_COEFF = 0.5
MODEL_CONFIG = TransformerModelConfig(
    d_model=64,
    n_layers=TRANSFORMER_LAYERS,
    n_heads=1,
    context_len=LOCAL_CONTEXT_LENGTH,
).to_dict()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    weights = mask.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


class ReinforceTorchLearner(PPOTorchLearner):
    """Monte-Carlo policy gradient with a learned state-value baseline.

    PPO supplies collection, return estimation, and recurrent batching. The
    objective is REINFORCE: it uses current-policy log probabilities directly,
    with no probability ratio, clipping, KL penalty, or repeated epochs.
    """

    @override(PPOTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id,
        config,
        batch,
        fwd_out,
    ):
        module = self.module[module_id].unwrapped()
        distribution = module.get_train_action_dist_cls().from_logits(
            fwd_out[Columns.ACTION_DIST_INPUTS]
        )
        log_prob = distribution.logp(batch[Columns.ACTIONS])
        entropy = distribution.entropy()
        mask = batch.get(Columns.LOSS_MASK)
        advantages = batch[Postprocessing.ADVANTAGES].detach()

        policy_loss = -_masked_mean(log_prob * advantages, mask)
        values = module.compute_values(
            batch,
            embeddings=fwd_out.get(Columns.EMBEDDINGS),
        )
        value_error = (values - batch[Postprocessing.VALUE_TARGETS]).square()
        value_loss = _masked_mean(value_error, mask)
        mean_entropy = _masked_mean(entropy, mask)
        total = (
            policy_loss
            + config.vf_loss_coeff * value_loss
            - (
                self.entropy_coeff_schedulers_per_module[
                    module_id
                ].get_current_value()
                * mean_entropy
            )
        )
        zero = total.new_zeros(())
        self.metrics.log_dict(
            {
                POLICY_LOSS_KEY: policy_loss,
                VF_LOSS_KEY: value_loss,
                LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY: value_loss,
                LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY: explained_variance(
                    batch[Postprocessing.VALUE_TARGETS],
                    values,
                ),
                ENTROPY_KEY: mean_entropy,
                LEARNER_RESULTS_KL_KEY: zero,
            },
            key=module_id,
            window=1,
        )
        return total


class ReinforceConfig(PPOConfig):
    """RLlib configuration for the experiment-local REINFORCE integration."""

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or Reinforce)

    @override(PPOConfig)
    def get_default_learner_class(self):
        if self.framework_str != "torch":
            raise ValueError("REINFORCE currently supports only framework='torch'")
        return ReinforceTorchLearner


class Reinforce(PPO):
    """REINFORCE training on RLlib's on-policy sampling infrastructure."""

    @classmethod
    @override(PPO)
    def get_default_config(cls) -> ReinforceConfig:
        return ReinforceConfig()


def build_config(context: RunContext, condition: str) -> ReinforceConfig:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    profile = context.hardware or PROFILES["cpu"]
    return (
        ReinforceConfig()
        .environment(HMMEnv, env_config=environment_config(condition))
        .framework("torch", torch_compile_learner=False, torch_compile_worker=False)
        .training(
            lr=SMOKE_LEARNING_RATE if context.smoke else LEARNING_RATE,
            gamma=GAMMA,
            lambda_=1.0,
            use_critic=True,
            use_gae=True,
            use_kl_loss=False,
            vf_loss_coeff=VALUE_LOSS_COEFF,
            entropy_coeff=ENTROPY_COEFF,
            train_batch_size_per_learner=(
                SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
            ),
            minibatch_size=(
                SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE
            ),
            num_epochs=1,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TwoFactorRewardReinforce,
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
            sample_timeout_s=600.0,
        )
        .learners(
            num_gpus_per_learner=(
                1 if profile.learner_device == "cuda" else 0
            )
        )
    )


def _resolved_recipe(context: RunContext, condition: str) -> dict[str, Any]:
    return {
        "study": "two_factor_reward_state_REINFORCE_cycle_3",
        "condition": condition,
        "hypothesis": (
            "selective reward makes the rewarded factor belief more linearly "
            "accessible under independent variant-3 control when optimized by "
            "Monte-Carlo REINFORCE"
        ),
        "primary_comparison": "reward_both versus reward_factor_1",
        "factor_count": FACTOR_COUNT,
        "factor_transition_matrix": TRANSITION_MATRIX.tolist(),
        "emission_alpha": MESS3_ALPHA,
        "joint_token_count": JOINT_TOKEN_COUNT,
        "action_pairs_by_flat_index": [list(pair) for pair in ACTION_PAIRS],
        "action_labels_by_flat_index": list(ACTION_LABELS),
        "environment": environment_config(condition),
        "analytic_design": analytic_design_summary(),
        "algorithm": "REINFORCE",
        "algorithm_variant": "Monte-Carlo policy gradient with learned value baseline",
        "gamma": GAMMA,
        "lambda": 1.0,
        "learning_rate": LEARNING_RATE,
        "entropy_coeff": ENTROPY_COEFF,
        "value_loss_coeff": VALUE_LOSS_COEFF,
        "train_batch_size_per_learner": TRAIN_BATCH_SIZE,
        "minibatch_size": MINIBATCH_SIZE,
        "num_epochs": 1,
        "model": MODEL_CONFIG,
        "transformer_raw_observation_lookback": TRANSFORMER_LOOKBACK,
        "total_env_steps": SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS,
        "checkpoint_schedule": "initial, powers of two iterations, final",
    }


def run_condition(context: RunContext, condition: str) -> dict[str, Any]:
    if context.seed is None:
        raise ValueError("two-factor REINFORCE requires a resolved seed")
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
        raise RuntimeError(f"{condition} REINFORCE training failed")
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
