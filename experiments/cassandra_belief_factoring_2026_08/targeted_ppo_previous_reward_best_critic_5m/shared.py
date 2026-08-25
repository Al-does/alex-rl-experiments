"""Shared recipe: previous reward visible + best vf-clip critic from Aug 2026 grid."""

from __future__ import annotations

from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from experiments.cassandra_belief_factoring_2026_08.shared import (
    build_config as build_shared_config,
    environment_config,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_interventions_5m.shared import (
    CassandraPreviousRewardObservationEnv,
)
from experiments.storage.training_curves import write_training_curves
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.runners import run_tune
from learners.models.transformer import TransformerModel, TransformerModelConfig


TOTAL_ENV_STEPS = 5_000_000
SMOKE_ENV_STEPS = 4_096
EXPERIMENT_SEED = 42
ENTROPY_COEFF = 0.03
BASELINE_LAMBDA = 0.95
BEST_VF_CLIP_PARAM = 100.0
BEST_VF_LOSS_COEFF = 0.01
MODEL_CONFIG = TransformerModelConfig(
    d_model=64,
    n_layers=4,
    n_heads=1,
    context_len=256,
    max_seq_len=256,
).to_dict()
HYPOTHESIS = (
    "Re-test visible previous reward with the vf100/coeff0.01 critic that "
    "achieved 495.30 on the standard targeted recipe; the earlier "
    "previous_reward run collapsed (~31.6) under the default vf_clip=10 critic."
)
PRIMARY_COMPARISON = (
    "previous_reward with best critic versus vf100_coeff001 without reward "
    "feature (495.30) and the earlier previous_reward baseline (31.60)"
)


def _require_comparison_seed(context: RunContext) -> None:
    if context.seed != EXPERIMENT_SEED:
        raise ValueError(
            "Cassandra previous-reward best-critic rerun requires seed 42; "
            f"received {context.seed!r}"
        )


def build_config(context: RunContext) -> PPOConfig:
    """Build targeted PPO with previous reward visible and best critic settings."""

    _require_comparison_seed(context)
    env_config = environment_config(action_scope="targeted")
    env_config["initial_state_distribution"] = "all_good"

    return (
        build_shared_config(context, action_scope="targeted")
        .environment(CassandraPreviousRewardObservationEnv, env_config=env_config)
        .training(
            entropy_coeff=ENTROPY_COEFF,
            gamma=0.990,
            lambda_=BASELINE_LAMBDA,
            vf_clip_param=BEST_VF_CLIP_PARAM,
            vf_loss_coeff=BEST_VF_LOSS_COEFF,
            use_kl_loss=False,
            kl_coeff=0.0,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TransformerModel,
                model_config=dict(MODEL_CONFIG),
            )
        )
    )


def run_condition(context: RunContext) -> dict[str, Any]:
    """Train the rerun and emit compact summaries."""

    config = build_config(context)
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        {
            "condition": "targeted_ppo_previous_reward_best_critic",
            "hypothesis": HYPOTHESIS,
            "primary_comparison": PRIMARY_COMPARISON,
            "seed": EXPERIMENT_SEED,
            "algorithm": "PPO",
            "environment": environment_config(action_scope="targeted")
            | {"initial_state_distribution": "all_good"},
            "transformer": dict(MODEL_CONFIG),
            "gamma": config.gamma,
            "lambda": config.lambda_,
            "vf_clip_param": config.vf_clip_param,
            "vf_loss_coeff": config.vf_loss_coeff,
            "entropy_coeff": config.entropy_coeff,
            "use_kl_loss": config.use_kl_loss,
            "previous_reward_visible": True,
            "best_critic_reference": "best_run_parameters.json vf100_coeff001",
            "total_env_steps": target_steps,
        },
    )
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
        raise RuntimeError(f"expected one trial, got {len(results)}")
    result = results[0]
    if result.error is not None:
        raise RuntimeError("previous_reward best-critic training failed") from result.error
    write_training_curves(context)
    summary = {
        "condition": "targeted_ppo_previous_reward_best_critic",
        "seed": EXPERIMENT_SEED,
        "smoke": context.smoke,
        "target_env_steps": target_steps,
        "status": "completed",
        "vf_clip_param": BEST_VF_CLIP_PARAM,
        "vf_loss_coeff": BEST_VF_LOSS_COEFF,
        "previous_reward_visible": True,
        "checkpoint": (
            str(result.checkpoint.path)
            if result.checkpoint is not None
            else None
        ),
    }
    outputs.write_json("condition_summary.json", summary)
    return summary


__all__ = [
    "BEST_VF_CLIP_PARAM",
    "BEST_VF_LOSS_COEFF",
    "EXPERIMENT_SEED",
    "HYPOTHESIS",
    "SMOKE_ENV_STEPS",
    "TOTAL_ENV_STEPS",
    "build_config",
    "run_condition",
]
