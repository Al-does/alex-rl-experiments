"""Shared recipe for targeted PPO value-clip / vf-loss-coeff grid runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from experiments.cassandra_belief_factoring_2026_08.environment import (
    CassandraActionObservationEnv,
)
from experiments.cassandra_belief_factoring_2026_08.shared import (
    build_config as build_shared_config,
    environment_config,
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
BASELINE_MODEL_CONFIG = TransformerModelConfig(
    d_model=64,
    n_layers=4,
    n_heads=1,
    context_len=256,
    max_seq_len=256,
).to_dict()


@dataclass(frozen=True, slots=True)
class VfClipGridCondition:
    """One value-function clipping and coefficient pair."""

    name: str
    vf_clip_param: float
    vf_loss_coeff: float
    hypothesis: str


def _require_comparison_seed(context: RunContext) -> None:
    if context.seed != EXPERIMENT_SEED:
        raise ValueError(
            "Cassandra vf-clip grid comparisons require seed 42; "
            f"received {context.seed!r}"
        )


def build_config(
    context: RunContext,
    *,
    condition: VfClipGridCondition,
) -> PPOConfig:
    """Build one grid cell from the dim-64 targeted baseline."""

    _require_comparison_seed(context)
    env_config = environment_config(action_scope="targeted")
    env_config["initial_state_distribution"] = "all_good"

    return (
        build_shared_config(context, action_scope="targeted")
        .environment(CassandraActionObservationEnv, env_config=env_config)
        .training(
            entropy_coeff=ENTROPY_COEFF,
            gamma=0.990,
            lambda_=BASELINE_LAMBDA,
            vf_clip_param=condition.vf_clip_param,
            vf_loss_coeff=condition.vf_loss_coeff,
            use_kl_loss=False,
            kl_coeff=0.0,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TransformerModel,
                model_config=dict(BASELINE_MODEL_CONFIG),
            )
        )
    )


def run_grid_condition(
    context: RunContext,
    *,
    condition: VfClipGridCondition,
) -> dict[str, Any]:
    """Train one grid cell and emit a compact run summary."""

    config = build_config(context, condition=condition)
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        {
            "condition": condition.name,
            "hypothesis": condition.hypothesis,
            "primary_comparison": (
                "value-clip / vf-loss-coeff grid versus the targeted dim-64 "
                "transformer PPO control (vf_clip=10, vf_loss_coeff=0.5)"
            ),
            "seed": EXPERIMENT_SEED,
            "algorithm": "PPO",
            "environment": environment_config(action_scope="targeted")
            | {"initial_state_distribution": "all_good"},
            "transformer": dict(config.rl_module_spec.model_config),
            "gamma": config.gamma,
            "lambda": config.lambda_,
            "vf_clip_param": config.vf_clip_param,
            "vf_loss_coeff": config.vf_loss_coeff,
            "entropy_coeff": config.entropy_coeff,
            "use_kl_loss": config.use_kl_loss,
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
        raise RuntimeError(
            f"{condition.name} expected one trial, got {len(results)}"
        )
    result = results[0]
    if result.error is not None:
        raise RuntimeError(f"{condition.name} training failed") from result.error
    write_training_curves(context)
    summary = {
        "condition": condition.name,
        "seed": EXPERIMENT_SEED,
        "smoke": context.smoke,
        "target_env_steps": target_steps,
        "status": "completed",
        "vf_clip_param": condition.vf_clip_param,
        "vf_loss_coeff": condition.vf_loss_coeff,
        "checkpoint": (
            str(result.checkpoint.path)
            if result.checkpoint is not None
            else None
        ),
    }
    outputs.write_json("grid_summary.json", summary)
    return summary


__all__ = [
    "BASELINE_LAMBDA",
    "BASELINE_MODEL_CONFIG",
    "ENTROPY_COEFF",
    "EXPERIMENT_SEED",
    "SMOKE_ENV_STEPS",
    "TOTAL_ENV_STEPS",
    "VfClipGridCondition",
    "build_config",
    "run_grid_condition",
]
