"""Shared helpers for targeted continuations from completed 250M runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.checkpoint_recovery import (
    recover_tune_artifacts,
    source_run_id,
)
from experiments.cassandra_belief_factoring_2026_08.best_critic_bptt64_250m.shared import (
    BEST_VF_CLIP_PARAM,
    BEST_VF_LOSS_COEFF,
    CHECKPOINT_STEP_INTERVAL,
    ENTROPY_COEFF,
    MODEL_CONFIG,
    SMOKE_ENV_STEPS,
    build_config as build_base_config,
)
from experiments.cassandra_belief_factoring_2026_08.shared import environment_config
from experiments.storage.training_curves import write_training_curves
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.runners import run_tune

PRIOR_LIFETIME_ENV_STEPS = 250_000_000
CONTINUE_LIFETIME_ENV_STEPS = 500_000_000
ANNEAL_LIFETIME_ENV_STEPS = 255_000_000
ANNEAL_DURATION_ENV_STEPS = 5_000_000
ANNEAL_FINAL_ENTROPY = 0.008
ENTROPY_ANNEAL_SCHEDULE = [
    [0, ENTROPY_COEFF],
    [ANNEAL_DURATION_ENV_STEPS, ANNEAL_FINAL_ENTROPY],
]
ACTION_SCOPE = "targeted"


def _resolve_resume_context(context: RunContext) -> RunContext:
    if context.smoke or context.resume_from is not None:
        return context
    if context.seed is None:
        raise ValueError("continuation runs require a resolved seed")
    run_id = source_run_id(context.seed)
    recover_tune_artifacts(
        destination=context.artifacts_dir,
        source_run_id=run_id,
    )
    return replace(context, resume_from=context.artifacts_dir / "tune")


def build_continue_config(context: RunContext) -> PPOConfig:
    """Best-critic targeted recipe resumed with fixed entropy."""

    return build_base_config(context, action_scope=ACTION_SCOPE)


def build_anneal_config(context: RunContext) -> PPOConfig:
    """Best-critic targeted recipe resumed with a 5M-step entropy anneal."""

    config = build_base_config(context, action_scope=ACTION_SCOPE)
    return config.training(entropy_coeff=ENTROPY_ANNEAL_SCHEDULE)


def run_continuation(
    context: RunContext,
    *,
    condition: str,
    hypothesis: str,
    lifetime_env_steps: int,
    config_builder,
    extra_recipe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resume one targeted agent and train to a lifetime step budget."""

    context = _resolve_resume_context(context)
    config = config_builder(context)
    target_steps = SMOKE_ENV_STEPS if context.smoke else lifetime_env_steps
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    recipe = {
        "condition": condition,
        "hypothesis": hypothesis,
        "seed": context.seed,
        "algorithm": "PPO",
        "action_scope": ACTION_SCOPE,
        "environment": environment_config(action_scope=ACTION_SCOPE)
        | {"initial_state_distribution": "all_good"},
        "transformer": dict(config.rl_module_spec.model_config),
        "gamma": config.gamma,
        "lambda": config.lambda_,
        "vf_clip_param": config.vf_clip_param,
        "vf_loss_coeff": config.vf_loss_coeff,
        "entropy_coeff": config.entropy_coeff,
        "use_kl_loss": config.use_kl_loss,
        "prior_lifetime_env_steps": None if context.smoke else PRIOR_LIFETIME_ENV_STEPS,
        "target_lifetime_env_steps": target_steps,
        "checkpoint_schedule": f"every_{CHECKPOINT_STEP_INTERVAL}_env_steps",
        "resume_from": str(context.resume_from) if context.resume_from else None,
        "source_run_id": None if context.smoke else source_run_id(context.seed),
    }
    if extra_recipe:
        recipe.update(dict(extra_recipe))
    outputs.write_json("resolved_recipe.json", recipe)
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
        raise RuntimeError(f"{condition} expected one trial, got {len(results)}")
    result = results[0]
    if result.error is not None:
        raise RuntimeError(f"{condition} training failed") from result.error
    write_training_curves(context)
    summary = {
        "condition": condition,
        "action_scope": ACTION_SCOPE,
        "seed": context.seed,
        "smoke": context.smoke,
        "target_lifetime_env_steps": target_steps,
        "status": "completed",
        "vf_clip_param": BEST_VF_CLIP_PARAM,
        "vf_loss_coeff": BEST_VF_LOSS_COEFF,
        "source_run_id": None if context.smoke else source_run_id(context.seed),
        "checkpoint": (
            str(result.checkpoint.path)
            if result.checkpoint is not None
            else None
        ),
    }
    outputs.write_json("condition_summary.json", summary)
    return summary


__all__ = [
    "ANNEAL_DURATION_ENV_STEPS",
    "ANNEAL_FINAL_ENTROPY",
    "ANNEAL_LIFETIME_ENV_STEPS",
    "CONTINUE_LIFETIME_ENV_STEPS",
    "ENTROPY_ANNEAL_SCHEDULE",
    "PRIOR_LIFETIME_ENV_STEPS",
    "build_anneal_config",
    "build_continue_config",
    "run_continuation",
]
