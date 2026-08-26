"""Shared recipe: best vf-clip critic + BPTT 64 for 250M-step Cassandra PPO runs."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from functools import partial
from numbers import Real
from pathlib import Path
from typing import Any, Literal

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
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_interventions_5m.shared import (
    CassandraPreviousRewardObservationEnv,
)
from experiments.storage.training_curves import write_training_curves
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.runners import run_tune
from learners.models.transformer import TransformerModel, TransformerModelConfig


ActionScope = Literal["global_aliases", "targeted"]

TOTAL_ENV_STEPS = 250_000_000
SMOKE_ENV_STEPS = 4_096
CHECKPOINT_STEP_INTERVAL = 50_000_000
ENTROPY_COEFF = 0.03
BASELINE_LAMBDA = 0.95
BEST_VF_CLIP_PARAM = 100.0
BEST_VF_LOSS_COEFF = 0.01
MODEL_CONFIG = TransformerModelConfig(
    d_model=64,
    n_layers=4,
    n_heads=1,
    context_len=64,
    max_seq_len=64,
).to_dict()
HYPOTHESIS = (
    "Combine the vf100/coeff0.01 critic with BPTT 64 on global-alias and "
    "targeted Cassandra maintenance for long-horizon (250M-step) learning."
)
PRIMARY_COMPARISON = (
    "global_aliases versus targeted actions under matched best-critic BPTT-64 "
    "recipe across two seeds"
)
PREVIOUS_REWARD_HYPOTHESIS = (
    "Extend the vf100/coeff0.01 BPTT-64 targeted recipe to 250M steps with "
    "visible previous reward in the observation, matching the 5M finding that "
    "best-critic scaling makes previous reward neutral to slightly helpful."
)
PREVIOUS_REWARD_PRIMARY_COMPARISON = (
    "targeted previous-reward BPTT-64 at 250M versus targeted BPTT-64 without "
    "previous reward across seeds 42 and 43"
)


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


def _save_milestone_checkpoint(
    *,
    algorithm: Any,
    result: Mapping[str, Any],
    checkpoint_root: str,
    **_: Any,
) -> None:
    """Save public Algorithm checkpoints at 50M-step milestones."""

    steps_value = _metric(
        result,
        "env_runners/num_env_steps_sampled_lifetime",
    )
    iteration_value = _metric(result, "training_iteration")
    if steps_value is None or iteration_value is None:
        return
    steps = int(steps_value)
    milestone = steps // CHECKPOINT_STEP_INTERVAL
    if milestone <= 0:
        return
    target_steps = milestone * CHECKPOINT_STEP_INTERVAL
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    records = (
        json.loads(index_path.read_text()).get("checkpoints", [])
        if index_path.is_file()
        else []
    )
    if any(int(record["agent_steps"]) == target_steps for record in records):
        return
    destination = root / f"steps_{target_steps:012d}"
    saved = Path(algorithm.save_to_path(str(destination)))
    records.append(
        {
            "path": str(saved),
            "checkpoint_name": saved.name,
            "training_iteration": int(iteration_value),
            "agent_steps": target_steps,
        }
    )
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"checkpoints": records}, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(index_path)


def build_config(
    context: RunContext,
    *,
    action_scope: ActionScope,
    previous_reward_visible: bool = False,
) -> PPOConfig:
    """Build one 250M-step recipe leaf for global-alias or targeted actions."""

    if context.seed is None:
        raise ValueError("best_critic_bptt64_250m requires a resolved seed")
    if previous_reward_visible and action_scope != "targeted":
        raise ValueError(
            "previous_reward_visible is only supported for targeted action_scope"
        )
    env_config = environment_config(action_scope=action_scope)
    env_config["initial_state_distribution"] = "all_good"
    env_class = (
        CassandraPreviousRewardObservationEnv
        if previous_reward_visible
        else CassandraActionObservationEnv
    )

    return (
        build_shared_config(context, action_scope=action_scope)
        .environment(env_class, env_config=env_config)
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
        .callbacks(
            on_train_result=partial(
                _save_milestone_checkpoint,
                checkpoint_root=str(
                    context.artifacts_dir / "milestone_checkpoints"
                ),
            )
        )
    )


def run_recipe(
    context: RunContext,
    *,
    action_scope: ActionScope,
    condition: str,
    previous_reward_visible: bool = False,
    config_builder: Callable[[RunContext], PPOConfig] | None = None,
    recipe_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one long-horizon condition and emit compact summaries."""

    config = (
        config_builder(context)
        if config_builder is not None
        else build_config(
            context,
            action_scope=action_scope,
            previous_reward_visible=previous_reward_visible,
        )
    )
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    recipe = {
        "condition": condition,
        "hypothesis": (
            PREVIOUS_REWARD_HYPOTHESIS
            if previous_reward_visible
            else HYPOTHESIS
        ),
        "primary_comparison": (
            PREVIOUS_REWARD_PRIMARY_COMPARISON
            if previous_reward_visible
            else PRIMARY_COMPARISON
        ),
        "seed": context.seed,
        "algorithm": "PPO",
        "environment": environment_config(action_scope=action_scope)
        | {"initial_state_distribution": "all_good"},
        "transformer": dict(config.rl_module_spec.model_config),
        "gamma": config.gamma,
        "lambda": config.lambda_,
        "vf_clip_param": config.vf_clip_param,
        "vf_loss_coeff": config.vf_loss_coeff,
        "entropy_coeff": config.entropy_coeff,
        "use_kl_loss": config.use_kl_loss,
        "total_env_steps": target_steps,
        "checkpoint_schedule": f"every_{CHECKPOINT_STEP_INTERVAL}_env_steps",
    }
    if previous_reward_visible:
        recipe["previous_reward_visible"] = True
    if recipe_metadata is not None:
        recipe.update(recipe_metadata)
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
        "action_scope": action_scope,
        "seed": context.seed,
        "smoke": context.smoke,
        "target_env_steps": target_steps,
        "status": "completed",
        "vf_clip_param": BEST_VF_CLIP_PARAM,
        "vf_loss_coeff": BEST_VF_LOSS_COEFF,
        "previous_reward_visible": previous_reward_visible,
        "checkpoint": (
            str(result.checkpoint.path)
            if result.checkpoint is not None
            else None
        ),
    }
    if recipe_metadata is not None:
        summary.update(recipe_metadata)
    outputs.write_json("condition_summary.json", summary)
    return summary


__all__ = [
    "ActionScope",
    "BEST_VF_CLIP_PARAM",
    "BEST_VF_LOSS_COEFF",
    "CHECKPOINT_STEP_INTERVAL",
    "HYPOTHESIS",
    "MODEL_CONFIG",
    "PREVIOUS_REWARD_HYPOTHESIS",
    "PREVIOUS_REWARD_PRIMARY_COMPARISON",
    "PRIMARY_COMPARISON",
    "SMOKE_ENV_STEPS",
    "TOTAL_ENV_STEPS",
    "build_config",
    "run_recipe",
]
