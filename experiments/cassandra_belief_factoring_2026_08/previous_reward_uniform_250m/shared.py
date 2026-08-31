"""Shared recipe: previous reward visible + uniform starts + 250M milestone checkpoints."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from functools import partial
from numbers import Real
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import gymnasium as gym
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from experiments.cassandra_belief_factoring_2026_08.shared import (
    build_config as build_shared_config,
    environment_config,
)
from experiments.cassandra_belief_factoring_2026_08.environment import (
    CassandraFullyObservablePreviousRewardEnv,
)
from experiments.cassandra_belief_factoring_2026_08.targeted_ppo_small_interventions_5m.shared import (
    CassandraPreviousRewardObservationEnv,
)
from experiments.storage.training_curves import write_training_curves
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.runners import run_algorithm, run_tune
from learners.models.transformer import TransformerModel, TransformerModelConfig


ModelWidth = Literal[64, 96]
ActionScope = Literal["targeted", "global_aliases"]
ObservationVariant = Literal["symbol", "state"]

TOTAL_ENV_STEPS = 250_000_000
SMOKE_ENV_STEPS = 4_096
CHECKPOINT_STEP_INTERVAL = 50_000_000
EXPERIMENT_SEED = 42
ENTROPY_COEFF = 0.03
BASELINE_LAMBDA = 0.95
BEST_VF_CLIP_PARAM = 100.0
BEST_VF_LOSS_COEFF = 0.01
CONTEXT_LEN = 256
LONG_RUN_ENV_RUNNERS = 8
LONG_RUN_ENVS_PER_RUNNER = 2
HYPOTHESIS = (
    "Scale the best previous-reward targeted PPO recipe to 250M steps with "
    "uniform initial machine states and milestone checkpoints every 50M steps."
)
PRIMARY_COMPARISON = (
    "dim-64 versus dim-96 transformers (both 3 layers, 4 heads) under "
    "previous-reward visibility, best critic, and uniform episode starts"
)


def _observation_env_class(
    observation_variant: ObservationVariant,
) -> type[gym.Env]:
    if observation_variant == "state":
        return CassandraFullyObservablePreviousRewardEnv
    return CassandraPreviousRewardObservationEnv


def _policy_observation_description(
    *,
    action_scope: ActionScope,
    observation_variant: ObservationVariant,
) -> str:
    if observation_variant == "state":
        return (
            "256-way joint-state one-hot plus preceding scalar reward; "
            "fully observable diagnostic"
        )
    action_count = 10
    return (
        f"16-way symbol one-hot plus previous {action_count}-way "
        "action one-hot plus preceding scalar reward"
    )


def model_config(*, d_model: ModelWidth) -> dict[str, Any]:
    """Return the transformer config for one width variant."""

    return TransformerModelConfig(
        d_model=d_model,
        n_layers=3,
        n_heads=4,
        context_len=CONTEXT_LEN,
        max_seq_len=CONTEXT_LEN,
    ).to_dict()


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


def _reached_env_step_target(
    target_steps: int,
) -> Callable[[Mapping[str, Any]], bool]:
    def _should_stop(metrics: Mapping[str, Any]) -> bool:
        steps = _metric(metrics, "env_runners/num_env_steps_sampled_lifetime")
        return steps is not None and steps >= target_steps

    return _should_stop


def _latest_algorithm_checkpoint(context: RunContext) -> Path:
    root = context.artifacts_dir / "checkpoints"
    candidates = sorted(root.glob("iteration_*"))
    if not candidates:
        raise RuntimeError(f"no algorithm checkpoints under {root}")
    return candidates[-1]


def _require_comparison_seed(context: RunContext) -> None:
    if context.seed != EXPERIMENT_SEED:
        raise ValueError(
            "Cassandra previous-reward uniform 250M requires seed 42; "
            f"received {context.seed!r}"
        )


def build_config(
    context: RunContext,
    *,
    d_model: ModelWidth,
    action_scope: ActionScope = "targeted",
    observation_variant: ObservationVariant = "symbol",
) -> PPOConfig:
    """Build PPO with previous reward, uniform starts, and milestones."""

    _require_comparison_seed(context)
    env_config = environment_config(action_scope=action_scope)
    env_config["initial_state_distribution"] = "uniform"
    env_class = _observation_env_class(observation_variant)

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
                model_config=model_config(d_model=d_model),
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
        .env_runners(
            num_env_runners=(
                0 if context.smoke else LONG_RUN_ENV_RUNNERS
            ),
            num_envs_per_env_runner=(
                1 if context.smoke else LONG_RUN_ENVS_PER_RUNNER
            ),
        )
    )


def run_recipe(
    context: RunContext,
    *,
    d_model: ModelWidth,
    condition: str,
    action_scope: ActionScope = "targeted",
    observation_variant: ObservationVariant = "symbol",
) -> dict[str, Any]:
    """Train one long-horizon width variant and emit compact summaries."""

    config = build_config(
        context,
        d_model=d_model,
        action_scope=action_scope,
        observation_variant=observation_variant,
    )
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    transformer = model_config(d_model=d_model)
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        {
            "condition": condition,
            "hypothesis": HYPOTHESIS,
            "primary_comparison": PRIMARY_COMPARISON,
            "seed": EXPERIMENT_SEED,
            "algorithm": "PPO",
            "environment": environment_config(action_scope=action_scope)
            | {"initial_state_distribution": "uniform"},
            "observation_variant": observation_variant,
            "policy_observation": _policy_observation_description(
                action_scope=action_scope,
                observation_variant=observation_variant,
            ),
            "transformer": transformer,
            "gamma": config.gamma,
            "lambda": config.lambda_,
            "vf_clip_param": config.vf_clip_param,
            "vf_loss_coeff": config.vf_loss_coeff,
            "entropy_coeff": config.entropy_coeff,
            "use_kl_loss": config.use_kl_loss,
            "previous_reward_visible": True,
            "best_critic_reference": (
                "targeted_ppo_previous_reward_best_critic_5m vf100/coeff0.01"
            ),
            "total_env_steps": target_steps,
            "checkpoint_schedule": (
                f"every_{CHECKPOINT_STEP_INTERVAL}_env_steps_after_first_milestone"
            ),
            "resume_from": (
                str(context.resume_from)
                if context.resume_from is not None
                else None
            ),
        },
    )
    if context.resume_from is not None:
        final_metrics = run_algorithm(
            config,
            context,
            should_stop=_reached_env_step_target(target_steps),
            checkpoint_at_end=True,
        )
        final_checkpoint = _latest_algorithm_checkpoint(context)
        result = SimpleNamespace(
            checkpoint=SimpleNamespace(path=str(final_checkpoint)),
            metrics=final_metrics,
            error=None,
        )
    else:
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
        "d_model": d_model,
        "seed": EXPERIMENT_SEED,
        "smoke": context.smoke,
        "target_env_steps": target_steps,
        "status": "completed",
        "vf_clip_param": BEST_VF_CLIP_PARAM,
        "vf_loss_coeff": BEST_VF_LOSS_COEFF,
        "previous_reward_visible": True,
        "observation_variant": observation_variant,
        "initial_state_distribution": "uniform",
        "action_scope": action_scope,
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
    "CHECKPOINT_STEP_INTERVAL",
    "CONTEXT_LEN",
    "EXPERIMENT_SEED",
    "HYPOTHESIS",
    "ObservationVariant",
    "ActionScope",
    "ModelWidth",
    "PRIMARY_COMPARISON",
    "SMOKE_ENV_STEPS",
    "TOTAL_ENV_STEPS",
    "build_config",
    "model_config",
    "run_recipe",
]
