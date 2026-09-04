"""Baseline-free Monte Carlo REINFORCE for the two-factor cycle-3 task."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial
from numbers import Real
from pathlib import Path
from types import SimpleNamespace
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
from experiments.mess3_belief_geometry_2026_07.shared import (
    apply_runtime_resources,
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
    TRANSITION_MATRIX,
    environment_config,
)
from experiments.two_factor_reward_state_PPO_cycle_2.task import (
    ACTION_LABELS,
    ACTION_PAIRS,
    CONDITIONS,
)
from experiments.two_factor_reward_state_REINFORCE_cycle_4.model import (
    TwoFactorRewardReinforceCycle4,
)
from harness.artifacts import RunArtifacts, record_result
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.hardware import configure_hardware, shutdown_ray_if_owned
from harness.runners import (
    _build_or_restore_algorithm,
    run_tune,
    save_algorithm_checkpoint,
)
from learners.models.transformer import TransformerModelConfig


TOTAL_ENV_STEPS = 8_000_000
CONTINUATION_TOTAL_ENV_STEPS = 24_000_000
STEP_CHECKPOINT_INTERVAL = 2_000_000
CONTINUATION_SPEC_FILENAME = "continuation_spec.json"
BUDGET_SPEC_FILENAME = "budget_spec.json"
SMOKE_ENV_STEPS = 4_096
TRAIN_BATCH_SIZE = 32_768
SMOKE_BATCH_SIZE = 2_048
LEARNING_RATE = 4.2e-4
ENTROPY_COEFF = 0.0
VALUE_LOSS_COEFF = 0.0
MODEL_CONFIG = TransformerModelConfig(
    d_model=64,
    n_layers=TRANSFORMER_LAYERS,
    n_heads=1,
    context_len=LOCAL_CONTEXT_LENGTH,
).to_dict()
CONTEXT32_L3_MODEL_CONFIG = {
    **MODEL_CONFIG,
    "context_len": 32,
    "n_layers": 3,
}
_active_algorithm: list[Any | None] = [None]


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


def _save_step_interval_checkpoint(
    *,
    algorithm: Any,
    result: Mapping[str, Any],
    checkpoint_root: str,
    step_interval: int = STEP_CHECKPOINT_INTERVAL,
    **_: Any,
) -> None:
    """Save Algorithm checkpoints each time lifetime env steps cross a boundary."""

    steps_value = _metric(result, "env_runners/num_env_steps_sampled_lifetime")
    iteration_value = _metric(result, "training_iteration")
    if steps_value is None or iteration_value is None:
        return
    steps = int(steps_value)
    boundary = (steps // step_interval) * step_interval
    if boundary <= 0:
        return
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    records = (
        json.loads(index_path.read_text()).get("checkpoints", [])
        if index_path.is_file()
        else []
    )
    if any(int(record["agent_steps"]) == boundary for record in records):
        return
    destination = root / f"steps_{boundary:09d}"
    saved = Path(algorithm.save_to_path(str(destination)))
    records.append(
        {
            "path": str(saved),
            "checkpoint_name": saved.name,
            "training_iteration": int(iteration_value),
            "agent_steps": boundary,
        }
    )
    records.sort(key=lambda row: int(row["agent_steps"]))
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"checkpoints": records}, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(index_path)


def _step_checkpoint_records(checkpoint_root: Path) -> list[dict[str, Any]]:
    index_path = checkpoint_root / "index.json"
    if not index_path.is_file():
        return []
    records = []
    for record in json.loads(index_path.read_text()).get("checkpoints", []):
        path = Path(record["path"])
        records.append(
            {
                "checkpoint_path": path,
                "checkpoint_name": str(record.get("checkpoint_name", path.name)),
                "training_iteration": int(record["training_iteration"]),
                "agent_steps": int(record["agent_steps"]),
            }
        )
    return sorted(records, key=lambda row: row["agent_steps"])


def _combine_on_train_result(
    *handlers: Callable[..., None],
) -> Callable[..., None]:
    def _combined(
        *,
        algorithm: Any,
        result: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        for handler in handlers:
            handler(algorithm=algorithm, result=result, **kwargs)

    return _combined


def _reached_env_step_target(
    target_steps: int,
) -> Callable[[Mapping[str, Any]], bool]:
    def _should_stop(metrics: Mapping[str, Any]) -> bool:
        steps = _metric(metrics, "env_runners/num_env_steps_sampled_lifetime")
        return steps is not None and steps >= target_steps

    return _should_stop


def _latest_algorithm_checkpoint(context: RunContext) -> Path:
    step_root = context.artifacts_dir / "step_checkpoints"
    step_records = _step_checkpoint_records(step_root)
    if step_records:
        return Path(step_records[-1]["checkpoint_path"])
    root = RunArtifacts.from_context(context).checkpoints_dir
    candidates = sorted(root.glob("iteration_*"))
    if not candidates:
        raise RuntimeError(
            f"no algorithm checkpoints under {step_root} or {root}"
        )
    return candidates[-1]


def _load_continuation_spec(context: RunContext) -> dict[str, Any] | None:
    path = context.artifacts_dir / CONTINUATION_SPEC_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_budget_spec(context: RunContext) -> dict[str, Any] | None:
    path = context.artifacts_dir / BUDGET_SPEC_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_budget_spec(context: RunContext, target_agent_steps: int) -> None:
    context.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (context.artifacts_dir / BUDGET_SPEC_FILENAME).write_text(
        json.dumps({"target_agent_steps": int(target_agent_steps)}, indent=2)
        + "\n"
    )


def _resolve_step_target(context: RunContext) -> int:
    if context.smoke:
        return SMOKE_ENV_STEPS
    spec = _load_continuation_spec(context)
    if spec is not None:
        return int(spec["target_agent_steps"])
    budget = _load_budget_spec(context)
    if budget is not None:
        return int(budget["target_agent_steps"])
    return TOTAL_ENV_STEPS


def _single_gpu_context(context: RunContext) -> RunContext:
    profile = context.hardware
    if (
        not context.smoke
        and profile is not None
        and profile.name == "cuda4090_gpuinfer"
    ):
        return replace(context, hardware=PROFILES["cuda4090"])
    return context


def _capture_algorithm_on_init(
    *,
    algorithm: Any,
    checkpoint_path: str,
    **_: Any,
) -> None:
    _active_algorithm[0] = algorithm
    _save_initial_checkpoint(algorithm=algorithm, checkpoint_path=checkpoint_path)


def _continuation_result_recorder(context: RunContext) -> Callable[..., None]:
    log_root = str(context.artifacts_dir / "log_spaced_checkpoints")
    step_root = str(context.artifacts_dir / "step_checkpoints")

    def _record(_context: RunContext, result: Mapping[str, Any]) -> None:
        record_result(_context, result)
        algorithm = _active_algorithm[0]
        if algorithm is None:
            return
        _save_log_spaced_checkpoint(
            algorithm=algorithm,
            result=result,
            checkpoint_root=log_root,
        )
        _save_step_interval_checkpoint(
            algorithm=algorithm,
            result=result,
            checkpoint_root=step_root,
        )

    return _record


def _run_continuation_algorithm(
    config: PPOConfig,
    context: RunContext,
    *,
    target_steps: int,
) -> Mapping[str, Any]:
    """Resume training with explicit algorithm capture for checkpoint hooks."""

    started_ray = (
        configure_hardware(context.hardware)
        if context.hardware is not None
        else False
    )
    algorithm = None
    iteration = 0
    recorder = _continuation_result_recorder(context)
    should_stop = _reached_env_step_target(target_steps)
    try:
        algorithm = _build_or_restore_algorithm(config, context)
        _active_algorithm[0] = algorithm
        while True:
            result = algorithm.train()
            iteration += 1
            recorder(context, result)
            if should_stop(result):
                save_algorithm_checkpoint(
                    algorithm,
                    context,
                    label=f"iteration_{iteration:06d}_final",
                )
                return result
    finally:
        _active_algorithm[0] = None
        try:
            if algorithm is not None:
                algorithm.stop()
        finally:
            shutdown_ray_if_owned(started_ray)


def build_config(
    context: RunContext,
    condition: str,
    *,
    model_config: Mapping[str, Any] | None = None,
    learning_rate: float | None = None,
    num_env_runners: int | None = None,
    num_envs_per_env_runner: int | None = None,
) -> PPOConfig:
    """Build one fresh zero-baseline, full-episode REINFORCE recipe."""

    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    resolved_model = dict(model_config) if model_config is not None else MODEL_CONFIG
    batch_size = SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
    config = (
        PPOConfig()
        .environment(HMMEnv, env_config=environment_config(condition))
        .framework(
            "torch",
            torch_compile_learner=False,
            torch_compile_worker=False,
        )
        .env_runners(batch_mode="complete_episodes")
        .training(
            lr=learning_rate if learning_rate is not None else LEARNING_RATE,
            gamma=GAMMA,
            lambda_=1.0,
            use_critic=False,
            use_gae=False,
            use_kl_loss=False,
            vf_loss_coeff=VALUE_LOSS_COEFF,
            entropy_coeff=ENTROPY_COEFF,
            train_batch_size_per_learner=batch_size,
            minibatch_size=None,
            num_epochs=1,
            shuffle_batch_per_epoch=False,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TwoFactorRewardReinforceCycle4,
                model_config=resolved_model,
            )
        )
        .callbacks(
            on_algorithm_init=partial(
                _capture_algorithm_on_init,
                checkpoint_path=str(
                    context.artifacts_dir / "initial_checkpoint"
                ),
            ),
            on_train_result=_combine_on_train_result(
                partial(
                    _save_log_spaced_checkpoint,
                    checkpoint_root=str(
                        context.artifacts_dir / "log_spaced_checkpoints"
                    ),
                ),
                partial(
                    _save_step_interval_checkpoint,
                    checkpoint_root=str(
                        context.artifacts_dir / "step_checkpoints"
                    ),
                ),
            ),
        )
        .debugging(seed=context.seed)
    )
    config = apply_runtime_resources(
        config,
        _single_gpu_context(context),
        default_env_runners=16,
    )
    if (
        num_env_runners is not None
        and not context.smoke
    ):
        config = config.env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner or 1,
        )
    return config


def _resolved_recipe(
    context: RunContext,
    condition: str,
    *,
    model_config: Mapping[str, Any] | None = None,
    recipe_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_model = dict(model_config) if model_config is not None else MODEL_CONFIG
    lookback = int(resolved_model["n_layers"]) * int(resolved_model["context_len"])
    recipe = {
        "study": "two_factor_reward_state_REINFORCE_cycle_4",
        "condition": condition,
        "hypothesis": (
            "selective reward makes the rewarded factor belief more linearly "
            "accessible under independent variant-3 control when optimized by "
            "baseline-free Monte Carlo REINFORCE"
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
        "rllib_engine": "PPOConfig/PPOTorchLearner",
        "algorithm_variant": (
            "complete-episode Monte Carlo policy gradient with zero baseline, "
            "normalized returns, and one full-batch optimizer step"
        ),
        "gamma": GAMMA,
        "lambda": 1.0,
        "learning_rate": LEARNING_RATE,
        "entropy_coeff": ENTROPY_COEFF,
        "value_loss_coeff": VALUE_LOSS_COEFF,
        "train_batch_size_per_learner": TRAIN_BATCH_SIZE,
        "minibatch_size": None,
        "num_epochs": 1,
        "model": resolved_model,
        "transformer_raw_observation_lookback": lookback,
        "total_env_steps": _resolve_step_target(context),
        "continuation_spec": _load_continuation_spec(context),
        "budget_spec": _load_budget_spec(context),
        "checkpoint_schedule": (
            "initial, powers of two iterations, every "
            f"{STEP_CHECKPOINT_INTERVAL:,} env steps, final"
        ),
        "step_checkpoint_interval": STEP_CHECKPOINT_INTERVAL,
    }
    if recipe_overrides:
        recipe.update(dict(recipe_overrides))
    return recipe


def run_condition(
    context: RunContext,
    condition: str,
    *,
    config_builder: Callable[[RunContext], PPOConfig] | None = None,
    model_config: Mapping[str, Any] | None = None,
    recipe_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if context.seed is None:
        raise ValueError("two-factor REINFORCE requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        _resolved_recipe(
            context,
            condition,
            model_config=model_config,
            recipe_overrides=recipe_overrides,
        ),
    )
    target_steps = _resolve_step_target(context)
    build = config_builder or (
        lambda ctx: build_config(ctx, condition, model_config=model_config)
    )
    if context.resume_from is not None:
        final_metrics = _run_continuation_algorithm(
            build(context),
            context,
            target_steps=target_steps,
        )
        final_checkpoint = _latest_algorithm_checkpoint(context)
        result = SimpleNamespace(
            checkpoint=SimpleNamespace(path=str(final_checkpoint)),
            metrics=final_metrics,
            error=None,
        )
    else:
        result_grid = run_tune(
            build(context),
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
        if len(results) != 1 or results[0].error is not None:
            raise RuntimeError(f"{condition} REINFORCE training failed")
        result = results[0]
    write_training_curves(context)
    initial_record = (
        []
        if context.resume_from is not None
        else [
            {
                "checkpoint_path": context.artifacts_dir / "initial_checkpoint",
                "checkpoint_name": "initial_checkpoint",
                "training_iteration": 0,
                "agent_steps": 0,
            }
        ]
    )
    step_records = _step_checkpoint_records(
        context.artifacts_dir / "step_checkpoints"
    )
    log_records = checkpoint_records(
        result,
        checkpoint_root=context.artifacts_dir / "log_spaced_checkpoints",
    )
    by_steps: dict[int, dict[str, Any]] = {}
    for record in [*initial_record, *step_records, *log_records]:
        by_steps[int(record["agent_steps"])] = record
    records = sorted(by_steps.values(), key=lambda row: row["agent_steps"])
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
        "algorithm": "REINFORCE",
        "checkpoint_reports": reports,
    }
    outputs.write_json("condition_summary.json", summary)
    return summary
