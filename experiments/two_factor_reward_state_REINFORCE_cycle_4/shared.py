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
    TRANSFORMER_LOOKBACK,
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
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.runners import run_algorithm, run_tune
from learners.models.transformer import TransformerModelConfig


TOTAL_ENV_STEPS = 8_000_000
CONTINUATION_TOTAL_ENV_STEPS = 16_000_000
STEP_CHECKPOINT_INTERVAL = 2_000_000
CONTINUATION_SPEC_FILENAME = "continuation_spec.json"
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


def _resolve_step_target(context: RunContext) -> int:
    if context.smoke:
        return SMOKE_ENV_STEPS
    spec = _load_continuation_spec(context)
    if spec is not None:
        return int(spec["target_agent_steps"])
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


def build_config(context: RunContext, condition: str) -> PPOConfig:
    """Build one fresh zero-baseline, full-episode REINFORCE recipe."""

    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
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
            lr=LEARNING_RATE,
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
                model_config=dict(MODEL_CONFIG),
            )
        )
        .callbacks(
            on_algorithm_init=partial(
                _save_initial_checkpoint,
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
    return apply_runtime_resources(
        config,
        _single_gpu_context(context),
        default_env_runners=16,
    )


def _resolved_recipe(context: RunContext, condition: str) -> dict[str, Any]:
    return {
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
        "model": MODEL_CONFIG,
        "transformer_raw_observation_lookback": TRANSFORMER_LOOKBACK,
        "total_env_steps": _resolve_step_target(context),
        "continuation_spec": _load_continuation_spec(context),
        "checkpoint_schedule": (
            "initial, powers of two iterations, every "
            f"{STEP_CHECKPOINT_INTERVAL:,} env steps, final"
        ),
        "step_checkpoint_interval": STEP_CHECKPOINT_INTERVAL,
    }


def run_condition(context: RunContext, condition: str) -> dict[str, Any]:
    if context.seed is None:
        raise ValueError("two-factor REINFORCE requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        _resolved_recipe(context, condition),
    )
    target_steps = _resolve_step_target(context)
    if context.resume_from is not None:
        final_metrics = run_algorithm(
            build_config(context, condition),
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
            build_config(context, condition),
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
