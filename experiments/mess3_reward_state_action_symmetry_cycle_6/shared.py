"""REINFORCE recipe and longitudinal probes for three action variants."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import torch
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override

from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.shared import (
    apply_runtime_resources,
)
from experiments.mess3_reward_state_action_symmetry_cycle_5.shared import (
    _log_spaced_records,
    _metric,
    checkpoint_records,
)
from experiments.mess3_reward_state_action_symmetry_cycle_6.analysis import (
    ProbeResult,
    plot_probe,
    probe_checkpoint,
)
from experiments.mess3_reward_state_action_symmetry_cycle_6.design import (
    CYCLE_6_TRANSITION_MATRIX,
    EFFECT_SIZE,
    analytic_design_summary,
)
from experiments.storage.training_curves import write_training_curves
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.runners import run_algorithm, run_tune
from harness.storage.b2 import is_b2_configured

from experiments.storage.b2_incremental import upload_artifact_path
from learners.models.transformer import TransformerModel, TransformerModelConfig


TOTAL_ENV_STEPS = 8_000_000
CONTINUED_TOTAL_ENV_STEPS = 300_000_000
CHECKPOINT_EVERY_ENV_STEPS = 25_000_000
SMOKE_CHECKPOINT_EVERY_ENV_STEPS = 2_048
BUDGET_SPEC_FILENAME = "budget_spec.json"
SMOKE_ENV_STEPS = 4_096
TRAIN_BATCH_SIZE = 32_768
SMOKE_BATCH_SIZE = 2_048
LEARNING_RATE = 4.2e-4
BASE_MODEL_CONFIG = TransformerModelConfig(
    d_model=64,
    n_layers=4,
    n_heads=1,
    context_len=10,
).to_dict()


class ReinforceTransformerModel(TransformerModel):
    """Transformer policy with an identically-zero REINFORCE baseline."""

    @override(TorchRLModule)
    def setup(self):
        super().setup()
        self._sampling_temperature = float(
            self.model_config.get("sampling_temperature", 1.0)
        )
        if (
            not math.isfinite(self._sampling_temperature)
            or self._sampling_temperature <= 0
        ):
            raise ValueError("sampling_temperature must be finite and positive")

    def _outputs(
        self,
        embeddings: torch.Tensor,
        state_out: Any | None,
        *,
        training: bool,
    ) -> dict[str, Any]:
        outputs = super()._outputs(embeddings, state_out, training=training)
        temperature = self._sampling_temperature
        if temperature != 1.0:
            outputs[Columns.ACTION_DIST_INPUTS] = (
                outputs[Columns.ACTION_DIST_INPUTS] / temperature
            )
        return outputs

    def compute_values(
        self,
        batch: dict[str, Any],
        embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reference = embeddings if embeddings is not None else batch[Columns.OBS]
        return torch.zeros(
            reference.shape[:-1],
            dtype=reference.dtype,
            device=reference.device,
        )


def _single_gpu_context(context: RunContext) -> RunContext:
    """Use cuda4090 on 1-GPU boxes; gpuinfer needs 1.8 GPUs to schedule."""

    profile = context.hardware
    if (
        not context.smoke
        and profile is not None
        and profile.name == "cuda4090_gpuinfer"
    ):
        return replace(context, hardware=PROFILES["cuda4090"])
    return context


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
    budget = _load_budget_spec(context)
    if budget is not None:
        return int(budget["target_agent_steps"])
    return TOTAL_ENV_STEPS


def environment_config(variant: int) -> dict[str, Any]:
    """Build the cycle-5 sticky-state environment for one action variant."""

    if variant not in (1, 2, 3):
        raise ValueError("variant must be one of 1, 2, or 3")
    return {
        "model": {
            "factory": "envs.mess3.model:control_model",
            "kwargs": {
                "alpha": 0.85,
                "transition_matrix": [
                    list(row) for row in CYCLE_6_TRANSITION_MATRIX
                ],
            },
        },
        "task": {
            "class": (
                "experiments.mess3_reward_state_action_symmetry_cycle_6.task:"
                "ActionSymmetryTask"
            ),
            "kwargs": {
                "variant": variant,
                "effect_size": EFFECT_SIZE,
            },
        },
        "delay": 0,
        "episode_length": 1024,
        "randomize_first_episode_length": True,
    }


def build_config(
    context: RunContext,
    variant: int,
    *,
    model_config: Mapping[str, Any] | None = None,
) -> PPOConfig:
    """Build a fresh Monte Carlo REINFORCE configuration."""

    batch_size = SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
    resolved_model_config = dict(model_config or BASE_MODEL_CONFIG)
    config = (
        PPOConfig()
        .environment(HMMEnv, env_config=environment_config(variant))
        .framework(
            "torch",
            torch_compile_learner=False,
            torch_compile_worker=False,
        )
        .env_runners(batch_mode="complete_episodes")
        .training(
            lr=LEARNING_RATE,
            gamma=0.99,
            lambda_=1.0,
            use_critic=False,
            use_gae=False,
            use_kl_loss=False,
            vf_loss_coeff=0.0,
            entropy_coeff=0.0,
            train_batch_size_per_learner=batch_size,
            minibatch_size=None,
            num_epochs=1,
            shuffle_batch_per_epoch=False,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=ReinforceTransformerModel,
                model_config=resolved_model_config,
            )
        )
        .debugging(seed=context.seed)
    )
    return apply_runtime_resources(
        config,
        _single_gpu_context(context),
        default_env_runners=8,
    )


def _save_initial_checkpoint(config: PPOConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    algorithm = config.build_algo()
    try:
        saved = algorithm.save_to_path(str(path))
    finally:
        algorithm.stop()
    return Path(saved)


def _probe_at(
    context: RunContext,
    *,
    checkpoint: Path,
    condition: str,
    agent_steps: int,
) -> tuple[ProbeResult, dict[str, Any]]:
    probe_dir = (
        context.results_dir
        / "checkpoint_probes"
        / f"steps_{agent_steps:09d}"
    )
    result = probe_checkpoint(
        replace(context, results_dir=probe_dir, resume_from=checkpoint),
        checkpoint=checkpoint,
        condition=condition,
        agent_steps=agent_steps,
    )
    point = {
        "agent_steps": agent_steps,
        "mse": float(result.metrics["mse"]),
        "target_variance": float(result.metrics["target_variance"]),
        "global_mse_ratio": float(result.metrics["global_mse_ratio"]),
        "branch_baseline_mse": float(
            result.metrics["branch_baseline_mse"]
        ),
        "fine_mse_ratio": float(result.metrics["fine_mse_ratio"]),
        "reward_state_2_fraction_greedy": float(
            result.metrics["reward_state_2_fraction_greedy"]
        ),
        "greedy_action_fractions": result.metrics["greedy_action_fractions"],
        "probe": result.metrics,
    }
    return result, point


def _env_steps_lifetime(metrics: Mapping[str, Any]) -> float | None:
    return _metric(metrics, "env_runners/num_env_steps_sampled_lifetime")


def _save_step_checkpoint_and_upload(
    *,
    algorithm: Any,
    result: Mapping[str, Any],
    checkpoint_root: str,
    step_interval: int,
    context: RunContext,
    upload: bool,
    **_: Any,
) -> None:
    """Checkpoint whenever lifetime env steps cross the interval grid."""

    steps_value = _env_steps_lifetime(result)
    if steps_value is None:
        return
    steps = int(steps_value)
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
    else:
        index = {
            "step_interval": step_interval,
            "next_threshold": (steps // step_interval + 1) * step_interval,
            "checkpoints": [],
        }
        index_path.write_text(json.dumps(index, indent=2) + "\n")
    if steps < int(index["next_threshold"]):
        return
    saved = Path(algorithm.save_to_path(str(root / f"steps_{steps:09d}")))
    index["checkpoints"].append(
        {
            "path": str(saved),
            "checkpoint_name": saved.name,
            "training_iteration": int(
                _metric(result, "training_iteration") or 0
            ),
            "agent_steps": steps,
        }
    )
    while int(index["next_threshold"]) <= steps:
        index["next_threshold"] = int(index["next_threshold"]) + step_interval
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(index, indent=2) + "\n")
    temporary.replace(index_path)
    upload_summary = None
    upload_error = None
    if upload:
        try:
            upload_summary = upload_artifact_path(context, saved)
            upload_artifact_path(context, index_path)
        except Exception as error:  # noqa: BLE001 - never kill training
            upload_error = f"{type(error).__name__}: {error}"
            print(
                f"[cycle6] step checkpoint upload failed: {upload_error}",
                flush=True,
            )
    RunArtifacts.from_context(context).append_jsonl(
        "checkpoint_uploads.jsonl",
        {
            "checkpoint_name": saved.name,
            "agent_steps": steps,
            "uploaded": upload_summary is not None,
            "upload_error": upload_error,
            "file_count": (upload_summary or {}).get("file_count"),
        },
    )


def _continuation_step_target(context: RunContext) -> int:
    budget = _load_budget_spec(context)
    if budget is not None:
        return int(budget["target_agent_steps"])
    return CONTINUED_TOTAL_ENV_STEPS


def _run_continuation(
    context: RunContext,
    variant: int,
    *,
    model_config: Mapping[str, Any] | None = None,
    recipe_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Continue REINFORCE training from a completed variant checkpoint."""

    condition = f"variant_{variant}"
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    target_steps = _continuation_step_target(context)
    checkpoint_interval = (
        SMOKE_CHECKPOINT_EVERY_ENV_STEPS
        if context.smoke
        else CHECKPOINT_EVERY_ENV_STEPS
    )
    upload = is_b2_configured() and (
        not context.smoke or context.publish_smoke
    )
    resolved_model_config = dict(model_config or BASE_MODEL_CONFIG)
    recipe = {
        "condition": condition,
        "mode": "continued_from_checkpoint",
        "algorithm": "REINFORCE",
        "rllib_engine": "PPOConfig/PPOTorchLearner",
        "resume_from": str(context.resume_from),
        "target_agent_steps": target_steps,
        "target_semantics": (
            "stop when env_runners/num_env_steps_sampled_lifetime "
            "(restored from the source checkpoint) reaches the target"
        ),
        "checkpoint_every_env_steps": checkpoint_interval,
        "checkpoint_upload": (
            "each step checkpoint uploads to B2 immediately after save"
            if upload
            else "disabled (unpublished smoke or B2 not configured)"
        ),
        "gamma": 0.99,
        "learning_rate": LEARNING_RATE,
        "environment": environment_config(variant),
        "analytic_design": analytic_design_summary(),
        "model_config": resolved_model_config,
    }
    recipe.update(recipe_overrides or {})
    outputs.write_json("resolved_recipe.json", recipe)

    state: dict[str, Any] = {"baseline": None}

    def should_stop(result: Mapping[str, Any]) -> bool:
        steps = _env_steps_lifetime(result)
        if steps is None:
            return False
        if state["baseline"] is None:
            state["baseline"] = steps
        limit = (
            state["baseline"] + SMOKE_ENV_STEPS
            if context.smoke
            else target_steps
        )
        return steps >= limit

    config = build_config(
        context, variant, model_config=resolved_model_config
    ).callbacks(
        on_train_result=partial(
            _save_step_checkpoint_and_upload,
            checkpoint_root=str(context.artifacts_dir / "step_checkpoints"),
            step_interval=checkpoint_interval,
            context=context,
            upload=upload,
        )
    )
    final = run_algorithm(
        config,
        context,
        should_stop=should_stop,
        checkpoint_at_end=True,
    )
    if upload:
        for checkpoint in sorted(
            (context.artifacts_dir / "checkpoints").glob("iteration_*_final")
        ):
            try:
                upload_artifact_path(context, checkpoint)
            except Exception as error:  # noqa: BLE001
                print(
                    "[cycle6] final checkpoint upload failed: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
    curves = write_training_curves(context)
    summary = {
        "condition": condition,
        "seed": context.seed,
        "smoke": context.smoke,
        "algorithm": "REINFORCE",
        "resumed_from": str(context.resume_from),
        "baseline_agent_steps": state["baseline"],
        "final_agent_steps": _env_steps_lifetime(final),
        "training_iteration": _metric(final, "training_iteration"),
        "target_agent_steps": target_steps,
        "checkpoint_every_env_steps": checkpoint_interval,
        "episode_return_mean": _metric(
            final, "env_runners/episode_return_mean"
        ),
        "training_curves": str(curves) if curves is not None else None,
    }
    outputs.write_json("condition_summary.json", summary)
    return summary


def run_condition(
    context: RunContext,
    variant: int,
    *,
    model_config: Mapping[str, Any] | None = None,
    recipe_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one REINFORCE variant and probe init plus spaced checkpoints."""

    if context.seed is None:
        raise ValueError("action-symmetry cycle requires a resolved seed")
    if context.resume_from is not None:
        return _run_continuation(
            context,
            variant,
            model_config=model_config,
            recipe_overrides=recipe_overrides,
        )
    condition = f"variant_{variant}"
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    target_steps = _resolve_step_target(context)
    resolved_model_config = dict(model_config or BASE_MODEL_CONFIG)
    recipe = {
        "condition": condition,
        "algorithm": "REINFORCE",
        "rllib_engine": "PPOConfig/PPOTorchLearner",
        "policy_gradient_equivalence": (
            "zero baseline, lambda=1 Monte Carlo returns, normalized "
            "advantages, one full-batch epoch, no clipping-active reuse"
        ),
        "gamma": 0.99,
        "learning_rate": LEARNING_RATE,
        "environment": environment_config(variant),
        "analytic_design": analytic_design_summary(),
        "total_env_steps": target_steps,
        "checkpoint_schedule": "init_then_iterations_1_2_4_8_and_final",
        "checkpoint_storage": (
            "every_iteration_unpruned_pending_generic_log_schedule"
        ),
        "model_config": resolved_model_config,
        "probe_target": "exact_predictive_bayesian_belief",
        "probe_sampling_distribution": "process_weighted_rollout",
    }
    recipe.update(recipe_overrides or {})
    outputs.write_json("resolved_recipe.json", recipe)
    config = build_config(
        context, variant, model_config=resolved_model_config
    )
    initial_checkpoint = _save_initial_checkpoint(
        config,
        context.artifacts_dir / "initial_checkpoint",
    )
    initial_probe, initial_point = _probe_at(
        context,
        checkpoint=initial_checkpoint,
        condition=f"{condition}_init",
        agent_steps=0,
    )

    result_grid = run_tune(
        config,
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": target_steps},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=None,
                checkpoint_frequency=1,
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
    selected = _log_spaced_records(checkpoint_records(result))
    if not selected:
        raise RuntimeError(f"{condition} retained no checkpoints")

    trajectory = [initial_point]
    checkpoint_probes: list[ProbeResult] = []
    for record in selected:
        probe, point = _probe_at(
            context,
            checkpoint=Path(record["checkpoint"].path),
            condition=condition,
            agent_steps=record["agent_steps"],
        )
        checkpoint_probes.append(probe)
        trajectory.append(
            {
                **point,
                "training_iteration": record["training_iteration"],
                "checkpoint_name": record["checkpoint_name"],
            }
        )
    final_probe = checkpoint_probes[-1]
    plot_probe(
        initial_probe,
        title=f"{condition} — init",
        path=context.results_dir / "belief_simplex_init.png",
    )
    plot_probe(
        final_probe,
        title=f"{condition} — final",
        path=context.results_dir / "belief_simplex_final.png",
    )
    outputs.write_json(
        "checkpoint_probe_curve.json",
        {"condition": condition, "checkpoints": trajectory},
    )
    summary = {
        "condition": condition,
        "seed": context.seed,
        "smoke": context.smoke,
        "algorithm": "REINFORCE",
        "initial_probe": initial_probe.metrics,
        "final_probe": final_probe.metrics,
        "checkpoint_probes": trajectory,
    }
    outputs.write_json("condition_summary.json", summary)
    return summary
