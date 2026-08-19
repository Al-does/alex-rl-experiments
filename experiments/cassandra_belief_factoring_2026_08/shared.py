"""Transformer PPO recipe and longitudinal probes for Cassandra maintenance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
import torch

from analysis.checkpoints import load_algorithm
from envs.cassandra_machine import DISCOUNT, action_names
from experiments.cassandra_belief_factoring_2026_08.analysis import (
    ProbeResult,
    plot_probe_trajectory,
    probe_checkpoint,
)
from experiments.cassandra_belief_factoring_2026_08.environment import (
    CassandraActionObservationEnv,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune
from harness.seeding import (
    child_seed_sequence,
    named_seed_sequences,
    seed_sequence_to_int,
)
from learners.models.transformer import TransformerModel, TransformerModelConfig


TOTAL_ENV_STEPS = 5_000_000
SMOKE_ENV_STEPS = 4_096
TRAIN_BATCH_SIZE = 4_096
SMOKE_BATCH_SIZE = 2_048
MINIBATCH_SIZE = 512
SMOKE_MINIBATCH_SIZE = 256
FULL_EVAL_EPISODES = 64
SMOKE_EVAL_EPISODES = 2
ENTROPY_COEFF = 0.005
EPISODE_LENGTH = 1_000
_EVALUATION_STREAMS = {"policy_eval": (500,)}
MODEL_CONFIG = TransformerModelConfig(
    d_model=96,
    n_layers=3,
    n_heads=4,
    context_len=256,
    max_seq_len=256,
).to_dict()


def environment_config(
    *,
    action_scope: str = "global",
) -> dict[str, Any]:
    """Return the hidden-symbol task with public diagnostics off."""

    return {
        "episode_length": EPISODE_LENGTH,
        "action_scope": action_scope,
        "initial_state_distribution": "uniform",
        "diagnostics": False,
    }


def _apply_runtime_resources(config: PPOConfig, context: RunContext) -> PPOConfig:
    profile = context.hardware or PROFILES["cpu"]
    return config.env_runners(
        num_env_runners=(
            0
            if context.smoke
            else resolve_env_runners(profile, default=8)
        ),
        num_envs_per_env_runner=(
            1 if context.smoke else profile.num_envs_per_env_runner
        ),
        num_gpus_per_env_runner=(
            0 if context.smoke else profile.num_gpus_per_env_runner
        ),
        sample_timeout_s=600.0,
    ).learners(
        num_gpus_per_learner=(
            1 if profile.learner_device == "cuda" else 0
        )
    )


def _save_log_spaced_checkpoint(
    *,
    algorithm: Any,
    result: Mapping[str, Any],
    checkpoint_root: str,
    **_: Any,
) -> None:
    """Save public Algorithm checkpoints at power-of-two iterations."""

    iteration_value = _metric(result, "training_iteration")
    steps_value = _metric(
        result,
        "env_runners/num_env_steps_sampled_lifetime",
    )
    if iteration_value is None or steps_value is None:
        return
    iteration = int(iteration_value)
    if iteration <= 0 or iteration.bit_count() != 1:
        return
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    records = (
        json.loads(index_path.read_text()).get("checkpoints", [])
        if index_path.is_file()
        else []
    )
    if any(int(record["training_iteration"]) == iteration for record in records):
        return
    destination = root / (
        f"iteration_{iteration:06d}_steps_{int(steps_value):09d}"
    )
    saved = Path(algorithm.save_to_path(str(destination)))
    records.append(
        {
            "path": str(saved),
            "checkpoint_name": saved.name,
            "training_iteration": iteration,
            "agent_steps": int(steps_value),
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
    action_scope: str = "global",
) -> PPOConfig:
    """Build a fresh transformer PPO configuration."""

    batch_size = SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
    config = (
        PPOConfig()
        .environment(
            CassandraActionObservationEnv,
            env_config=environment_config(action_scope=action_scope),
        )
        .framework(
            "torch",
            torch_compile_learner=False,
            torch_compile_worker=False,
        )
        .training(
            lr=3e-4,
            gamma=DISCOUNT,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=ENTROPY_COEFF,
            train_batch_size_per_learner=batch_size,
            minibatch_size=(
                SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE
            ),
            num_epochs=4,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TransformerModel,
                model_config=dict(MODEL_CONFIG),
            )
        )
        .callbacks(
            on_train_result=partial(
                _save_log_spaced_checkpoint,
                checkpoint_root=str(
                    context.artifacts_dir / "log_spaced_checkpoints"
                ),
            )
        )
        .debugging(seed=context.seed)
    )
    return _apply_runtime_resources(config, context)


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


def checkpoint_records(
    result: Any,
    *,
    checkpoint_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Return custom log-spaced plus final Tune checkpoints."""

    by_iteration: dict[int, dict[str, Any]] = {}
    index_path = (
        checkpoint_root / "index.json"
        if checkpoint_root is not None
        else None
    )
    if index_path is not None and index_path.is_file():
        for record in json.loads(index_path.read_text()).get("checkpoints", []):
            iteration = int(record["training_iteration"])
            path = Path(record["path"])
            by_iteration[iteration] = {
                "checkpoint_path": path,
                "checkpoint_name": str(record.get("checkpoint_name", path.name)),
                "training_iteration": iteration,
                "agent_steps": int(record["agent_steps"]),
            }

    candidates = list(result.best_checkpoints or [])
    if result.checkpoint is not None:
        candidates.append((result.checkpoint, result.metrics or {}))
    seen: set[str] = set()
    for checkpoint, metrics in candidates:
        path = str(checkpoint.path)
        if path in seen:
            continue
        steps = _metric(metrics, "env_runners/num_env_steps_sampled_lifetime")
        iteration = _metric(metrics, "training_iteration")
        if steps is None or iteration is None:
            continue
        seen.add(path)
        by_iteration[int(iteration)] = {
            "checkpoint_path": Path(path),
            "checkpoint_name": Path(path).name,
            "training_iteration": int(iteration),
            "agent_steps": int(steps),
        }
    return [by_iteration[key] for key in sorted(by_iteration)]


def log_spaced_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select iterations 1, 2, 4, ... and always retain the final one."""

    if not records:
        return []
    selected = [
        record
        for record in records
        if record["training_iteration"] > 0
        and record["training_iteration"].bit_count() == 1
    ]
    if not selected or (
        selected[-1]["checkpoint_name"] != records[-1]["checkpoint_name"]
    ):
        selected.append(records[-1])
    return selected


def training_curve(result: Any) -> list[dict[str, Any]]:
    """Extract compact reward and episode statistics from Tune history."""

    dataframe = getattr(result, "metrics_dataframe", None)
    if dataframe is None:
        return []
    records: list[dict[str, Any]] = []
    for _, row in dataframe.iterrows():
        values = row.to_dict()
        iteration = _metric(values, "training_iteration")
        steps = _metric(
            values,
            "env_runners/num_env_steps_sampled_lifetime",
        )
        if iteration is None or steps is None:
            continue
        records.append(
            {
                "training_iteration": int(iteration),
                "sampled_env_steps": int(steps),
                "num_episodes": (
                    int(episodes)
                    if (
                        episodes := _metric(
                            values,
                            "env_runners/num_episodes",
                        )
                    )
                    is not None
                    else None
                ),
                "episode_return_mean": _metric(
                    values,
                    "env_runners/episode_return_mean",
                ),
                "episode_return_min": _metric(
                    values,
                    "env_runners/episode_return_min",
                ),
                "episode_return_max": _metric(
                    values,
                    "env_runners/episode_return_max",
                ),
                "episode_len_mean": _metric(
                    values,
                    "env_runners/episode_len_mean",
                ),
            }
        )
    return records


def _last_reported_return(
    curve: list[dict[str, Any]],
) -> dict[str, Any] | None:
    return next(
        (
            point
            for point in reversed(curve)
            if point["episode_return_mean"] is not None
            and (point["num_episodes"] or 0) > 0
        ),
        None,
    )


def _plot_reward_curve(curve: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    points = [
        point for point in curve if point["episode_return_mean"] is not None
    ]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    if points:
        axis.plot(
            [point["sampled_env_steps"] for point in points],
            [point["episode_return_mean"] for point in points],
            linewidth=2,
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No iterations completed an episode",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_xlabel("Sampled environment steps")
    axis.set_ylabel("Mean completed-episode return")
    axis.set_title("PPO training reward")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _initial_module_state(
    module: Any,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(value)
        .unsqueeze(0)
        .repeat(batch_size, *([1] * value.ndim))
        .to(device)
        for key, value in module.get_initial_state().items()
    }


@torch.inference_mode()
def evaluate_policy_checkpoint(
    context: RunContext,
    *,
    checkpoint: Path,
    n_episodes: int,
) -> dict[str, Any]:
    """Run deterministic fixed-seed episodes from a frozen checkpoint."""

    if context.seed is None:
        raise ValueError("policy evaluation requires a resolved seed")
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    profile = context.hardware or PROFILES["cpu"]
    device = torch.device(
        "cuda"
        if profile.learner_device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    streams = named_seed_sequences(context.seed, _EVALUATION_STREAMS)
    returns: list[float] = []
    action_counts: np.ndarray | None = None
    batch_limit = min(8, n_episodes)

    with load_algorithm(checkpoint) as algorithm:
        module = algorithm.get_module()
        if module is None:
            raise KeyError("checkpoint has no default RLModule")
        module = module.to(device).eval()
        env_class = algorithm.config.env
        env_config = dict(algorithm.config.env_config)
        env_config["diagnostics"] = False
        action_scope = str(env_config.get("action_scope", "global"))
        names = action_names(action_scope)
        action_counts = np.zeros(len(names), dtype=np.int64)

        for start in range(0, n_episodes, batch_limit):
            batch_size = min(batch_limit, n_episodes - start)
            environments = [env_class(env_config) for _ in range(batch_size)]
            try:
                observations = []
                for offset, environment in enumerate(environments):
                    episode = start + offset
                    seed = seed_sequence_to_int(
                        child_seed_sequence(
                            streams["policy_eval"],
                            (episode,),
                        )
                    )
                    observation, _ = environment.reset(seed=seed)
                    observations.append(np.asarray(observation))
                state = _initial_module_state(module, batch_size, device)
                episode_returns = np.zeros(batch_size, dtype=np.float64)
                finished = np.zeros(batch_size, dtype=bool)
                while not finished.all():
                    observation_tensor = torch.from_numpy(
                        np.stack(observations)
                    ).float().to(device)
                    embeddings, state = module.encode_step(
                        observation_tensor,
                        state,
                    )
                    logits = module.action_distribution_inputs(embeddings)
                    actions = logits.argmax(dim=-1).cpu().numpy()
                    next_observations = []
                    for index, environment in enumerate(environments):
                        if finished[index]:
                            next_observations.append(observations[index])
                            continue
                        (
                            next_observation,
                            reward,
                            terminated,
                            truncated,
                            _,
                        ) = environment.step(int(actions[index]))
                        episode_returns[index] += float(reward)
                        action_counts[int(actions[index])] += 1
                        finished[index] = terminated or truncated
                        next_observations.append(
                            np.asarray(next_observation)
                        )
                    observations = next_observations
                returns.extend(episode_returns.tolist())
            finally:
                for environment in environments:
                    environment.close()

    values = np.asarray(returns, dtype=np.float64)
    fractions = action_counts / max(int(action_counts.sum()), 1)
    return {
        "checkpoint": str(checkpoint),
        "seed": context.seed,
        "n_episodes": n_episodes,
        "episode_length": EPISODE_LENGTH,
        "action_selection": "deterministic_argmax",
        "episode_return_mean": float(values.mean()),
        "episode_return_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "episode_return_min": float(values.min()),
        "episode_return_max": float(values.max()),
        "episode_returns": values.tolist(),
        "action_counts": action_counts.tolist(),
        "action_fractions": fractions.tolist(),
        "action_fractions_by_name": {
            name: float(fraction)
            for name, fraction in zip(names, fractions)
        },
    }


def _validate_compact_outputs(
    context: RunContext,
    required: list[Path],
) -> None:
    missing = [
        str(path.relative_to(context.results_dir))
        for path in required
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            "Cassandra run is missing required compact outputs: "
            + ", ".join(missing)
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
    return result, {
        "agent_steps": agent_steps,
        "targets": result.metrics["targets"],
        "factor_specific": result.metrics["factor_specific"],
        "geometry": result.metrics["geometry"],
        "hypothesis_diagnostics": result.metrics["hypothesis_diagnostics"],
        "behavior_reward_mean": result.metrics["behavior_reward_mean"],
    }


def _training_change(
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "target_r_squared_delta": {
            name: (
                float(final["targets"][name]["r_squared"])
                - float(initial["targets"][name]["r_squared"])
            )
            for name in initial["targets"]
        },
        "coarse_over_identity_r2_advantage_delta": (
            float(
                final["hypothesis_diagnostics"][
                    "coarse_over_identity_r2_advantage"
                ]
            )
            - float(
                initial["hypothesis_diagnostics"][
                    "coarse_over_identity_r2_advantage"
                ]
            )
        ),
        "component_subspace_overlap_delta": (
            float(
                final["hypothesis_diagnostics"][
                    "mean_component_subspace_overlap"
                ]
            )
            - float(
                initial["hypothesis_diagnostics"][
                    "mean_component_subspace_overlap"
                ]
            )
        ),
    }


def run_condition(
    context: RunContext,
    *,
    action_scope: str = "global",
    condition: str = "global_actions_transformer_ppo",
    hypothesis: str = (
        "Global actions create pressure for a coarse, permutation-invariant "
        "machine representation rather than separate labeled component "
        "beliefs."
    ),
) -> dict[str, Any]:
    """Train PPO and probe initialization plus log-spaced checkpoints."""

    if context.seed is None:
        raise ValueError("Cassandra belief factoring requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    action_count = len(action_names(action_scope))
    outputs.write_json(
        "resolved_recipe.json",
        {
            "condition": condition,
            "hypothesis": hypothesis,
            "primary_comparison": (
                "trained-minus-initialization decodability of aggregate belief "
                "versus identity-specific component residuals"
            ),
            "algorithm": "PPO",
            "gamma": DISCOUNT,
            "lambda": 0.95,
            "entropy_coeff": ENTROPY_COEFF,
            "environment": environment_config(action_scope=action_scope),
            "policy_observation": (
                f"16-way symbol one-hot plus previous {action_count}-way "
                "action one-hot; no belief, hidden state, or reward"
            ),
            "model_config": MODEL_CONFIG,
            "bptt_sequence_length": MODEL_CONFIG["max_seq_len"],
            "total_env_steps": target_steps,
            "checkpoint_schedule": (
                "initialization_then_power_of_two_iterations_and_final"
            ),
            "probe_sampling": (
                "fixed checkpoint-independent behavior policy with disjoint "
                "train/test seed streams"
            ),
            "probe_representation": "pre_final_layer_norm_decision_token",
            "policy_evaluation": (
                "fixed-seed deterministic_argmax final-checkpoint episodes"
            ),
            "paper": "https://arxiv.org/abs/2602.02385",
            "controls": [
                "step-zero untrained transformer",
                "current-observation-plus-previous-action linear baseline",
                "target PCA dimensions",
                "independent frozen final-policy return and action frequencies",
            ],
        },
    )

    config = build_config(context, action_scope=action_scope)
    initial_checkpoint = _save_initial_checkpoint(
        config,
        context.artifacts_dir / "initial_checkpoint",
    )
    initial_probe, initial_point = _probe_at(
        context,
        checkpoint=initial_checkpoint,
        condition=f"{condition}_initialization",
        agent_steps=0,
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
        raise RuntimeError(f"{condition} expected one trial, got {len(results)}")
    result = results[0]
    if result.error is not None:
        raise RuntimeError(f"{condition} training failed") from result.error
    curve = training_curve(result)
    outputs.write_json(
        "reward_curve.json",
        {
            "condition": condition,
            "seed": context.seed,
            "iterations": curve,
        },
    )
    _plot_reward_curve(curve, context.results_dir / "reward_curve.png")
    selected = log_spaced_records(
        checkpoint_records(
            result,
            checkpoint_root=(
                context.artifacts_dir / "log_spaced_checkpoints"
            ),
        )
    )
    if not selected:
        raise RuntimeError(f"{condition} retained no checkpoints")

    trajectory = [initial_point]
    trained_probes: list[ProbeResult] = []
    for record in selected:
        probe, point = _probe_at(
            context,
            checkpoint=record["checkpoint_path"],
            condition=condition,
            agent_steps=record["agent_steps"],
        )
        trained_probes.append(probe)
        trajectory.append(
            {
                **point,
                "training_iteration": record["training_iteration"],
                "checkpoint_name": record["checkpoint_name"],
            }
        )
    final_probe = trained_probes[-1]
    plot_probe_trajectory(
        trajectory,
        path=context.results_dir / "probe_trajectory.png",
    )
    outputs.write_json(
        "checkpoint_probe_curve.json",
        {"condition": condition, "checkpoints": trajectory},
    )

    evaluation = evaluate_policy_checkpoint(
        context,
        checkpoint=selected[-1]["checkpoint_path"],
        n_episodes=(
            SMOKE_EVAL_EPISODES if context.smoke else FULL_EVAL_EPISODES
        ),
    )
    outputs.write_json("policy_eval.json", evaluation)
    last_reported = _last_reported_return(curve)
    final_iteration_return = _metric(
        result.metrics or {},
        "env_runners/episode_return_mean",
    )
    training_metrics = {
        "episode_return_mean": (
            last_reported["episode_return_mean"]
            if last_reported is not None
            else None
        ),
        "episode_return_source": (
            "last_iteration_with_completed_episodes"
            if last_reported is not None
            else "unavailable"
        ),
        "episode_return_mean_final_iteration": final_iteration_return,
        "episode_return_last_reported_iteration": (
            last_reported["training_iteration"]
            if last_reported is not None
            else None
        ),
        "episode_return_last_reported_steps": (
            last_reported["sampled_env_steps"]
            if last_reported is not None
            else None
        ),
        "episode_len_mean": (
            last_reported["episode_len_mean"]
            if last_reported is not None
            else None
        ),
        "sampled_env_steps": _metric(
            result.metrics or {},
            "env_runners/num_env_steps_sampled_lifetime",
        ),
        "policy_evaluation_return_mean": evaluation[
            "episode_return_mean"
        ],
        "policy_evaluation_episodes": evaluation["n_episodes"],
    }
    summary = {
        "condition": condition,
        "seed": context.seed,
        "smoke": context.smoke,
        "algorithm": "PPO",
        "training_metrics": training_metrics,
        "training_curve": curve,
        "policy_evaluation": evaluation,
        "initial_probe": initial_probe.metrics,
        "final_probe": final_probe.metrics,
        "training_change": _training_change(
            initial_probe.metrics,
            final_probe.metrics,
        ),
        "checkpoint_probes": trajectory,
        "conclusion_status": (
            "smoke_diagnostic_only"
            if context.smoke
            else "single_seed_exploratory"
        ),
    }
    outputs.write_json("condition_summary.json", summary)
    required = [
        context.results_dir / "resolved_recipe.json",
        context.results_dir / "tune_summary.json",
        context.results_dir / "progress.jsonl",
        context.results_dir / "reward_curve.json",
        context.results_dir / "reward_curve.png",
        context.results_dir / "policy_eval.json",
        context.results_dir / "checkpoint_probe_curve.json",
        context.results_dir / "probe_trajectory.png",
        context.results_dir / "condition_summary.json",
        (
            context.results_dir
            / "checkpoint_probes"
            / "steps_000000000"
            / "probe_metrics.json"
        ),
        (
            context.results_dir
            / "checkpoint_probes"
            / f"steps_{selected[-1]['agent_steps']:09d}"
            / "probe_metrics.json"
        ),
    ]
    _validate_compact_outputs(context, required)
    outputs.write_json(
        "output_validation.json",
        {
            "status": "completed",
            "required_files": [
                str(path.relative_to(context.results_dir))
                for path in required
            ],
        },
    )
    return summary
