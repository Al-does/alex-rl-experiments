"""Compare myopic and differential average-reward IQN PPO."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from ray import tune

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.plots import simplex_scatter
from experiments.mess3_token_guess_cycle_1.analysis import (
    ProbeResult,
    probe_checkpoint,
)
from experiments.mess3_token_guess_cycle_1.average_reward import (
    EMA_DECAY_KEY,
    VALUE_ANCHOR_COEFFICIENT_KEY,
    AverageRewardIQNPPOTorchLearner,
)
from experiments.mess3_token_guess_cycle_1.iqn_value.experiment import (
    LEARNER_CONFIG,
    build_config as build_standard_iqn_config,
)
from experiments.mess3_token_guess_cycle_1.iqn_value_20m.experiment import (
    _metric,
    _steps,
)
from harness.artifacts import RunArtifacts, flatten_scalar_metrics
from harness.context import RunContext
from harness.runners import run_tune


TOTAL_ENV_STEPS = 3_000_000
SMOKE_ENV_STEPS = 4_096
AVERAGE_REWARD_EMA_DECAY = 0.95
VALUE_ANCHOR_COEFFICIENT = 0.01


@dataclass(frozen=True, slots=True)
class Arm:
    name: str
    objective: str


ARMS = (
    Arm("gamma_zero", "immediate reward with gamma=0"),
    Arm(
        "average_reward",
        "differential reward r-rho with gamma=1",
    ),
)


def build_config(
    context: RunContext,
    condition: str = "gamma_zero",
):
    """Build one controlled IQN return-objective condition."""

    config = build_standard_iqn_config(context)
    if condition == "gamma_zero":
        return config.training(gamma=0.0)
    if condition == "average_reward":
        return config.learners(
            learner_class=AverageRewardIQNPPOTorchLearner,
            learner_config_dict={
                **LEARNER_CONFIG,
                EMA_DECAY_KEY: AVERAGE_REWARD_EMA_DECAY,
                VALUE_ANCHOR_COEFFICIENT_KEY: VALUE_ANCHOR_COEFFICIENT,
            },
        ).training(gamma=1.0)
    raise ValueError(f"unknown return-objective condition {condition!r}")


def _subcontext(context: RunContext, arm: Arm) -> RunContext:
    return replace(
        context,
        results_dir=context.results_dir / arm.name,
        artifacts_dir=context.artifacts_dir / arm.name,
        resume_from=None,
    )


def _training_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
    flattened = flatten_scalar_metrics(metrics)

    def value(path: str) -> float | None:
        direct = flattened.get(path)
        return float(direct) if isinstance(direct, (int, float)) else None

    return {
        "episode_return_mean": value("env_runners/episode_return_mean"),
        "episode_len_mean": value("env_runners/episode_len_mean"),
        "iqn_loss": value("learners/default_policy/iqn_value/loss"),
        "mean_quantile_spread": value(
            "learners/default_policy/iqn_value/mean_quantile_spread"
        ),
        "value_explained_variance": value(
            "learners/default_policy/vf_explained_var"
        ),
        "average_reward_estimate": value(
            "learners/default_policy/average_reward/estimate"
        ),
        "value_anchor_loss": value(
            "learners/default_policy/average_reward/value_anchor_loss"
        ),
    }


def _run_arm(
    context: RunContext,
    arm: Arm,
) -> tuple[ProbeResult, dict[str, Any]]:
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    outputs.write_json(
        "resolved_recipe.json",
        {
            "condition": arm.name,
            "objective": arm.objective,
            "total_env_steps": target_steps,
            "average_reward_ema_decay": (
                AVERAGE_REWARD_EMA_DECAY
                if arm.name == "average_reward"
                else None
            ),
            "value_anchor_coefficient": (
                VALUE_ANCHOR_COEFFICIENT
                if arm.name == "average_reward"
                else None
            ),
        },
    )
    result_grid = run_tune(
        build_config(context, arm.name),
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": target_steps},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=True,
            ),
        },
    )
    results = list(result_grid)
    if len(results) != 1:
        raise RuntimeError(f"{arm.name} expected one trial, got {len(results)}")
    result = results[0]
    if result.error is not None:
        raise RuntimeError(f"{arm.name} training failed") from result.error
    if result.checkpoint is None:
        raise RuntimeError(f"{arm.name} produced no final checkpoint")
    probe = probe_checkpoint(
        context,
        checkpoint=Path(result.checkpoint.path),
        condition=arm.name,
    )
    sampled_steps = _steps(result.metrics or {})
    if sampled_steps is None:
        raise RuntimeError(f"{arm.name} result omitted sampled steps")
    summary = {
        "condition": arm.name,
        "objective": arm.objective,
        "sampled_agent_steps": sampled_steps,
        "probe": probe.metrics,
        "training": _training_metrics(result.metrics or {}),
    }
    outputs.write_json("condition_summary.json", summary)
    return probe, summary


def _latest_json(root: Path, filename: str) -> dict[str, Any]:
    candidates = sorted(root.glob(f"*/{filename}"))
    if not candidates:
        raise FileNotFoundError(f"no prior {filename} under {root}")
    return json.loads(candidates[-1].read_text())


def _references() -> dict[str, dict[str, float]]:
    family = Path(__file__).parents[1]
    standard = _latest_json(
        family / "iqn_value" / "results",
        "iqn_comparison_summary.json",
    )
    gamma_one = _latest_json(
        family / "iqn_gamma_1_3m" / "results",
        "gamma_1_summary.json",
    )
    return {
        "gamma_0.99": {
            "r_squared": float(
                standard["conditions"]["iqn_value"]["r_squared"]
            ),
            "token_accuracy": float(
                standard["conditions"]["iqn_value"][
                    "token_accuracy_greedy"
                ]
            ),
        },
        "gamma_1.0": {
            "r_squared": float(gamma_one["probe"]["r_squared"]),
            "token_accuracy": float(
                gamma_one["probe"]["token_accuracy_greedy"]
            ),
        },
    }


def _plot(
    probes: dict[str, ProbeResult],
    comparison: dict[str, dict[str, float]],
    *,
    path: Path,
) -> None:
    figure = plt.figure(figsize=(10.5, 12.0))
    grid = figure.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 0.9))
    for row, (name, probe) in enumerate(probes.items()):
        colors = np.clip(probe.target_display, 0.0, 1.0)
        exact_axis = figure.add_subplot(grid[row, 0])
        decoded_axis = figure.add_subplot(grid[row, 1])
        simplex_scatter(
            exact_axis,
            probe.target_display,
            colors=colors,
            s=0.5,
            alpha=0.32,
            title=f"{name.replace('_', ' ')}: exact Bayesian",
            labels=("s0", "s1", "s2"),
        )
        simplex_scatter(
            decoded_axis,
            probe.decoded_display,
            colors=colors,
            s=0.5,
            alpha=0.32,
            title=(
                f"{name.replace('_', ' ')}: affine decode\n"
                f"R²={probe.metrics['r_squared']:.4f}"
            ),
            labels=("s0", "s1", "s2"),
        )
    axis = figure.add_subplot(grid[2, :])
    names = list(comparison)
    positions = np.arange(len(names))
    width = 0.36
    axis.bar(
        positions - width / 2,
        [comparison[name]["r_squared"] for name in names],
        width,
        label="belief-probe R²",
    )
    axis.bar(
        positions + width / 2,
        [comparison[name]["token_accuracy"] for name in names],
        width,
        label="greedy token accuracy",
    )
    axis.set_xticks(positions, [name.replace("_", " ") for name in names])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("held-out score")
    axis.set_title("IQN return-objective comparison")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _findings(
    comparison: dict[str, dict[str, float]],
    summaries: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# IQN return-objective comparison",
        "",
        "| condition | held-out R² | greedy token accuracy |",
        "|---|---:|---:|",
    ]
    for name, metrics in comparison.items():
        lines.append(
            f"| {name} | {metrics['r_squared']:.4f} | "
            f"{metrics['token_accuracy']:.4f} |"
        )
    average_training = summaries["average_reward"]["training"]
    lines.extend(
        [
            "",
            f"Final average-reward estimate: "
            f"{average_training['average_reward_estimate']}",
            f"Final average-reward value anchor loss: "
            f"{average_training['value_anchor_loss']}",
            "",
        ]
    )
    return "\n".join(lines)


def run(context: RunContext):
    if context.seed is None:
        raise ValueError("the return-objective comparison needs a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    probes: dict[str, ProbeResult] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        probe, summary = _run_arm(_subcontext(context, arm), arm)
        probes[arm.name] = probe
        summaries[arm.name] = summary

    comparison = _references()
    for name, probe in probes.items():
        comparison[name] = {
            "r_squared": float(probe.metrics["r_squared"]),
            "token_accuracy": float(
                probe.metrics["token_accuracy_greedy"]
            ),
        }
    figure_path = context.results_dir / "return_objective_comparison.png"
    _plot(probes, comparison, path=figure_path)
    result = {
        "seed": context.seed,
        "smoke": context.smoke,
        "conditions": summaries,
        "comparison": comparison,
        "figure": str(figure_path),
    }
    outputs.write_json("return_objective_summary.json", result)
    (context.results_dir / "findings.md").write_text(
        _findings(comparison, summaries)
    )
    return result
