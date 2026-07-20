"""Train IQN PPO for 20M steps and probe belief geometry over training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np
from ray import tune

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.mess3_token_guess_cycle_1.analysis import (
    ProbeResult,
    probe_checkpoint,
)
from experiments.mess3_token_guess_cycle_1.iqn_value.experiment import (
    build_config as build_iqn_config,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.runners import run_tune


TOTAL_ENV_STEPS = 20_000_000
SMOKE_ENV_STEPS = 4_096
CHECKPOINT_INTERVAL = 25
INTERMEDIATE_PROBE_TRAIN_STEPS = 30_000
INTERMEDIATE_PROBE_TEST_STEPS = 15_000


def build_config(context: RunContext):
    """Build the same IQN PPO configuration used by the 2.5M-step run."""

    return build_iqn_config(context)


def _metric(metrics: Mapping[str, Any], path: str) -> float | None:
    direct = metrics.get(path)
    if isinstance(direct, (int, float, np.number)):
        return float(direct)
    value: Any = metrics
    for part in path.split("/"):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return float(value) if isinstance(value, (int, float, np.number)) else None


def _steps(metrics: Mapping[str, Any]) -> int | None:
    for key in (
        "env_runners/num_env_steps_sampled_lifetime",
        "num_env_steps_sampled_lifetime",
    ):
        value = _metric(metrics, key)
        if value is not None:
            return int(value)
    return None


def _reward_percentage(metrics: Mapping[str, Any]) -> float | None:
    reward = _metric(metrics, "env_runners/episode_return_mean")
    length = _metric(metrics, "env_runners/episode_len_mean")
    if reward is None or length is None or length <= 0.0:
        return None
    return 100.0 * reward / length


def training_curve(result: Any) -> list[dict[str, float | int]]:
    """Extract every reported reward point from a Tune result."""

    dataframe = result.metrics_dataframe
    if dataframe is None:
        return []
    records: list[dict[str, float | int]] = []
    for _, row in dataframe.iterrows():
        values = row.to_dict()
        steps = _steps(values)
        reward_percentage = _reward_percentage(values)
        iteration = _metric(values, "training_iteration")
        if (
            steps is None
            or reward_percentage is None
            or iteration is None
        ):
            continue
        records.append(
            {
                "training_iteration": int(iteration),
                "agent_steps": steps,
                "reward_percentage": reward_percentage,
            }
        )
    return records


def checkpoint_records(result: Any) -> list[dict[str, Any]]:
    """Return retained checkpoints ordered by sampled agent steps."""

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for checkpoint, metrics in result.best_checkpoints or []:
        checkpoint_path = str(checkpoint.path)
        if checkpoint_path in seen:
            continue
        seen.add(checkpoint_path)
        steps = _steps(metrics)
        iteration = _metric(metrics, "training_iteration")
        reward_percentage = _reward_percentage(metrics)
        if steps is None or iteration is None:
            continue
        records.append(
            {
                "checkpoint": checkpoint,
                "checkpoint_name": Path(checkpoint_path).name,
                "training_iteration": int(iteration),
                "agent_steps": steps,
                "training_reward_percentage": reward_percentage,
            }
        )
    return sorted(records, key=lambda record: record["agent_steps"])


def _prior_conditions() -> dict[str, dict[str, float]]:
    results_root = (
        Path(__file__).parents[1]
        / "comparison"
        / "results"
    )
    candidates = sorted(results_root.glob("*/comparison_summary.json"))
    if not candidates:
        raise FileNotFoundError(
            "long IQN run requires the completed three-arm comparison"
        )
    summary = json.loads(candidates[-1].read_text())
    return {
        name: {
            "r_squared": float(values["r_squared"]),
            "token_accuracy_greedy": float(
                values["token_accuracy_greedy"]
            ),
        }
        for name, values in summary["conditions"].items()
    }


def _probe_curve(
    context: RunContext,
    checkpoints: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], ProbeResult]:
    if not checkpoints:
        raise RuntimeError("long IQN run retained no checkpoints")
    curve: list[dict[str, Any]] = []
    final_probe: ProbeResult | None = None
    for index, record in enumerate(checkpoints):
        is_final = index == len(checkpoints) - 1
        probe = probe_checkpoint(
            context,
            checkpoint=Path(record["checkpoint"].path),
            condition="iqn_value_20m",
            train_steps=(
                512
                if context.smoke
                else (
                    60_000 if is_final else INTERMEDIATE_PROBE_TRAIN_STEPS
                )
            ),
            test_steps=(
                256
                if context.smoke
                else (
                    30_000 if is_final else INTERMEDIATE_PROBE_TEST_STEPS
                )
            ),
            write_outputs=is_final,
        )
        curve.append(
            {
                "checkpoint_name": record["checkpoint_name"],
                "training_iteration": record["training_iteration"],
                "agent_steps": record["agent_steps"],
                "training_reward_percentage": record[
                    "training_reward_percentage"
                ],
                "greedy_reward_percentage": (
                    100.0 * probe.metrics["token_accuracy_greedy"]
                ),
                "belief_r_squared": probe.metrics["r_squared"],
                "belief_mse": probe.metrics["mse"],
                "probe_fit_samples": probe.metrics["n_fit"],
                "probe_test_samples": probe.metrics["n_test"],
            }
        )
        if is_final:
            final_probe = probe
    assert final_probe is not None
    return curve, final_probe


def _plot_curves(
    training: list[dict[str, Any]],
    probes: list[dict[str, Any]],
    prior: dict[str, dict[str, float]],
    *,
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9.0, 8.0), sharex=True)
    training_steps = np.asarray(
        [record["agent_steps"] for record in training],
        dtype=np.float64,
    ) / 1_000_000.0
    axes[0].plot(
        training_steps,
        [record["reward_percentage"] for record in training],
        color="black",
        linewidth=1.4,
        label="IQN training reward",
    )
    probe_steps = np.asarray(
        [record["agent_steps"] for record in probes],
        dtype=np.float64,
    ) / 1_000_000.0
    axes[0].plot(
        probe_steps,
        [record["greedy_reward_percentage"] for record in probes],
        "o-",
        markersize=3.5,
        label="IQN held-out greedy reward",
    )
    axes[1].plot(
        probe_steps,
        [record["belief_r_squared"] for record in probes],
        "o-",
        color="tab:blue",
        markersize=3.5,
        label="IQN held-out belief R²",
    )
    colors = ("tab:gray", "tab:orange", "tab:green")
    for (name, values), color in zip(prior.items(), colors):
        label = name.replace("_", " ")
        axes[0].axhline(
            100.0 * values["token_accuracy_greedy"],
            color=color,
            linestyle="--",
            linewidth=0.9,
            alpha=0.8,
            label=f"{label} final",
        )
        axes[1].axhline(
            values["r_squared"],
            color=color,
            linestyle="--",
            linewidth=0.9,
            alpha=0.8,
            label=f"{label} final",
        )
    axes[0].set_ylabel("correct token guesses (%)")
    axes[0].set_title("IQN reward learning over 20M sampled agent steps")
    axes[1].set_ylabel("held-out belief-probe R²")
    axes[1].set_xlabel("sampled agent steps (millions)")
    axes[1].set_ylim(0.0, 1.0)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _findings(
    probes: list[dict[str, Any]],
    final_probe: ProbeResult,
) -> str:
    first = probes[0]
    final = probes[-1]
    return "\n".join(
        [
            "# IQN 20M-step learning curve",
            "",
            f"- Checkpoints probed: {len(probes)}",
            f"- First checkpoint: {first['agent_steps']:,} agent steps, "
            f"{first['greedy_reward_percentage']:.2f}% greedy reward, "
            f"R² {first['belief_r_squared']:.4f}",
            f"- Final checkpoint: {final['agent_steps']:,} agent steps, "
            f"{final['greedy_reward_percentage']:.2f}% greedy reward, "
            f"R² {final['belief_r_squared']:.4f}",
            f"- Final probe MSE: {final_probe.metrics['mse']:.6f}",
            "",
        ]
    )


def run(context: RunContext):
    if context.seed is None:
        raise ValueError("the long IQN run requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    prior = _prior_conditions()
    checkpoint_interval = 1 if context.smoke else CHECKPOINT_INTERVAL
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    outputs.write_json(
        "resolved_recipe.json",
        {
            "condition": "iqn_value_20m",
            "total_env_steps": target_steps,
            "checkpoint_interval_iterations": checkpoint_interval,
            "checkpoint_retention": "all",
            "intermediate_probe_train_steps": (
                512 if context.smoke else INTERMEDIATE_PROBE_TRAIN_STEPS
            ),
            "intermediate_probe_test_steps": (
                256 if context.smoke else INTERMEDIATE_PROBE_TEST_STEPS
            ),
            "comparison_conditions": list(prior),
        },
    )
    result_grid = run_tune(
        build_config(context),
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": target_steps},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=None,
                checkpoint_frequency=checkpoint_interval,
                checkpoint_at_end=True,
            ),
        },
    )
    results = list(result_grid)
    if len(results) != 1:
        raise RuntimeError(f"long IQN run expected one trial, got {len(results)}")
    result = results[0]
    if result.error is not None:
        raise RuntimeError("long IQN training failed") from result.error

    training = training_curve(result)
    checkpoints = checkpoint_records(result)
    probes, final_probe = _probe_curve(context, checkpoints)
    figure_path = context.results_dir / "iqn_20m_learning_curves.png"
    _plot_curves(training, probes, prior, path=figure_path)
    compact_checkpoints = [
        {
            key: value
            for key, value in record.items()
            if key != "checkpoint"
        }
        for record in checkpoints
    ]
    summary = {
        "seed": context.seed,
        "smoke": context.smoke,
        "target_agent_steps": target_steps,
        "training_curve": training,
        "checkpoint_index": compact_checkpoints,
        "probe_curve": probes,
        "prior_conditions": prior,
        "final_probe": final_probe.metrics,
        "learning_curve_figure": str(figure_path),
    }
    outputs.write_json("iqn_20m_summary.json", summary)
    outputs.write_json("probe_curve.json", {"checkpoints": probes})
    outputs.write_json("reward_curve.json", {"iterations": training})
    (context.results_dir / "findings.md").write_text(
        _findings(probes, final_probe)
    )
    return summary
