"""Paper-style MSE-over-training bar charts for token-guess cycle 2."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


CONDITIONS = (
    "a2c",
    "ppo",
    "predictive_loss",
    "decoupled_kelly",
    "iqn",
)
SEEDS = (42, 43, 44)
EXPECTED_BOOTSTRAP_N = 1_000
EXPECTED_BOOTSTRAP_CLUSTER = "environment_episode"
CONDITION_COLORS = {
    "a2c": "#8c8c8c",
    "ppo": "#4c78a8",
    "predictive_loss": "#f58518",
    "decoupled_kelly": "#54a24b",
    "iqn": "#b279a2",
}
SEED_COLORS = {
    42: "#4c78a8",
    43: "#f58518",
    44: "#54a24b",
}


def _curve_path(results_root: Path, condition: str, seed: int) -> Path:
    run_name = f"mess3_token_guess_cycle_2-{condition}-seed{seed}"
    return (
        results_root
        / condition
        / "results"
        / run_name
        / "checkpoint_probe_curve.json"
    )


def _validated_points(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_points = payload.get("checkpoints")
    if not isinstance(raw_points, list) or len(raw_points) < 2:
        raise ValueError("checkpoint curve must contain init and trained points")

    points: list[dict[str, Any]] = []
    previous_step = -1
    for index, raw in enumerate(raw_points):
        if not isinstance(raw, Mapping):
            raise ValueError(f"checkpoint {index} is not an object")
        probe = raw.get("probe")
        if not isinstance(probe, Mapping):
            raise ValueError(f"checkpoint {index} is missing probe metrics")

        step = int(raw["agent_steps"])
        mse = float(raw["mse"])
        ci_low = float(probe["mse_ci_95_low"])
        ci_high = float(probe["mse_ci_95_high"])
        if step <= previous_step:
            raise ValueError("checkpoint steps must be strictly increasing")
        if not ci_low <= mse <= ci_high:
            raise ValueError("bootstrap interval must contain checkpoint MSE")
        if int(probe.get("bootstrap_n", 0)) != EXPECTED_BOOTSTRAP_N:
            raise ValueError("checkpoint does not use 1,000 bootstrap resamples")
        if probe.get("bootstrap_cluster") != EXPECTED_BOOTSTRAP_CLUSTER:
            raise ValueError("checkpoint bootstrap is not episode-clustered")
        if not np.isclose(float(probe["mse"]), mse):
            raise ValueError("top-level and probe MSE disagree")

        points.append(
            {
                "checkpoint_index": index,
                "agent_steps": step,
                "training_iteration": raw.get("training_iteration"),
                "mse": mse,
                "mse_ci_95_low": ci_low,
                "mse_ci_95_high": ci_high,
                "bootstrap_n": EXPECTED_BOOTSTRAP_N,
                "bootstrap_cluster": EXPECTED_BOOTSTRAP_CLUSTER,
                "sampling_distribution": probe.get("sampling_distribution"),
                "representation": probe.get("representation"),
                "n_fit": probe.get("n_fit"),
                "n_test": probe.get("n_test"),
            }
        )
        previous_step = step

    if points[0]["agent_steps"] != 0:
        raise ValueError("first checkpoint must be true initialization (step zero)")
    return points


def load_mse_curves(
    results_root: Path,
    *,
    conditions: Sequence[str] = CONDITIONS,
    seeds: Sequence[int] = SEEDS,
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    """Load and validate every compact checkpoint-probe curve."""

    curves: dict[str, dict[int, list[dict[str, Any]]]] = {}
    expected_count: int | None = None
    for condition in conditions:
        condition_curves: dict[int, list[dict[str, Any]]] = {}
        for seed in seeds:
            path = _curve_path(results_root, condition, seed)
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"could not load {path}") from error
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path} is not a JSON object")
            points = _validated_points(payload)
            if expected_count is None:
                expected_count = len(points)
            elif len(points) != expected_count:
                raise ValueError("all runs must have the same checkpoint count")
            condition_curves[int(seed)] = points
        curves[condition] = condition_curves
    return curves


def _step_labels(points: Sequence[Mapping[str, Any]]) -> list[str]:
    labels = ["init"]
    labels.extend(
        f"{float(point['agent_steps']) / 1_000_000:.2f}"
        for point in points[1:]
    )
    return labels


def _asymmetric_ci(points: Sequence[Mapping[str, Any]]) -> np.ndarray:
    mse = np.asarray([float(point["mse"]) for point in points])
    low = np.asarray([float(point["mse_ci_95_low"]) for point in points])
    high = np.asarray([float(point["mse_ci_95_high"]) for point in points])
    return np.vstack((mse - low, high - mse))


def _style_axis(axis: Any, *, title: str, labels: Sequence[str]) -> None:
    axis.set_title(title)
    axis.set_xlabel("Environment steps (millions)")
    axis.set_ylabel("Held-out affine-probe MSE")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    axis.grid(axis="y", alpha=0.2, linewidth=0.7)
    axis.set_axisbelow(True)


def _plot_one_run(
    *,
    condition: str,
    seed: int,
    points: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    x = np.arange(len(points))
    mse = np.asarray([float(point["mse"]) for point in points])
    colors = ["#b5b5b5"] + [CONDITION_COLORS[condition]] * (len(points) - 1)
    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    axis.bar(
        x,
        mse,
        yerr=_asymmetric_ci(points),
        color=colors,
        edgecolor="#333333",
        linewidth=0.55,
        capsize=2.5,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )
    _style_axis(
        axis,
        title=f"{condition.replace('_', ' ')} · seed {seed}",
        labels=_step_labels(points),
    )
    axis.text(
        0.99,
        0.98,
        "95% episode-cluster bootstrap CI",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#444444",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _plot_all_runs(
    curves: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
    *,
    path: Path,
) -> None:
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(15.0, 11.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    width = 0.24
    reference = curves[CONDITIONS[0]][SEEDS[0]]
    x = np.arange(len(reference))
    labels = _step_labels(reference)
    for axis, condition in zip(axes.flat, CONDITIONS, strict=False):
        for seed_index, seed in enumerate(SEEDS):
            points = curves[condition][seed]
            mse = np.asarray([float(point["mse"]) for point in points])
            offset = (seed_index - 1) * width
            axis.bar(
                x + offset,
                mse,
                width=width,
                yerr=_asymmetric_ci(points),
                color=SEED_COLORS[seed],
                label=f"seed {seed}",
                capsize=1.2,
                error_kw={"elinewidth": 0.55, "capthick": 0.55},
            )
        _style_axis(
            axis,
            title=condition.replace("_", " "),
            labels=labels,
        )
        axis.set_xlabel("")
        axis.set_ylabel("")
    axes.flat[-1].axis("off")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower right",
        bbox_to_anchor=(0.96, 0.08),
        frameon=False,
    )
    figure.suptitle(
        "MESS3 token-guess cycle 2: MSE over training (all 15 runs)",
        fontsize=15,
    )
    figure.supxlabel("Environment steps (millions)")
    figure.supylabel("Held-out affine-probe MSE", x=0.01)
    figure.text(
        0.74,
        0.15,
        "Error bars: 95% episode-cluster bootstrap CI\n"
        "Held-out fit/test rollouts · post-final-LayerNorm",
        ha="center",
        va="center",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0.04, 0.0, 1.0, 0.96))
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _plot_condition_means(
    curves: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
    *,
    path: Path,
) -> None:
    """Plot model-seed means; error bars are seed SD, not bootstrap CIs."""

    reference = curves[CONDITIONS[0]][SEEDS[0]]
    x = np.arange(len(reference))
    labels = _step_labels(reference)
    width = 0.16
    figure, axis = plt.subplots(figsize=(14.0, 6.0))
    for condition_index, condition in enumerate(CONDITIONS):
        values = np.asarray(
            [
                [float(point["mse"]) for point in curves[condition][seed]]
                for seed in SEEDS
            ]
        )
        offset = (condition_index - (len(CONDITIONS) - 1) / 2) * width
        centers = x + offset
        axis.bar(
            centers,
            values.mean(axis=0),
            width=width,
            yerr=values.std(axis=0),
            color=CONDITION_COLORS[condition],
            label=condition.replace("_", " "),
            capsize=2,
            error_kw={"elinewidth": 0.7, "capthick": 0.7},
        )
        for row in values:
            axis.scatter(
                centers,
                row,
                s=8,
                color="#202020",
                alpha=0.55,
                zorder=3,
            )
    _style_axis(
        axis,
        title="",
        labels=labels,
    )
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        frameon=False,
    )
    axis.text(
        0.99,
        0.98,
        "Black points: individual model seeds",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#444444",
    )
    figure.suptitle(
        "MSE over training by condition (mean ± SD across 3 model seeds)",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 0.87, 0.95))
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _bootstrap_assessment() -> str:
    return """# Bootstrap assessment

The checkpoint results already contain the bootstrap calculation recommended by
`analysis/probes/README.md`: each MSE uses 1,000 percentile-bootstrap resamples
clustered by complete environment episode. The per-run charts use those 95%
intervals directly. The fitted probe remains fixed, so these intervals estimate
evaluation-rollout sampling uncertainty, not probe-fit uncertainty.

No additional bootstrap should be applied to individual timesteps or training
checkpoints: timesteps are correlated within episodes, and checkpoints are
repeated measurements of one trained model.

The combined condition chart keeps independently trained model-seed variability
separate. It shows all three seed values and mean ± population SD. A bootstrap
over only three model seeds would be coarse and potentially misleading, so it is
not used. If inferential condition comparisons become important, run more model
seeds and then use paired seed differences (the same seed set is shared by every
condition); a hierarchical seed-then-episode bootstrap would be appropriate only
with enough model seeds and retained per-episode probe errors.

The existing held-out permutation nulls answer a different question—whether the
activation/target association generalizes—and should not be used as MSE error
bars.
"""


def write_mse_bar_charts(
    curves: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Write 15 per-run charts, two combined charts, and compact source data."""

    output_dir.mkdir(parents=True, exist_ok=True)
    by_run_dir = output_dir / "by_run"
    by_run_dir.mkdir(parents=True, exist_ok=True)
    run_files: dict[str, str] = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            filename = f"{condition}_seed{seed}_mse_over_training.png"
            _plot_one_run(
                condition=condition,
                seed=seed,
                points=curves[condition][seed],
                path=by_run_dir / filename,
            )
            run_files[f"{condition}/seed{seed}"] = f"by_run/{filename}"

    all_runs_path = output_dir / "mse_over_training_all_runs.png"
    means_path = output_dir / "mse_over_training_condition_means.png"
    _plot_all_runs(curves, path=all_runs_path)
    _plot_condition_means(curves, path=means_path)

    summary = {
        "metric": "held_out_affine_probe_mse",
        "sampling_distribution": "process_weighted_rollout",
        "representation": "post_final_layer_norm",
        "per_checkpoint_uncertainty": {
            "method": "percentile_cluster_bootstrap",
            "cluster": EXPECTED_BOOTSTRAP_CLUSTER,
            "n_resamples": EXPECTED_BOOTSTRAP_N,
            "interval": 0.95,
            "probe_refit": False,
        },
        "model_seed_summary": {
            "seeds": list(SEEDS),
            "error_bar": "population_standard_deviation",
            "individual_values_shown": True,
            "bootstrap_used": False,
        },
        "run_charts": run_files,
        "combined_charts": {
            "all_runs": all_runs_path.name,
            "condition_means": means_path.name,
        },
        "curves": {
            condition: {
                str(seed): list(curves[condition][seed])
                for seed in SEEDS
            }
            for condition in CONDITIONS
        },
    }
    (output_dir / "mse_over_training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (output_dir / "bootstrap_assessment.md").write_text(_bootstrap_assessment())
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build MSE-over-training bars from compact result curves."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Path ending at experiments/mess3_token_guess_cycle_2",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    curves = load_mse_curves(args.results_root)
    write_mse_bar_charts(curves, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
