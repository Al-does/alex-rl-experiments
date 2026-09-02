"""Plot multi-seed CEV95 trajectories and final actor CEV curves."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

STUDY_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = Path("/opt/cursor/artifacts/two_factor_reinforce_multiseed_cev")
DEFAULT_SEEDS = (42, 43, 44)
SEED_PATTERN = re.compile(r"seed(\d+)")

MSE_COLORS = {
    "joint_mixed_state": "#355c9a",
    "factor_1": "#c45135",
    "factor_2": "#2a9d8f",
}
CEV95_COLOR = "#7d3ac1"
CEV95_LABEL_COLOR = "#5a2d91"
SEED_COLORS = {42: "#4c78a8", 43: "#f58518", 44: "#54a24b"}
CONDITION_LABELS = {
    "reward_both": "reward both factors",
    "reward_factor_1": "reward factor 1 only",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _seed(run_dir: Path) -> int | None:
    match = SEED_PATTERN.search(run_dir.name)
    if match:
        return int(match.group(1))
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    runtime = manifest.get("runtime") or {}
    overrides = runtime.get("overrides") or {}
    if overrides.get("seed") is not None:
        return int(overrides["seed"])
    command = manifest.get("command") or []
    for index, token in enumerate(command[:-1]):
        if token == "--seed":
            return int(command[index + 1])
    return None


def _ended_at(run_dir: Path) -> str:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return ""
    return str(_read_json(manifest_path).get("ended_at") or "")


def latest_runs(
    results_root: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[int, Path]:
    """Select the latest completed run for each seed."""
    selected: dict[int, Path] = {}
    timestamps: dict[int, str] = {}
    for run_dir in results_root.iterdir():
        if not run_dir.is_dir():
            continue
        seed = _seed(run_dir)
        if seed not in seeds:
            continue
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "completed":
            continue
        ended_at = str(manifest.get("ended_at") or "")
        if seed not in selected or ended_at > timestamps[seed]:
            selected[seed] = run_dir
            timestamps[seed] = ended_at
    missing = sorted(set(seeds) - set(selected))
    if missing:
        raise FileNotFoundError(
            f"missing completed runs for seeds {missing} under {results_root}"
        )
    return selected


def _reports(run_dir: Path) -> list[dict]:
    reports = [
        _read_json(path)
        for path in run_dir.glob("checkpoint_probes/steps_*/probe_battery.json")
    ]
    if not reports:
        raise FileNotFoundError(f"no probe reports under {run_dir}")
    return sorted(reports, key=lambda report: int(report["agent_steps"]))


def _common_schedule(curves: dict[int, list[dict]]) -> np.ndarray:
    """Use the middle seed's schedule over the range shared by every seed."""
    seeds = sorted(curves)
    reference = curves[seeds[len(seeds) // 2]]
    shared_final = min(int(curve[-1]["agent_steps"]) for curve in curves.values())
    schedule = [
        int(report["agent_steps"])
        for report in reference
        if int(report["agent_steps"]) <= shared_final
    ]
    if not schedule or schedule[0] != 0:
        raise ValueError("expected a shared schedule beginning at initialization")
    return np.asarray(schedule, dtype=np.float64)


def _interpolate(
    reports: list[dict],
    schedule: np.ndarray,
    value,
) -> np.ndarray:
    steps = np.asarray(
        [int(report["agent_steps"]) for report in reports],
        dtype=np.float64,
    )
    values = np.asarray([float(value(report)) for report in reports], dtype=np.float64)
    return np.interp(schedule, steps, values)


def _mean_trajectories(
    curves: dict[int, list[dict]],
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    schedule = _common_schedule(curves)
    mse: dict[str, list[np.ndarray]] = {
        "joint_mixed_state": [],
        "factor_1": [],
        "factor_2": [],
    }
    cev95: list[np.ndarray] = []
    for reports in curves.values():
        for key in mse:
            mse[key].append(
                _interpolate(
                    reports,
                    schedule,
                    lambda report, key=key: report["probe_fits"][key]["mse"],
                )
            )
        cev95.append(
            _interpolate(
                reports,
                schedule,
                lambda report: report["cev"]["actor_activation"]["cev95_dimension"],
            )
        )
    return (
        schedule,
        {key: np.mean(np.stack(values), axis=0) for key, values in mse.items()},
        np.mean(np.stack(cev95), axis=0),
    )


def _plot_trajectory(
    *,
    condition: str,
    curves: dict[int, list[dict]],
    output_dir: Path,
) -> tuple[Path, dict]:
    steps, mse, cev95 = _mean_trajectories(curves)
    figure, axis = plt.subplots(figsize=(9.2, 5.0))
    lines = []
    labels = {
        "joint_mixed_state": "joint probe MSE",
        "factor_1": "factor 1 probe MSE",
        "factor_2": "factor 2 probe MSE",
    }
    for key, values in mse.items():
        lines.append(
            axis.plot(
                steps,
                values,
                color=MSE_COLORS[key],
                linewidth=2.0,
                marker="o",
                markersize=4.0,
                label=labels[key],
            )[0]
        )
    axis.set_yscale("log")
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Held-out linear-probe MSE (log scale)")
    axis.grid(alpha=0.25)
    axis.xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: "0" if value == 0 else f"{value / 1_000_000:g}M"
        )
    )
    axis.set_xlim(-float(steps[-1]) * 0.02, float(steps[-1]) * 1.02)

    dimension_axis = axis.twinx()
    dimension_line = dimension_axis.plot(
        steps,
        cev95,
        color=CEV95_COLOR,
        linewidth=2.0,
        marker="s",
        markersize=4.0,
        label="actor dimensions for 95% CEV",
        zorder=4,
    )[0]
    dimension_axis.set_ylabel(
        "Actor dimensions for 95% CEV",
        color=CEV95_LABEL_COLOR,
    )
    dimension_axis.tick_params(axis="y", colors=CEV95_LABEL_COLOR)
    dimension_axis.spines["right"].set_color(CEV95_COLOR)
    dimension_axis.yaxis.set_major_locator(MaxNLocator(integer=True))

    seeds = sorted(curves)
    axis.set_title(
        f"Cycle 4 REINFORCE — {CONDITION_LABELS[condition]}\n"
        f"mean over seeds {', '.join(map(str, seeds))}; shared checkpoint range"
    )
    figure.legend(
        handles=[*lines, dimension_line],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        fontsize=9,
        frameon=True,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"cycle4_{condition}_mse_cev95_mean.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path, {
        "steps": [int(step) for step in steps],
        "mse_mean": {key: values.tolist() for key, values in mse.items()},
        "actor_cev95_dimension_mean": cev95.tolist(),
    }


def _plot_final_cev(
    *,
    condition: str,
    curves: dict[int, list[dict]],
    output_dir: Path,
) -> tuple[Path, dict]:
    seed_curves: dict[int, np.ndarray] = {}
    seed_dimensions: dict[int, int] = {}
    seed_steps: dict[int, int] = {}
    for seed, reports in curves.items():
        final = reports[-1]
        actor = final["cev"]["actor_activation"]
        seed_curves[seed] = np.asarray(
            actor["cumulative_explained_variance"],
            dtype=np.float64,
        )
        seed_dimensions[seed] = int(actor["cev95_dimension"])
        seed_steps[seed] = int(final["agent_steps"])
    ranks = {values.size for values in seed_curves.values()}
    if len(ranks) != 1:
        raise ValueError(f"actor CEV curves have inconsistent ranks: {sorted(ranks)}")
    rank = ranks.pop()
    dimensions = np.arange(1, rank + 1)
    mean_curve = np.mean(np.stack(list(seed_curves.values())), axis=0)
    mean_dimension = float(np.mean(list(seed_dimensions.values())))
    mean_curve_dimension = int(np.searchsorted(mean_curve, 0.95) + 1)

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    for seed, values in sorted(seed_curves.items()):
        axis.plot(
            dimensions,
            values,
            color=SEED_COLORS.get(seed, "#888888"),
            linewidth=1.2,
            alpha=0.45,
            label=f"seed {seed}",
        )
    mean_line = axis.plot(
        dimensions,
        mean_curve,
        color="#172b4d",
        linewidth=2.8,
        label="mean CEV",
    )[0]
    threshold_line = axis.axhline(
        0.95,
        color=CEV95_COLOR,
        linestyle="--",
        linewidth=1.5,
        label="95% threshold",
    )
    axis.axvline(
        mean_curve_dimension,
        color=CEV95_COLOR,
        linestyle=":",
        linewidth=1.5,
        label=f"mean-curve CEV95 dimension = {mean_curve_dimension}",
    )
    axis.set_xlim(1, rank)
    axis.set_ylim(0.0, 1.01)
    axis.set_xlabel("Actor representation dimension count")
    axis.set_ylabel("Cumulative explained variance")
    axis.grid(alpha=0.25)
    axis.set_title(
        f"Cycle 4 REINFORCE — {CONDITION_LABELS[condition]}\n"
        "mean CEV at latest checkpoint per seed"
    )
    handles, labels = axis.get_legend_handles_labels()
    ordered_handles = [mean_line, threshold_line, *handles[-1:], *handles[:3]]
    ordered_labels = [
        "mean CEV",
        "95% threshold",
        f"mean-curve CEV95 dimension = {mean_curve_dimension}",
        *[f"seed {seed}" for seed in sorted(seed_curves)],
    ]
    axis.legend(
        ordered_handles,
        ordered_labels,
        loc="lower right",
        fontsize=9,
        frameon=True,
    )
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"cycle4_{condition}_latest_actor_cev_curve.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path, {
        "dimension_count": dimensions.tolist(),
        "mean_cumulative_explained_variance": mean_curve.tolist(),
        "seed_cumulative_explained_variance": {
            str(seed): values.tolist() for seed, values in sorted(seed_curves.items())
        },
        "seed_cev95_dimensions": {
            str(seed): value for seed, value in sorted(seed_dimensions.items())
        },
        "mean_cev95_dimension": mean_dimension,
        "mean_curve_cev95_dimension": mean_curve_dimension,
        "latest_agent_steps": {
            str(seed): value for seed, value in sorted(seed_steps.items())
        },
    }


def generate(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    summary: dict[str, dict] = {}
    for condition in CONDITION_LABELS:
        results_root = STUDY_ROOT / condition / "results"
        runs = latest_runs(results_root, seeds)
        curves = {seed: _reports(run_dir) for seed, run_dir in runs.items()}
        trajectory_path, trajectory = _plot_trajectory(
            condition=condition,
            curves=curves,
            output_dir=output_dir,
        )
        cev_path, final_cev = _plot_final_cev(
            condition=condition,
            curves=curves,
            output_dir=output_dir,
        )
        generated.extend([trajectory_path, cev_path])
        summary[condition] = {
            "runs": {str(seed): run_dir.name for seed, run_dir in sorted(runs.items())},
            "trajectory": trajectory,
            "latest_checkpoint_cev": final_cev,
        }
    summary_path = output_dir / "cycle4_two_factor_cev_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    generated.append(summary_path)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()
    for path in generate(args.output_dir, tuple(args.seeds)):
        print(path)


if __name__ == "__main__":
    main()
