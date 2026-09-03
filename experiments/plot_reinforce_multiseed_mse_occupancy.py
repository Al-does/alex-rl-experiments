"""Generate tracked multi-seed MSE and occupancy plots for cycles 4 and 6."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

EXPERIMENTS_ROOT = Path(__file__).resolve().parent
CYCLE4_ROOT = EXPERIMENTS_ROOT / "two_factor_reward_state_REINFORCE_cycle_4"
CYCLE6_ROOT = EXPERIMENTS_ROOT / "mess3_reward_state_action_symmetry_cycle_6"
SEEDS = (42, 43, 44)
SEED_PATTERN = re.compile(r"seed(\d+)")

MSE_COLORS = {
    "joint_mixed_state": "#355c9a",
    "factor_1": "#c45135",
    "factor_2": "#2a9d8f",
    "belief": "#355c9a",
}
OCCUPANCY_COLOR = "#7d3ac1"
OCCUPANCY_LABEL_COLOR = "#5a2d91"


def _read_json(path: Path) -> dict[str, Any]:
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
    if runtime.get("seed") is not None:
        return int(runtime["seed"])
    command = manifest.get("command") or []
    for index, token in enumerate(command[:-1]):
        if token == "--seed":
            return int(command[index + 1])
    return None


def _latest_runs(results_root: Path) -> dict[int, Path]:
    selected: dict[int, Path] = {}
    timestamps: dict[int, str] = {}
    for run_dir in results_root.iterdir():
        if not run_dir.is_dir():
            continue
        seed = _seed(run_dir)
        if seed not in SEEDS:
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
    missing = sorted(set(SEEDS) - set(selected))
    if missing:
        raise FileNotFoundError(
            f"missing completed runs for seeds {missing} under {results_root}"
        )
    return selected


def _common_schedule(curves: dict[int, list[dict[str, Any]]]) -> np.ndarray:
    """Use seed 43's checkpoints over the range shared by all three seeds."""
    shared_final = min(int(points[-1]["agent_steps"]) for points in curves.values())
    schedule = [
        int(point["agent_steps"])
        for point in curves[43]
        if int(point["agent_steps"]) <= shared_final
    ]
    return np.asarray(schedule, dtype=np.float64)


def _mean_on_schedule(
    curves: dict[int, list[dict[str, Any]]],
    schedule: np.ndarray,
    value: Callable[[dict[str, Any]], float],
) -> np.ndarray:
    interpolated = []
    for points in curves.values():
        steps = np.asarray(
            [int(point["agent_steps"]) for point in points],
            dtype=np.float64,
        )
        values = np.asarray([float(value(point)) for point in points])
        interpolated.append(np.interp(schedule, steps, values))
    return np.mean(np.stack(interpolated), axis=0)


def _limits(values: np.ndarray, *, log: bool) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    lo = float(finite.min())
    hi = float(finite.max())
    if log:
        span = np.log10(hi) - np.log10(lo)
        pad = max(span * 0.08, 0.02)
        return 10 ** (np.log10(lo) - pad), 10 ** (np.log10(hi) + pad)
    span = hi - lo
    pad = span * 0.06 if span else max(abs(lo), 0.01) * 0.06
    return lo - pad, hi + pad


def _save_plot(
    *,
    steps: np.ndarray,
    mse: dict[str, np.ndarray],
    occupancy: np.ndarray,
    title: str,
    subtitle: str,
    output_stem: Path,
    runs: dict[int, Path],
) -> Path:
    figure, axis = plt.subplots(figsize=(9.2, 5.0))
    labels = {
        "joint_mixed_state": "joint probe MSE",
        "factor_1": "factor 1 probe MSE",
        "factor_2": "factor 2 probe MSE",
        "belief": "belief probe MSE",
    }
    mse_lines = [
        axis.plot(
            steps,
            values,
            color=MSE_COLORS[key],
            linewidth=2.0,
            marker="o",
            markersize=4.0,
            label=labels[key],
        )[0]
        for key, values in mse.items()
    ]
    all_mse = np.concatenate(list(mse.values()))
    axis.set_yscale("log")
    axis.set_ylim(*_limits(all_mse, log=True))
    axis.set_xlim(-float(steps[-1]) * 0.02, float(steps[-1]) * 1.02)
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Held-out linear-probe MSE (log scale)")
    axis.xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: "0" if value == 0 else f"{value / 1_000_000:g}M"
        )
    )
    axis.grid(alpha=0.25)

    occupancy_axis = axis.twinx()
    occupancy_line = occupancy_axis.plot(
        steps,
        occupancy,
        color=OCCUPANCY_COLOR,
        linewidth=2.0,
        marker="s",
        markersize=4.0,
        label="reward occupancy",
        zorder=4,
    )[0]
    occupancy_axis.set_ylim(*_limits(occupancy, log=False))
    occupancy_axis.set_ylabel(
        "Reward occupancy (state 2)",
        color=OCCUPANCY_LABEL_COLOR,
    )
    occupancy_axis.tick_params(axis="y", colors=OCCUPANCY_LABEL_COLOR)
    occupancy_axis.spines["right"].set_color(OCCUPANCY_COLOR)
    occupancy_axis.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))

    axis.set_title(f"{title}\n{subtitle}")
    figure.legend(
        handles=[*mse_lines, occupancy_line],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(mse_lines) + 1,
        fontsize=9,
        frameon=True,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    output_stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "runs": {
                    str(seed): run_dir.name for seed, run_dir in sorted(runs.items())
                },
                "steps": steps.astype(int).tolist(),
                "mse_mean": {key: values.tolist() for key, values in mse.items()},
                "occupancy_mean": occupancy.tolist(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return png_path


def _cycle4(condition: str) -> Path:
    runs = _latest_runs(CYCLE4_ROOT / condition / "results")
    curves = {
        seed: sorted(
            (
                _read_json(path)
                for path in run_dir.glob(
                    "checkpoint_probes/steps_*/probe_battery.json"
                )
            ),
            key=lambda point: int(point["agent_steps"]),
        )
        for seed, run_dir in runs.items()
    }
    steps = _common_schedule(curves)
    mse = {
        key: _mean_on_schedule(
            curves,
            steps,
            lambda point, key=key: point["probe_fits"][key]["mse"],
        )
        for key in ("joint_mixed_state", "factor_1", "factor_2")
    }

    def occupancy(point: dict[str, Any]) -> float:
        policy = point["policy"]
        if condition == "reward_both":
            return 0.5 * (
                float(policy["factor_1_state_2_fraction"])
                + float(policy["factor_2_state_2_fraction"])
            )
        return float(policy["factor_1_state_2_fraction"])

    return _save_plot(
        steps=steps,
        mse=mse,
        occupancy=_mean_on_schedule(curves, steps, occupancy),
        title=(
            "Cycle 4 REINFORCE — reward both factors"
            if condition == "reward_both"
            else "Cycle 4 REINFORCE — reward factor 1 only"
        ),
        subtitle="mean over seeds 42, 43, 44; latest run, shared checkpoint range",
        output_stem=CYCLE4_ROOT / "figures" / f"{condition}_mse_occupancy_mean",
        runs=runs,
    )


def _cycle6(variant: int) -> Path:
    runs = _latest_runs(CYCLE6_ROOT / "battery" / "results")
    curves = {
        seed: _read_json(
            run_dir / f"variant_{variant}" / "checkpoint_probe_curve.json"
        )["checkpoints"]
        for seed, run_dir in runs.items()
    }
    steps = _common_schedule(curves)
    return _save_plot(
        steps=steps,
        mse={
            "belief": _mean_on_schedule(
                curves,
                steps,
                lambda point: point["mse"],
            )
        },
        occupancy=_mean_on_schedule(
            curves,
            steps,
            lambda point: point["reward_state_2_fraction_greedy"],
        ),
        title=f"Cycle 6 REINFORCE action symmetry — variant {variant}",
        subtitle="mean over seeds 42, 43, 44; latest run per seed",
        output_stem=(
            CYCLE6_ROOT / "figures" / f"variant_{variant}_mse_occupancy_mean"
        ),
        runs=runs,
    )


def main() -> None:
    for condition in ("reward_both", "reward_factor_1"):
        print(_cycle4(condition))
    for variant in (1, 2, 3):
        print(_cycle6(variant))


if __name__ == "__main__":
    main()
