"""Aggregate and plot one all-checkpoint scalar-probe campaign."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.seed_queue import (
    SEEDS,
    TARGET_VARIANTS,
    TRAJECTORY_SUFFIX,
)

PLOT_MAX_ENV_STEPS = 750_000
BOOTSTRAP_N = 10_000
BOOTSTRAP_CI = 0.95
BOOTSTRAP_SEED = 42
PLOT_VARIANTS: dict[str, tuple[int, ...]] = {
    "symmetric_b2": (2, 3),
    "antisymmetric_b0_minus_b1": (2, 3),
    "coarse_b2": (2,),
}
COMPARISON_TARGETS: dict[str, tuple[str, str]] = {
    "coarse_b2": ("symmetric_b2", "full belief P(state=2)"),
}
VARIANT_COLORS: dict[int, str] = {
    2: "#ff7f0e",
    3: "#2ca02c",
}
COMPARISON_COLOR = "#1f77b4"


def _run_id(cycle: int, target: str, variant: int, seed: int) -> str:
    return (
        f"mess3-rsa-c{cycle}-belief-trajectory-{TRAJECTORY_SUFFIX}-"
        f"{target.replace('_', '-')}-v{variant}-seed{seed}"
    )


def _load_run(
    root: Path,
    *,
    cycle: int,
    target: str,
    variant: int,
    seed: int,
) -> dict[str, Any]:
    path = (
        root
        / f"variant_{variant}"
        / "results"
        / _run_id(cycle, target, variant, seed)
        / "condition_summary.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"required trajectory run missing: {path}")
    payload = json.loads(path.read_text())
    if payload.get("requested_target") != target:
        raise ValueError(f"{path}: expected requested_target={target!r}")
    return payload


def _study_root(probes_root: Path) -> Path:
    return probes_root.parent


def _checkpoint_step_lookup(
    study_root: Path,
    *,
    cycle: int,
    variant: int,
    seed: int,
) -> dict[str, int]:
    curve_path = (
        study_root
        / f"variant_{variant}"
        / "results"
        / f"mess3-rsa-c{cycle}-v{variant}-seed{seed}"
        / "checkpoint_probe_curve.json"
    )
    if not curve_path.is_file():
        return {"initial": 0}
    payload = json.loads(curve_path.read_text())
    lookup: dict[str, int] = {"initial": 0}
    for point in payload["checkpoints"]:
        if int(point["agent_steps"]) == 0:
            continue
        checkpoint_name = point.get("checkpoint_name")
        if isinstance(checkpoint_name, str):
            lookup[checkpoint_name] = int(point["agent_steps"])
    return lookup


def _interpolate_env_steps(
    training_iteration: int,
    anchor_iterations: list[int],
    anchor_steps: dict[int, int],
) -> int:
    if training_iteration in anchor_steps:
        return anchor_steps[training_iteration]
    for left, right in zip(anchor_iterations, anchor_iterations[1:], strict=False):
        if left <= training_iteration <= right:
            left_steps = anchor_steps[left]
            right_steps = anchor_steps[right]
            if right == left:
                return left_steps
            fraction = (training_iteration - left) / (right - left)
            return int(round(left_steps + fraction * (right_steps - left_steps)))
    if training_iteration < anchor_iterations[0]:
        return 0
    left = anchor_iterations[-2]
    right = anchor_iterations[-1]
    left_steps = anchor_steps[left]
    right_steps = anchor_steps[right]
    if right == left:
        return right_steps
    fraction = (training_iteration - left) / (right - left)
    return int(round(left_steps + fraction * (right_steps - left_steps)))


def _env_step_schedule(
    labels: list[str],
    iterations: list[int],
    lookup: dict[str, int],
) -> list[int]:
    anchor_iterations = [0]
    anchor_steps = {0: 0}
    for label in labels:
        if label == "initial":
            continue
        iteration = iterations[labels.index(label)]
        if label in lookup:
            anchor_iterations.append(iteration)
            anchor_steps[iteration] = lookup[label]
    anchor_iterations = sorted(set(anchor_iterations))
    if len(anchor_iterations) == 1:
        final_iteration = max(iterations)
        final_steps = 33_000 * final_iteration
        return [
            0
            if iteration == 0
            else int(round(iteration / final_iteration * final_steps))
            for iteration in iterations
        ]
    return [
        _interpolate_env_steps(iteration, anchor_iterations, anchor_steps)
        for iteration in iterations
    ]


def _truncate_to_plot_window(
    labels: list[str],
    iterations: list[int],
    agent_steps: list[int],
    seed_curves: dict[str, list[float]],
) -> tuple[list[str], list[int], list[int], dict[str, list[float]]]:
    keep = [
        index
        for index, step in enumerate(agent_steps)
        if step <= PLOT_MAX_ENV_STEPS
    ]
    if not keep:
        raise ValueError("no checkpoints fall within the plot env-step window")
    return (
        [labels[index] for index in keep],
        [iterations[index] for index in keep],
        [agent_steps[index] for index in keep],
        {seed: [values[index] for index in keep] for seed, values in seed_curves.items()},
    )


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_resamples: int = BOOTSTRAP_N,
    ci: float = BOOTSTRAP_CI,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap CI for the seed-mean MSE at each checkpoint."""

    if values.ndim != 2:
        raise ValueError("expected seed-by-checkpoint array")
    n_seeds, n_checkpoints = values.shape
    if n_seeds < 2:
        raise ValueError("bootstrap CI requires at least two seeds")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_seeds, size=(n_resamples, n_seeds, n_checkpoints))
    col_idx = np.arange(n_checkpoints)[None, None, :]
    resampled = values[indices, col_idx]
    boot_means = resampled.mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    ci_low = np.percentile(boot_means, 100.0 * alpha, axis=0)
    ci_high = np.percentile(boot_means, 100.0 * (1.0 - alpha), axis=0)
    return values.mean(axis=0), ci_low, ci_high


def _aggregate_variant(
    root: Path,
    *,
    cycle: int,
    target: str,
    variant: int,
    study_root: Path,
) -> dict[str, Any]:
    runs = {
        seed: _load_run(
            root,
            cycle=cycle,
            target=target,
            variant=variant,
            seed=seed,
        )
        for seed in SEEDS
    }
    reference_schedule = runs[SEEDS[0]]["checkpoint_schedule"]
    labels = [point["label"] for point in reference_schedule]
    iterations = [point["training_iteration"] for point in reference_schedule]
    schedule_seed = SEEDS[len(SEEDS) // 2]
    agent_steps = _env_step_schedule(
        labels,
        iterations,
        _checkpoint_step_lookup(
            study_root,
            cycle=cycle,
            variant=variant,
            seed=schedule_seed,
        ),
    )
    seed_curves: dict[str, list[float]] = {}
    for seed, run in runs.items():
        if run["checkpoint_schedule"] != reference_schedule:
            raise ValueError(
                f"variant {variant} seed {seed} checkpoint schedule differs"
            )
        seed_curves[str(seed)] = [
            float(run["checkpoints"][label]["targets"][target]["mse"])
            for label in labels
        ]
    labels, iterations, agent_steps, seed_curves = _truncate_to_plot_window(
        labels,
        iterations,
        agent_steps,
        seed_curves,
    )
    values = np.asarray(list(seed_curves.values()), dtype=np.float64)
    mean, ci_low, ci_high = _bootstrap_mean_ci(values)
    return {
        "checkpoint_labels": labels,
        "training_iterations": iterations,
        "agent_steps": agent_steps,
        "seed_curves": seed_curves,
        "mean": mean.tolist(),
        "ci_95_low": ci_low.tolist(),
        "ci_95_high": ci_high.tolist(),
    }


def aggregate(root: Path, *, cycle: int, target: str) -> dict[str, Any]:
    if target not in TARGET_VARIANTS:
        raise ValueError(f"unknown trajectory target: {target}")
    study_root = _study_root(root)
    variants = {
        f"variant_{variant}": _aggregate_variant(
            root,
            cycle=cycle,
            target=target,
            variant=variant,
            study_root=study_root,
        )
        for variant in TARGET_VARIANTS[target]
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "cycle": cycle,
        "target": target,
        "target_definition": {
            "symmetric_b2": "exact full-filter P(state=2), the last belief element",
            "antisymmetric_b0_minus_b1": "exact full-filter P(state=0)-P(state=1)",
            "coarse_b2": (
                "separate two-state HMM filter over A={state0,state1}, B={state2}"
            ),
        }[target],
        "variants": variants,
        "seeds": list(SEEDS),
        "metric": "held-out affine probe MSE",
        "uncertainty_band": {
            "method": "bootstrap_across_seeds",
            "statistic": "mean",
            "n_resamples": BOOTSTRAP_N,
            "ci": BOOTSTRAP_CI,
            "seed": BOOTSTRAP_SEED,
        },
        "plot_max_env_steps": PLOT_MAX_ENV_STEPS,
        "plot_variants": list(PLOT_VARIANTS[target]),
        "checkpoint_scope": (
            "initialization and every saved training checkpoint through "
            f"{PLOT_MAX_ENV_STEPS:,} environment steps"
        ),
    }
    if target in COMPARISON_TARGETS:
        comparison_target, comparison_label = COMPARISON_TARGETS[target]
        comparison_variant = next(iter(PLOT_VARIANTS[target]))
        summary["comparison"] = {
            "target": comparison_target,
            "label": comparison_label,
            f"variant_{comparison_variant}": _aggregate_variant(
                root,
                cycle=cycle,
                target=comparison_target,
                variant=comparison_variant,
                study_root=study_root,
            ),
        }
    return summary


def _format_step_label(step: int) -> str:
    if step >= 1_000_000:
        return f"{step / 1_000_000:.2f}M".rstrip("0").rstrip(".") + "M"
    if step >= 1_000:
        return f"{round(step / 1_000)}k"
    return str(step)


def _linear_tick_indices(agent_steps: list[int], *, max_ticks: int = 8) -> list[int]:
    if len(agent_steps) <= max_ticks:
        return list(range(len(agent_steps)))
    indices = np.linspace(0, len(agent_steps) - 1, max_ticks, dtype=int)
    return sorted({0, len(agent_steps) - 1, *indices.tolist()})


def _plot_variant_curve(
    axis: plt.Axes,
    *,
    plot_x: np.ndarray,
    curve: dict[str, Any],
    color: Any,
    label: str,
    linestyle: str = "-",
    seed_alpha: float = 0.25,
) -> tuple[list[float], float]:
    all_mse: list[float] = []
    for values in curve["seed_curves"].values():
        values_arr = np.asarray(values, dtype=np.float64)
        all_mse.extend(values_arr.tolist())
        axis.plot(
            plot_x,
            values_arr,
            color=color,
            linewidth=0.9,
            alpha=seed_alpha,
            linestyle=linestyle,
            label=None,
        )
    mean = np.asarray(curve["mean"], dtype=np.float64)
    ci_low = np.asarray(curve["ci_95_low"], dtype=np.float64)
    ci_high = np.asarray(curve["ci_95_high"], dtype=np.float64)
    all_mse.extend(mean.tolist())
    axis.plot(
        plot_x,
        mean,
        color=color,
        linewidth=2.4,
        marker="o",
        linestyle=linestyle,
        label=label,
    )
    axis.fill_between(
        plot_x,
        ci_low,
        ci_high,
        color=color,
        alpha=0.13,
    )
    return all_mse, float(mean[0])


def _plotted_variants(summary: dict[str, Any]) -> dict[str, Any]:
    allowed = set(PLOT_VARIANTS[summary["target"]])
    selected = {
        name: curve
        for name, curve in summary["variants"].items()
        if int(name.split("_", 1)[1]) in allowed
    }
    return dict(
        sorted(
            selected.items(),
            key=lambda item: int(item[0].split("_", 1)[1]),
        )
    )


def _variant_color(variant_name: str) -> str:
    variant = int(variant_name.split("_", 1)[1])
    try:
        return VARIANT_COLORS[variant]
    except KeyError as exc:
        raise ValueError(f"no standard color configured for {variant_name}") from exc


def _plot(summary: dict[str, Any], output_stem: Path) -> None:
    plotted = _plotted_variants(summary)
    reference_curve = next(iter(plotted.values()))
    agent_steps = reference_curve["agent_steps"]
    plot_x = np.asarray(agent_steps, dtype=np.float64)
    tick_indices = _linear_tick_indices(agent_steps)
    tick_positions = plot_x[tick_indices]
    tick_labels = [
        "init" if agent_steps[index] == 0 else _format_step_label(agent_steps[index])
        for index in tick_indices
    ]

    comparison = summary.get("comparison")
    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    all_mse: list[float] = []
    init_mse: list[float] = []
    for variant_name, curve in plotted.items():
        series_mse, series_init = _plot_variant_curve(
            axis,
            plot_x=plot_x,
            curve=curve,
            color=_variant_color(variant_name),
            label=variant_name.replace("_", " "),
        )
        all_mse.extend(series_mse)
        init_mse.append(series_init)

    if comparison is not None:
        comparison_key = next(
            key for key in comparison if key.startswith("variant_")
        )
        comparison_curve = comparison[comparison_key]
        series_mse, series_init = _plot_variant_curve(
            axis,
            plot_x=plot_x,
            curve=comparison_curve,
            color=COMPARISON_COLOR,
            label=comparison["label"],
            linestyle="--",
            seed_alpha=0.18,
        )
        all_mse.extend(series_mse)
        init_mse.append(series_init)

    axis.set_xscale("linear")
    axis.set_yscale("log")
    axis.set_xticks(tick_positions, tick_labels)
    axis.set_xlim(-25_000, float(agent_steps[-1]) * 1.02)
    axis.tick_params(axis="x", labelrotation=0)
    axis.set_xlabel("Environment steps (0 = untrained initialization)")
    axis.set_ylabel("Held-out affine-probe MSE (lower is better)")
    axis.set_title(
        f"Cycle {summary['cycle']} — {summary['target'].replace('_', ' ')}"
    )

    positive_mse = [value for value in all_mse if value > 0]
    if positive_mse:
        ymin = min(positive_mse) * 0.75
        ymax = max(max(init_mse) * 4.0, max(positive_mse) * 1.05)
        axis.set_ylim(ymin, ymax)

    axis.grid(alpha=0.25, which="both")
    axis.legend()
    axis.text(
        0.02,
        0.02,
        (
            f"Shaded bands: bootstrapped 95% CI for mean MSE across seeds "
            f"(n={BOOTSTRAP_N:,})"
        ),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.35",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.85",
            "alpha": 0.92,
        },
    )
    figure.tight_layout()
    figure.savefig(output_stem.with_suffix(".png"), dpi=200)
    plt.close(figure)


def write_campaign(root: Path, *, cycle: int, target: str) -> Path:
    summary = aggregate(root, cycle=cycle, target=target)
    output = root / "results" / "trajectory_campaign" / target
    output.mkdir(parents=True, exist_ok=True)
    (output / "campaign_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    _plot(summary, output / "probe_trajectory")
    return output / "probe_trajectory.png"


def main(argv: list[str] | None = None, *, cycle: int = 4) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=tuple(TARGET_VARIANTS))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    output = write_campaign(args.root, cycle=cycle, target=args.target)
    print(output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
