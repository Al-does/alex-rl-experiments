"""Aggregate and plot one all-checkpoint scalar-probe campaign."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.seed_queue import (
    SEEDS,
    TARGET_VARIANTS,
    TRAJECTORY_SUFFIX,
)

PLOT_MAX_ENV_STEPS = 750_000


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


def aggregate(root: Path, *, cycle: int, target: str) -> dict[str, Any]:
    if target not in TARGET_VARIANTS:
        raise ValueError(f"unknown trajectory target: {target}")
    study_root = _study_root(root)
    variants: dict[str, Any] = {}
    for variant in TARGET_VARIANTS[target]:
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
        variants[f"variant_{variant}"] = {
            "checkpoint_labels": labels,
            "training_iterations": iterations,
            "agent_steps": agent_steps,
            "seed_curves": seed_curves,
            "mean": values.mean(axis=0).tolist(),
            "stdev": values.std(axis=0, ddof=1).tolist(),
        }
    return {
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
        "plot_max_env_steps": PLOT_MAX_ENV_STEPS,
        "checkpoint_scope": (
            "initialization and every saved training checkpoint through "
            f"{PLOT_MAX_ENV_STEPS:,} environment steps"
        ),
    }


def _format_step_label(step: int) -> str:
    if step >= 1_000_000:
        return f"{step / 1_000_000:.2f}M".rstrip("0").rstrip(".") + "M"
    if step >= 1_000:
        return f"{round(step / 1_000)}k"
    return str(step)


def _plot_x_positions(agent_steps: list[int]) -> np.ndarray:
    positive_steps = [step for step in agent_steps if step > 0]
    init_x = positive_steps[0] / 4.0
    return np.asarray(
        [init_x if step == 0 else float(step) for step in agent_steps],
        dtype=np.float64,
    )


def _plot(summary: dict[str, Any], output_stem: Path) -> None:
    reference_curve = next(iter(summary["variants"].values()))
    agent_steps = reference_curve["agent_steps"]
    plot_x = _plot_x_positions(agent_steps)
    positive_steps = [step for step in agent_steps if step > 0]
    init_x = float(plot_x[0])
    tick_labels = [
        "init" if step == 0 else _format_step_label(step) for step in agent_steps
    ]

    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(summary["variants"])))
    all_mse: list[float] = []
    init_mse: list[float] = []
    for color, (variant_name, curve) in zip(
        colors, summary["variants"].items(), strict=True
    ):
        for seed, values in curve["seed_curves"].items():
            values_arr = np.asarray(values, dtype=np.float64)
            all_mse.extend(values_arr.tolist())
            axis.plot(
                plot_x,
                values_arr,
                color=color,
                linewidth=0.9,
                alpha=0.25,
                label=None,
            )
        mean = np.asarray(curve["mean"], dtype=np.float64)
        stdev = np.asarray(curve["stdev"], dtype=np.float64)
        init_mse.append(float(mean[0]))
        all_mse.extend(mean.tolist())
        label = variant_name.replace("_", " ")
        axis.plot(
            plot_x,
            mean,
            color=color,
            linewidth=2.4,
            marker="o",
            label=label,
        )
        axis.fill_between(
            plot_x,
            mean - stdev,
            mean + stdev,
            color=color,
            alpha=0.13,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(plot_x, tick_labels)
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlim(init_x / 1.6, positive_steps[-1] * 1.12)
    axis.tick_params(axis="x", labelrotation=0)
    axis.set_xlabel("Environment steps (init = untrained checkpoint before training)")
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
