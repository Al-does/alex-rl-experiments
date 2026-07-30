"""Plot multi-seed probe MSE curves for one cycle-5 variant (includes init)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402


def load_curves(results_dir: Path, variant: int, seeds: list[int]) -> dict[int, list[dict]]:
    curves: dict[int, list[dict]] = {}
    for seed in seeds:
        run_dir = results_dir / f"mess3-rsa-c5-v{variant}-seed{seed}"
        curve_path = run_dir / "checkpoint_probe_curve.json"
        if not curve_path.is_file():
            raise FileNotFoundError(f"missing checkpoint curve: {curve_path}")
        payload = json.loads(curve_path.read_text())
        points = payload["checkpoints"]
        init = points[0]
        if init["agent_steps"] != 0:
            raise ValueError(f"{curve_path}: first checkpoint must be untrained init")
        if not init.get("probe", {}).get("is_untrained"):
            raise ValueError(f"{curve_path}: first checkpoint is not marked untrained")
        curves[seed] = points
    return curves


def _format_step_label(step: int) -> str:
    if step >= 1_000_000:
        return f"{step / 1_000_000:.2f}M".rstrip("0").rstrip(".") + "M"
    if step >= 1_000:
        return f"{round(step / 1_000)}k"
    return str(step)


def _reference_schedule(curves: dict[int, list[dict]]) -> list[int]:
    """Return one canonical step schedule shared across seeds."""

    seeds = sorted(curves)
    n_points = len(curves[seeds[0]])
    for seed in seeds:
        points = curves[seed]
        if len(points) != n_points:
            raise ValueError(
                f"seed {seed} has {len(points)} checkpoints, expected {n_points}"
            )
        if points[0]["agent_steps"] != 0:
            raise ValueError(f"seed {seed} is missing untrained init checkpoint")
    middle_seed = seeds[len(seeds) // 2]
    return [point["agent_steps"] for point in curves[middle_seed]]


def _plot_x_positions(schedule: list[int]) -> np.ndarray:
    positive_steps = [step for step in schedule if step > 0]
    init_x = positive_steps[0] / 4.0
    return np.asarray(
        [init_x if step == 0 else float(step) for step in schedule],
        dtype=np.float64,
    )


def plot_variant_mse_curves(
    curves: dict[int, list[dict]],
    *,
    title: str,
    output_stem: Path,
) -> tuple[Path, Path]:
    seeds = sorted(curves)
    schedule = _reference_schedule(curves)
    plot_x = _plot_x_positions(schedule)
    positive_steps = [step for step in schedule if step > 0]
    init_x = float(plot_x[0])
    tick_labels = [
        "init" if step == 0 else _format_step_label(step) for step in schedule
    ]

    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
    for color, seed in zip(colors, seeds, strict=True):
        points = curves[seed]
        mse = np.asarray([point["mse"] for point in points], dtype=np.float64)
        axis.plot(
            plot_x,
            mse,
            marker="o",
            linewidth=1.6,
            color=color,
            alpha=0.85,
            label=f"seed {seed}",
        )

    n_points = len(schedule)
    mean_mse = []
    std_mse = []
    for index in range(n_points):
        values = [curves[seed][index]["mse"] for seed in seeds]
        mean_mse.append(float(np.mean(values)))
        std_mse.append(float(np.std(values)))
    plot_ref_steps = plot_x
    mean_mse_arr = np.asarray(mean_mse)
    std_mse_arr = np.asarray(std_mse)
    axis.plot(
        plot_ref_steps,
        mean_mse_arr,
        color="black",
        linewidth=2.4,
        marker="s",
        markersize=5,
        label=f"mean ({len(seeds)} seeds)",
        zorder=5,
    )
    axis.fill_between(
        plot_ref_steps,
        mean_mse_arr - std_mse_arr,
        mean_mse_arr + std_mse_arr,
        color="black",
        alpha=0.12,
        label="±1 std",
    )

    axis.set_xscale("log")
    axis.set_xticks(plot_x, tick_labels)
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlim(init_x / 1.6, positive_steps[-1] * 1.12)
    axis.tick_params(axis="x", labelrotation=0)
    axis.set_xlabel("Environment steps (init = untrained checkpoint before training)")
    axis.set_ylabel("Held-out affine-probe MSE (lower is better)")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", fontsize=9, ncol=2)
    figure.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=200)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=int, choices=(1, 2, 3))
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Generate charts for variants 1, 2, and 3.",
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    variants = [1, 2, 3] if args.all_variants else [args.variant or 1]
    for variant in variants:
        results_dir = args.study_root / f"variant_{variant}" / "results"
        curves = load_curves(results_dir, variant, args.seeds)
        output_stem = (
            args.study_root
            / f"variant_{variant}"
            / "figures"
            / f"variant_{variant}_mse_curve_with_init"
        )
        png_path, pdf_path = plot_variant_mse_curves(
            curves,
            title=(
                f"Cycle 5 variant {variant} — belief probe MSE "
                "(includes untrained init)"
            ),
            output_stem=output_stem,
        )
        print(png_path)
        print(pdf_path)


if __name__ == "__main__":
    main()
