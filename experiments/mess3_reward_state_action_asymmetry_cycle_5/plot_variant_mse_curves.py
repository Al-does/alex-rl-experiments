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


def plot_variant_mse_curves(
    curves: dict[int, list[dict]],
    *,
    title: str,
    output_stem: Path,
) -> tuple[Path, Path]:
    seeds = sorted(curves)
    all_steps = sorted({point["agent_steps"] for seed in seeds for point in curves[seed]})
    positive_steps = [step for step in all_steps if step > 0]
    init_x = positive_steps[0] / 4.0
    tick_positions = [init_x if step == 0 else float(step) for step in all_steps]
    tick_labels = [
        "init"
        if step == 0
        else (
            f"{step / 1_000_000:g}M"
            if step >= 1_000_000
            else f"{step / 1_000:g}k"
        )
        for step in all_steps
    ]

    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
    for color, seed in zip(colors, seeds, strict=True):
        points = curves[seed]
        steps = np.asarray([point["agent_steps"] for point in points], dtype=np.float64)
        mse = np.asarray([point["mse"] for point in points], dtype=np.float64)
        plot_steps = np.where(steps == 0.0, init_x, steps)
        axis.plot(
            plot_steps,
            mse,
            marker="o",
            linewidth=1.6,
            color=color,
            alpha=0.85,
            label=f"seed {seed}",
        )

    reference_seed = seeds[0]
    n_points = len(curves[reference_seed])
    mean_mse = []
    std_mse = []
    for index in range(n_points):
        values = [curves[seed][index]["mse"] for seed in seeds]
        mean_mse.append(float(np.mean(values)))
        std_mse.append(float(np.std(values)))
    ref_steps = np.asarray(
        [curves[reference_seed][index]["agent_steps"] for index in range(n_points)],
        dtype=np.float64,
    )
    plot_ref_steps = np.where(ref_steps == 0.0, init_x, ref_steps)
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
    axis.set_xticks(tick_positions, tick_labels)
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlim(init_x / 1.4, positive_steps[-1] * 1.15)
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
    parser.add_argument("--variant", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument(
        "--study-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    results_dir = args.study_root / f"variant_{args.variant}" / "results"
    curves = load_curves(results_dir, args.variant, args.seeds)
    output_stem = (
        args.study_root
        / f"variant_{args.variant}"
        / "figures"
        / f"variant_{args.variant}_mse_curve_with_init"
    )
    png_path, pdf_path = plot_variant_mse_curves(
        curves,
        title=(
            f"Cycle 5 variant {args.variant} — belief probe MSE "
            "(includes untrained init)"
        ),
        output_stem=output_stem,
    )
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
