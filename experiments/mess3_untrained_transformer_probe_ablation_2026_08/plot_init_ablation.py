"""Bar chart of untrained init-probe MSE by architecture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "results" / "init_architecture_ablation.json"
DEFAULT_OUTPUT = HERE / "results" / "init_ablation_mse_by_architecture.png"

DISPLAY_ORDER = (
    "small_baseline",
    "width96_small_style",
    "ablate_heads",
    "ablate_layers",
    "ablate_context",
    "large_full",
)
SHORT_LABELS = {
    "small_baseline": "small baseline\n64/4/1/10",
    "width96_small_style": "96-wide\nsmall-style",
    "ablate_heads": "more heads\n(4 vs 1)",
    "ablate_layers": "fewer layers\n(3 vs 4)",
    "ablate_context": "longer context\n(64 vs 10)",
    "large_full": "all interventions\n96/3/4/64",
}


def plot(source: Path, output: Path) -> None:
    payload = json.loads(source.read_text())
    summary = payload["summary"]
    baseline_mse = summary["small_baseline"]["mse_mean"]

    labels: list[str] = []
    means: list[float] = []
    sds: list[float] = []
    colors: list[str] = []
    for key in DISPLAY_ORDER:
        row = summary[key]
        mean = row["mse_mean"]
        labels.append(SHORT_LABELS[key])
        means.append(mean)
        sds.append(row["mse_sd"])
        if key == "small_baseline":
            colors.append("#4c72b0")
        elif mean < baseline_mse:
            colors.append("#c44e52")
        else:
            colors.append("#55a868")

    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    axis.bar(
        x,
        means,
        yerr=sds,
        capsize=4,
        color=colors,
        edgecolor="0.2",
        linewidth=0.6,
        error_kw={"elinewidth": 1.0, "ecolor": "0.25"},
    )
    axis.axhline(
        baseline_mse,
        color="0.35",
        linestyle="--",
        linewidth=1.0,
        label="small baseline mean",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(labels, fontsize=9)
    axis.set_ylabel("Held-out affine-probe MSE at init (lower = more spurious fit)")
    axis.set_title(
        "Untrained belief-probe MSE by transformer architecture\n"
        "5 model seeds, 60k/80k held-out rollout protocol"
    )
    for index, mean in enumerate(means):
        if index == 0:
            continue
        delta_pct = 100.0 * (mean - baseline_mse) / baseline_mse
        sign = "+" if delta_pct >= 0 else ""
        axis.text(
            index,
            mean + sds[index] + 0.00015,
            f"{sign}{delta_pct:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.25",
        )
    axis.legend(loc="upper left", frameon=False)
    axis.set_ylim(0.0, max(means) + max(sds) + 0.0012)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.source, args.output)


if __name__ == "__main__":
    main()
