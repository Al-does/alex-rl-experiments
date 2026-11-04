"""Render the complete A/B sensitivity curve without restoring any checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def curve_rows(report: dict) -> list[dict]:
    points = report["points"]
    expected = {"initial_checkpoint"} | {f"checkpoint_{i:06d}" for i in range(26)}
    names = [point["checkpoint"]["name"] for point in points]
    if len(points) != 27 or set(names) != expected:
        raise ValueError("plot requires initialization and every saved checkpoint exactly once")
    points = sorted(points, key=lambda point: point["checkpoint"]["agent_steps"])
    steps = [point["checkpoint"]["agent_steps"] for point in points]
    if steps[0] != 0 or any(a >= b for a, b in zip(steps, steps[1:])):
        raise ValueError("actual training steps must start at zero and increase strictly")
    rows = []
    for point in points:
        for mode in ("greedy", "sampled"):
            result = point["results"][mode]
            tv = result["total_variation"]
            low, high = tv["ci95"]
            if not 0 <= low <= tv["mean"] <= high <= 1:
                raise ValueError("invalid TV estimate or interval")
            rows.append({
                "checkpoint": point["checkpoint"]["name"],
                "training_iteration": point["checkpoint"]["training_iteration"],
                "agent_steps": point["checkpoint"]["agent_steps"],
                "mode": mode, "mean_TV": tv["mean"], "ci95_low": low, "ci95_high": high,
                "rms_logit_change": result["rms_logit_preference_change"],
                "greedy_disagreement_fraction": result["greedy_disagreements"] / result["n_decisions"],
                "baseline_entropy_nats": result["baseline_entropy_nats"],
                "baseline_mean_max_action_probability": result["baseline_mean_max_action_probability"],
                "n_decisions": result["n_decisions"],
            })
    return rows


def plot_curve(rows: list[dict], report: dict, path: Path, *, log_y: bool, preview: Path | None):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.22, top=0.87)
    for mode, label, color, style in (
        ("greedy", "Greedy trajectories", "#b65a34", "-"),
        ("sampled", "Sampled trajectories", "#216078", "--"),
    ):
        selected = [row for row in rows if row["mode"] == mode]
        x = np.array([row["agent_steps"] for row in selected]) / 1e6
        y = np.array([row["mean_TV"] for row in selected])
        low = np.array([row["ci95_low"] for row in selected])
        high = np.array([row["ci95_high"] for row in selected])
        if log_y and np.any(low <= 0):
            plt.close(fig)
            raise ValueError("log view requires positive intervals; linear view preserves zeros")
        ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, y, style, color=color, marker="o", markersize=3, linewidth=1.5, label=label)
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Mean total variation")
    ax.set_title("A/B sensitivity across training")
    ax.set_xlim(left=0)
    if log_y:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.legend(frameon=False)
    spec = report["protocol"]
    fig.text(
        0.14, 0.04,
        f"Seed 42; {spec['episodes']} episodes/checkpoint; pointwise 95% episode-bootstrap intervals.",
        fontsize=7,
    )
    fig.savefig(path)
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    rows = curve_rows(report)
    base = str(args.output_prefix)
    files = [Path(base + ext) for ext in (".csv", ".pdf", "_log.pdf", ".plot_manifest.json")]
    for path in files + ([] if args.preview is None else [args.preview]):
        if path.exists():
            parser.error(f"refusing to overwrite {path}")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with files[0].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_curve(rows, report, files[1], log_y=False, preview=args.preview)
    if all(row["ci95_low"] > 0 for row in rows):
        plot_curve(rows, report, files[2], log_y=True, preview=None)
    outputs = [path for path in files[:3] if path.exists()]
    manifest = {
        "input_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matplotlib_version": matplotlib.__version__,
        "physical_size_inches": [5.5, 3.4],
        "pointwise_intervals": True,
        "outputs": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in outputs
        },
    }
    files[3].write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
