"""Render 1 - R^2 curves from committed transducer probe results.

Reads every ``<arm>/<variant>/results/<run-id>/transducer_probes/*.json``
written by ``probe_analysis`` and writes a log-log figure plus a compact
summary table under the study-level ``results/`` directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = Path(__file__).resolve().parent
INITIAL_PLOT_STEPS = 1e5


def load_points(study: Path = STUDY) -> dict[str, list[dict[str, object]]]:
    arms: dict[str, list[dict[str, object]]] = {}
    for path in sorted(study.glob("*/variant_*/results/*/transducer_probes/*.json")):
        metrics = json.loads(path.read_text())
        arm = f"{path.parts[-6]}.{path.parts[-5]}"
        arms.setdefault(arm, []).append(
            {
                "checkpoint_name": metrics["checkpoint_name"],
                "env_steps": int(metrics["env_steps"]),
                "r_squared": float(metrics["r_squared"]),
                "one_minus_r_squared": float(metrics["one_minus_r_squared"]),
                "n_fit": metrics["n_fit"],
                "n_test": metrics["n_test"],
                "checkpoint": metrics["checkpoint"],
            }
        )
    for points in arms.values():
        points.sort(key=lambda p: int(p["env_steps"]))
    return arms


def plot(arms: dict[str, list[dict[str, object]]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for arm, points in arms.items():
        x = [max(int(p["env_steps"]), INITIAL_PLOT_STEPS) for p in points]
        y = [float(p["one_minus_r_squared"]) for p in points]
        ax.plot(x, y, marker="o", label=arm)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        f"environment steps (initial checkpoint plotted at {INITIAL_PLOT_STEPS:.0e})"
    )
    ax.set_ylabel("1 - R^2 (held-out affine probe -> transducer belief)")
    ax.set_title("Pusher-B cycle 1: belief decodability vs training (seed 42)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=STUDY / "results")
    args = parser.parse_args()
    arms = load_points()
    if not arms:
        raise SystemExit("no transducer probe results found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot(arms, args.output_dir / "transducer_probe_one_minus_r2.png")
    summary = {
        "target": "controlled_transducer_belief_delay_one_edge_emitting",
        "probe": "held_out_affine_least_squares",
        "note": (
            "Affine decodability does not establish causal policy use; "
            "single training seed per arm."
        ),
        "arms": arms,
    }
    (args.output_dir / "transducer_probe_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    for arm, points in arms.items():
        final = points[-1]
        print(
            f"{arm:28s} final R2={final['r_squared']:.4f} "
            f"1-R2={final['one_minus_r_squared']:.3e} ({len(points)} checkpoints)"
        )


if __name__ == "__main__":
    main()
