"""Plot reward-state-2 occupancy vs held-out belief-probe MSE for variant 3.

Reads every ``checkpoint_probe_curve.json`` under ``results/`` and renders one
training trajectory per seed: x = fraction of greedy steps spent in reward
state 2, y = held-out affine belief-probe MSE. Checkpoints are connected in
training order so each curve runs from initialization to the final checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT = RESULTS_DIR / "reward_occupancy_vs_mse.png"
SEED_COLORS = ("#355c9a", "#c45135", "#3a7d44", "#7a5195", "#b8860b")


def _format_steps(steps: int) -> str:
    if steps == 0:
        return "init"
    if steps >= 1_000_000:
        return f"{steps / 1_000_000:g}M"
    if steps >= 1_000:
        return f"{steps / 1_000:g}k"
    return str(steps)


def main() -> Path:
    runs = []
    for path in sorted(RESULTS_DIR.glob("*/checkpoint_probe_curve.json")):
        data = json.loads(path.read_text())
        run_id = path.parent.name
        seed = int(run_id.rsplit("seed", 1)[1])
        points = sorted(data["checkpoints"], key=lambda c: c["agent_steps"])
        runs.append((seed, points))
    if not runs:
        raise SystemExit(f"no checkpoint_probe_curve.json files under {RESULTS_DIR}")

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=160)
    legend_handles = []
    for index, (seed, points) in enumerate(sorted(runs)):
        color = SEED_COLORS[index % len(SEED_COLORS)]
        xs = [p["reward_state_2_fraction_greedy"] for p in points]
        ys = [p["mse"] for p in points]
        ax.plot(xs, ys, "-", color=color, alpha=0.35, linewidth=1.2, zorder=2)
        ax.plot(
            xs,
            ys,
            "o",
            color=color,
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            _format_steps(points[0]["agent_steps"]),
            (xs[0], ys[0]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7.5,
            color=color,
        )
        legend_handles.append(
            Line2D([], [], color=color, marker="o", linewidth=1.2, label=f"seed {seed}")
        )

    final_x = max(p["reward_state_2_fraction_greedy"] for _, pts in runs for p in pts[-1:])
    final_y = max(p["mse"] for _, pts in runs for p in pts[-1:])
    final_steps = max(pts[-1]["agent_steps"] for _, pts in runs)
    ax.annotate(
        f"final ≈{_format_steps(final_steps)}",
        (final_x, final_y),
        textcoords="offset points",
        xytext=(8, 10),
        fontsize=8,
        color="#333333",
    )

    ax.set_xlabel("Reward-state occupancy (greedy fraction in reward state 2)")
    ax.set_ylabel("Belief-probe MSE (held out)")
    ax.set_title("Variant 3: reward occupancy vs belief-probe MSE\n"
                 "mess3_reward_state_action_symmetry_cycle_4")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(main())
