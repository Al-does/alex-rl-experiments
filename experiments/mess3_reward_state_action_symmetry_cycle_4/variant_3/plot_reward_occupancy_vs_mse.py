"""Plot reward-state-2 occupancy and belief-probe MSE vs training steps.

Reads every ``checkpoint_probe_curve.json`` under ``results/`` and renders two
stacked panels sharing a log-scaled agent-steps x-axis: the top panel shows the
fraction of greedy steps spent in reward state 2, the bottom the held-out
affine belief-probe MSE. The untrained initialization checkpoint is shown at a
synthetic ``init`` x position to the left of the first trained checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullFormatter, NullLocator  # noqa: E402

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

    checkpoint_steps = sorted({p["agent_steps"] for _, pts in runs for p in pts})
    positive_steps = [s for s in checkpoint_steps if s > 0]
    init_x = positive_steps[0] / 4.0 if positive_steps else 1.0

    # Seeds record slightly different step counts; cluster values within ~2%
    # so each nominal checkpoint gets one tick position.
    clusters: list[list[int]] = []
    for step in checkpoint_steps:
        if clusters and step <= clusters[-1][-1] * 1.02:
            clusters[-1].append(step)
        else:
            clusters.append([step])
    tick_x = [init_x if c[0] == 0 else float(c[-1]) for c in clusters]
    tick_label = [_format_steps(0 if c[0] == 0 else c[-1]) for c in clusters]
    step_to_x = {s: x for c, x in zip(clusters, tick_x) for s in c}

    def x_of(steps: int) -> float:
        return step_to_x[steps]

    fig, (ax_mse, ax_occ) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        dpi=160,
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    legend_handles = []
    for index, (seed, points) in enumerate(sorted(runs)):
        color = SEED_COLORS[index % len(SEED_COLORS)]
        xs = [x_of(p["agent_steps"]) for p in points]
        ax_occ.plot(
            xs,
            [p["reward_state_2_fraction_greedy"] for p in points],
            "-o",
            color=color,
            alpha=0.85,
            linewidth=1.4,
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
        ax_mse.plot(
            xs,
            [p["mse"] for p in points],
            "-o",
            color=color,
            alpha=0.85,
            linewidth=1.4,
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
        legend_handles.append(
            Line2D([], [], color=color, marker="o", linewidth=1.4, label=f"seed {seed}")
        )

    ax_occ.set_ylabel("Reward-state occupancy\n(greedy fraction in state 2)")
    ax_mse.set_title(
        "Variant 3: belief-probe MSE and reward occupancy vs training steps\n"
        "mess3_reward_state_action_symmetry_cycle_4"
    )
    ax_mse.set_ylabel("Belief-probe MSE (held out)")
    ax_occ.set_xlabel("Agent steps (log scale; init shown before first checkpoint)")
    ax_occ.set_xscale("log")
    ax_occ.set_xticks(tick_x)
    ax_occ.set_xticklabels(tick_label)
    ax_occ.xaxis.set_minor_locator(NullLocator())
    ax_occ.xaxis.set_minor_formatter(NullFormatter())
    ax_occ.tick_params(axis="x", rotation=30)
    for ax in (ax_occ, ax_mse):
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.margins(x=0.03)
    ax_occ.legend(
        handles=legend_handles, loc="upper left", fontsize=8.5, framealpha=0.9
    )
    fig.savefig(OUTPUT, bbox_inches="tight")
    return OUTPUT


if __name__ == "__main__":
    print(main())
