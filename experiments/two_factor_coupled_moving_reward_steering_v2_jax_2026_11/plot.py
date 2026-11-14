"""Best-action maps for the XOR and moving rewards, and confirmation reward panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from experiments.two_factor_coupled_moving_reward_steering_v2_jax_2026_11.steering import (  # noqa: E402
    HORIZON,
    PERIOD,
    q_tables,
)
from experiments.two_factor_moved_reward_steering_v2_jax_2026_11.plot import (  # noqa: E402
    ACTION_NAMES,
    AGENT_MARKERS,
    COLORS,
    CORNERS,
    _simplex_axes,
)
from experiments.two_factor_moved_reward_steering_v2_jax_2026_11.steering import (  # noqa: E402
    ACTION_TABLE,
)

CMAP = ListedColormap(COLORS)


def _factor_2_region(ax, q: np.ndarray, factor_1: np.ndarray, resolution: int = 300) -> None:
    xs, ys = np.meshgrid(np.linspace(0, 1, resolution), np.linspace(0, CORNERS[2, 1], resolution))
    b2 = ys / CORNERS[2, 1]
    b1 = xs - 0.5 * b2
    b0 = 1 - b1 - b2
    inside = (b0 >= 0) & (b1 >= 0)
    second = np.stack((b0, b1, b2), -1)
    joint = (factor_1[:, None] * second[..., None, :]).reshape(*second.shape[:2], 9)
    actions = ACTION_TABLE[(joint @ q).argmax(-1)][..., 1]
    ax.imshow(np.where(inside, actions, np.nan), origin="lower", extent=(0, 1, 0, CORNERS[2, 1]),
              cmap=CMAP, vmin=-0.5, vmax=2.5, alpha=0.75, interpolation="nearest", rasterized=True)


def plot_maps(output: Path) -> None:
    fig = plt.figure(figsize=(7.2, 4.6))
    grid = fig.add_gridspec(3, 4, height_ratios=(1, 1, 0.55), hspace=0.55, wspace=0.3)
    xor = q_tables("xor", "discounted")[HORIZON // 2]
    for col, (factor_1, label) in enumerate(((np.eye(3)[0], "factor 1 in state 1"),
                                             (np.eye(3)[2], "factor 1 in state 3"),
                                             (np.full(3, 1 / 3), "factor 1 uniform"))):
        ax = fig.add_subplot(grid[0, col])
        _factor_2_region(ax, xor, factor_1)
        _simplex_axes(ax, f"XOR, {label}")
    ax = fig.add_subplot(grid[0, 3])
    ax.axis("off")
    ax.text(0, 0.5, "Factor-2 action of the\nlong-run XOR map\n(reward 1 when exactly\none factor is in state 3)",
            fontsize=7, va="center")
    moving = q_tables("moving", "discounted")
    factor_1 = np.eye(3)[2]
    for col, t in enumerate((PERIOD // 2, PERIOD - 20, PERIOD + PERIOD // 2, 2 * PERIOD - 5)):
        ax = fig.add_subplot(grid[1, col])
        _factor_2_region(ax, moving[t], factor_1)
        pays = "state 3" if (t // PERIOD) % 2 == 0 else "states 1/2"
        _simplex_axes(ax, f"Moving, t={t}\n(factor 2 pays {pays})")
    ax = fig.add_subplot(grid[2, :])
    known = ACTION_TABLE[moving[:, 6:9].argmax(-1)][..., 1].T  # factor 1 in state 3, factor 2 known
    ax.imshow(known, aspect="auto", cmap=CMAP, vmin=-0.5, vmax=2.5, interpolation="nearest",
              extent=(0, HORIZON, 2.5, -0.5), rasterized=True)
    for boundary in range(PERIOD, HORIZON, PERIOD):
        ax.axvline(boundary, color="black", lw=0.6, ls="--")
    ax.set_yticks(range(3), [f"state {i + 1}" for i in range(3)])
    ax.set_xlabel("Episode step (dashed: factor-2 reward switches)")
    ax.set_title("Moving reward: long-run factor-2 action for a known factor-2 state", fontsize=8)
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=n) for c, n in zip(COLORS, ACTION_NAMES)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02, dpi=300)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


ROWS = (
    ("Unsteered", lambda c: c["intact"]),
    ("Uniform random", lambda c: c["random"]),
    ("Factored steering, next-step map", lambda c: c["comparisons"]["myopic_marginal_block_4"]),
    ("Joint steering, next-step map", lambda c: c["comparisons"]["myopic_joint_block_4"]),
    ("Factored steering, long-run map", lambda c: c["comparisons"]["discounted_marginal_block_4"]),
    ("Joint steering, long-run map", lambda c: c["comparisons"]["discounted_joint_block_4"]),
    ("Exact belief, next-step map", lambda c: c["exact_belief_myopic_map"]),
    ("Exact belief, long-run map", lambda c: c["exact_belief_discounted_map"]),
)
PANELS = (
    ("xor", "XOR reward\n(both factors)", lambda e: e["total_reward"], 0.8),
    ("moving", "Moving:\nfactor 1 (state 3)", lambda e: e["component_reward"][0], 0.8),
    ("moving", "Moving: factor 2,\nphases paying state 3", lambda e: e["phase_component_reward"][0][1], 1),
    ("moving", "Moving: factor 2,\nphases paying states 1/2", lambda e: e["phase_component_reward"][1][1], 1),
)


def plot_rewards(reports: list[Path], output: Path) -> None:
    loaded = [json.loads(p.read_text()) for p in reports]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(7.4, 2.8), sharey=True, gridspec_kw={"wspace": 0.1})
    y = np.arange(len(ROWS))[::-1]
    for ax, (reward, title, field, xmax) in zip(axes, PANELS):
        values = np.array([[field(pick(r["rewards"][reward]["confirmation"])) for _, pick in ROWS]
                           for r in loaded])
        ax.barh(y, values.mean(0), height=0.6, color="#2ca02c", alpha=0.4)
        for report, row in zip(loaded, values):
            study, seed = report["source"]["study"], report["source"]["seed"]
            tag = study.removeprefix("reward_factors_persistent_nuisance_v2_jax_temp15_")
            ax.scatter(row, y + (seed - 42.5) * 0.3, s=10, marker=AGENT_MARKERS[tag],
                       facecolor="none" if seed == 43 else "black", edgecolor="black", lw=0.6,
                       label=f"{'no ε' if tag == 'ctx32_2026_11' else 'ε=0.05'}, seed {seed}")
        ax.set_title(title, fontsize=7)
        ax.set_xlabel("Mean reward per step", fontsize=7)
        ax.set_xlim(0, xmax)
        ax.set_xticks(np.linspace(0, xmax, 3))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, [name for name, _ in ROWS])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=4, fontsize=6,
               bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Joint Reward agents, XOR and moving rewards (block 4, 256 episodes per agent)",
                 fontsize=8, y=1.02)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_maps(args.output_dir / "v2_xor_moving_best_action_maps.pdf")
    reports = sorted(args.results.glob("*__reward_both__seed*.json"))
    if reports:
        plot_rewards(reports, args.output_dir / "v2_xor_moving_steering_reward_both.pdf")


if __name__ == "__main__":
    main()
