"""Best-action maps on the factor simplex and confirmation reward bars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from experiments.two_factor_moved_reward_steering_2026_11.action_map import (  # noqa: E402
    FACTOR_ACTIONS,
    FACTOR_ACTION_NAMES,
    best_factor_action,
)
from experiments.two_factor_moved_reward_steering_2026_11.steering import (  # noqa: E402
    ACTION_TABLE,
)

COLORS = {0: "#9e9e9e", 1: "#1f77b4", 2: "#d62728"}
CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42,
})


def to_xy(belief: np.ndarray) -> np.ndarray:
    return np.asarray(belief) @ CORNERS


def _simplex_axes(ax, title: str) -> None:
    ax.plot(*np.vstack((CORNERS, CORNERS[:1])).T, color="black", lw=0.6)
    for (x, y), label, dx, dy in zip(CORNERS, ("1", "2", "3"),
                                     (0.0, 0.0, 0.0), (-0.07, -0.07, 0.06)):
        ax.text(x + dx, y + dy, f"state {label}", ha="center", va="center", fontsize=7)
    ax.set_title(title, fontsize=8, pad=12)
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, CORNERS[2, 1] + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")


def _region(ax, reward_state: int, resolution: int = 400) -> None:
    xs, ys = np.meshgrid(np.linspace(0, 1, resolution), np.linspace(0, CORNERS[2, 1], resolution))
    b2 = ys / CORNERS[2, 1]
    b1 = xs - 0.5 * b2
    b0 = 1 - b1 - b2
    inside = (b0 >= 0) & (b1 >= 0)
    belief = np.stack((b0, b1, b2), axis=-1)
    actions = best_factor_action(belief.reshape(-1, 3), reward_state).reshape(xs.shape)
    image = np.where(inside, actions, np.nan).astype(float)
    cmap = ListedColormap([COLORS[a] for a in FACTOR_ACTIONS])
    ax.imshow(image, origin="lower", extent=(0, 1, 0, CORNERS[2, 1]), cmap=cmap,
              vmin=-0.5, vmax=2.5, alpha=0.75, interpolation="nearest", rasterized=True)


def plot_maps(arrays: Path, output: Path) -> None:
    data = np.load(arrays)
    beliefs = data["factor_belief"][:, 1]
    actions = ACTION_TABLE[data["actions"]][:, 1]
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.4), gridspec_kw={"wspace": 0.25})
    _region(axes[0], 2)
    _simplex_axes(axes[0], "Best action, reward state 3\n(training task)")
    _region(axes[1], 0)
    _simplex_axes(axes[1], "Best action, reward state 1\n(moved factor-2 reward)")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(beliefs))[:6000]
    xy = to_xy(beliefs[order])
    axes[2].scatter(xy[:, 0], xy[:, 1], s=1.5, lw=0,
                    c=[COLORS[int(a)] for a in actions[order]], rasterized=True)
    _simplex_axes(axes[2], "Unsteered agent's factor-2 action\n(exact belief, held-out)")
    handles = [plt.Line2D([], [], marker="s", ls="", color=COLORS[a], label=FACTOR_ACTION_NAMES[a])
               for a in FACTOR_ACTIONS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02, dpi=300)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


def plot_rewards(report: Path, output: Path) -> None:
    confirmation = json.loads(report.read_text())["confirmation"]
    strengths = json.loads(report.read_text())["strength_tuning"]["selected_strength"]
    rows = [
        ("Unsteered", confirmation["intact"]),
        ("Uniform random", confirmation["random"]),
        (f"Factored steering ($\\lambda$={strengths['marginal']})",
         confirmation["comparisons"]["block_4_marginal"]),
        (f"Joint steering ($\\lambda$={strengths['joint']})",
         confirmation["comparisons"]["block_4_joint"]),
        ("Exact-belief best-action map", confirmation["exact_belief_best_action_map"]),
    ]
    fig, ax = plt.subplots(figsize=(3.25, 2.4))
    y = np.arange(len(rows))[::-1]
    for offset, factor, color in ((0.18, 0, "#7f7f7f"), (-0.18, 1, "#2ca02c")):
        ax.barh(y + offset, [r["factor_reward"][factor] for _, r in rows], height=0.34,
                xerr=[r["factor_reward_se"][factor] for _, r in rows], color=color,
                label=f"Factor {factor + 1}", error_kw={"lw": 0.6})
    ax.set_yticks(y, [name for name, _ in rows])
    ax.set_xlabel("Mean reward per step")
    ax.legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_maps(args.arrays, args.output_dir / "moved_reward_best_action_map.pdf")
    plot_rewards(args.report, args.output_dir / "moved_reward_steering_rewards.pdf")


if __name__ == "__main__":
    main()
