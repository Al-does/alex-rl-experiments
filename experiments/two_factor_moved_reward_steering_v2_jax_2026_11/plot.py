"""Best-action maps on the factor-2 simplex and confirmation reward panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from experiments.two_factor_moved_reward_steering_v2_jax_2026_11.steering import (  # noqa: E402
    ACTION_TABLE,
    action_table,
)

ACTION_NAMES = ("Noop", "Tilt X", "Tilt Y")
COLORS = ("#9e9e9e", "#1f77b4", "#d62728")
CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
AGENT_MARKERS = {"ctx32_2026_11": "o", "ctx32_eps05_2026_11": "s"}
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42,
})


def _simplex_axes(ax, title: str) -> None:
    ax.plot(*np.vstack((CORNERS, CORNERS[:1])).T, color="black", lw=0.6)
    for (x, y), label, dy in zip(CORNERS, ("1", "2", "3"), (-0.07, -0.07, 0.06)):
        ax.text(x, y + dy, f"state {label}", ha="center", va="center", fontsize=7)
    ax.set_title(title, fontsize=8, pad=12)
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, CORNERS[2, 1] + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")


def _region(ax, table: np.ndarray, resolution: int = 400) -> None:
    xs, ys = np.meshgrid(np.linspace(0, 1, resolution), np.linspace(0, CORNERS[2, 1], resolution))
    b2 = ys / CORNERS[2, 1]
    b1 = xs - 0.5 * b2
    b0 = 1 - b1 - b2
    inside = (b0 >= 0) & (b1 >= 0)
    actions = (np.stack((b0, b1, b2), -1) @ table).argmax(-1)
    ax.imshow(np.where(inside, actions, np.nan), origin="lower", extent=(0, 1, 0, CORNERS[2, 1]),
              cmap=ListedColormap(COLORS), vmin=-0.5, vmax=2.5, alpha=0.75,
              interpolation="nearest", rasterized=True)


def plot_maps(arrays: Path, output: Path) -> None:
    data = np.load(arrays)
    beliefs = data["factor_belief"][:, 1]
    actions = ACTION_TABLE[data["actions"]][:, 1]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0), gridspec_kw={"wspace": 0.35})
    for ax, (kind, state, title) in zip(axes, (
            ("myopic", 2, "Next-step map,\nreward state 3 (training)"),
            ("myopic", 0, "Next-step map,\nreward state 1 (moved)"),
            ("discounted", 0, "Long-run map, reward\nstate 1 (moved, $\\gamma$=0.99)"))):
        _region(ax, action_table(kind, state))
        _simplex_axes(ax, title)
    order = np.random.default_rng(0).permutation(len(beliefs))[:6000]
    xy = beliefs[order] @ CORNERS
    axes[3].scatter(xy[:, 0], xy[:, 1], s=1.5, lw=0, c=[COLORS[a] for a in actions[order]],
                    rasterized=True)
    _simplex_axes(axes[3], "Unsteered Joint Reward agent\n(no $\\varepsilon$, seed 42)")
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=n)
               for c, n in zip(COLORS, ACTION_NAMES)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.02))
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


def plot_rewards(reports: list[Path], output: Path, title: str) -> None:
    loaded = [json.loads(p.read_text()) for p in reports]
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.6), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(ROWS))[::-1]
    for factor, ax in enumerate(axes):
        values = np.array([[pick(r["confirmation"])["factor_reward"][factor] for _, pick in ROWS]
                           for r in loaded])
        ax.barh(y, values.mean(0), height=0.6, color=("#7f7f7f", "#2ca02c")[factor], alpha=0.45)
        for report, row in zip(loaded, values):
            study, seed = report["source"]["study"], report["source"]["seed"]
            tag = study.removeprefix("reward_factors_persistent_nuisance_v2_jax_temp15_")
            ax.scatter(row, y + (seed - 42.5) * 0.3, s=10, marker=AGENT_MARKERS[tag],
                       facecolor="none" if seed == 43 else "black", edgecolor="black", lw=0.6,
                       label=f"{'no ε' if tag == 'ctx32_2026_11' else 'ε=0.05'}, seed {seed}")
        ax.set_title(f"Factor {factor + 1}" + (" (moved to state 1)" if factor else " (state 3)"), fontsize=8)
        ax.set_xlabel("Mean reward per step")
        ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, [name for name, _ in ROWS])
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=4, fontsize=6,
               bbox_to_anchor=(0.5, -0.1))
    fig.suptitle(title, fontsize=8, y=1.02)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_maps(args.arrays, args.output_dir / "v2_moved_reward_best_action_map.pdf")
    for condition, label in (("reward_both", "Joint Reward agents"),
                             ("reward_factor_1", "Unrewarded Factor 2 agents")):
        reports = sorted(args.results.glob(f"*__{condition}__seed*.json"))
        if reports:
            plot_rewards(reports, args.output_dir / f"v2_moved_reward_steering_{condition}.pdf",
                         f"{label} in the moved-reward environment (block 4)")


if __name__ == "__main__":
    main()
