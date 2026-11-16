"""Per-agent long-run-map steering at block 4: emission-eps02 agents against the v2 agents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STUDY_DIR = Path(__file__).resolve().parent
V2_RESULTS = STUDY_DIR.parent / "two_factor_coupled_moving_reward_steering_v2_jax_2026_11" / "results"
ROWS = (("Unsteered", lambda c: c["intact"]),
        ("Factored steering", lambda c: c["comparisons"]["discounted_marginal_block_4"]),
        ("Joint steering", lambda c: c["comparisons"]["discounted_joint_block_4"]),
        ("Null, factored", lambda c: c["matched_null"]["discounted_marginal"]),
        ("Null, joint", lambda c: c["matched_null"]["discounted_joint"]))
PANELS = (("xor", "XOR reward"), ("moving", "Moving reward, total"))
GROUPS = (("emission ε=0.02 (15 seeds)", "black", 0.15), ("no emission ε (15 seeds)", "#1f77b4", -0.2))


def plot(eps02: list[dict], v2: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6), sharey=True, gridspec_kw={"wspace": 0.08})
    rng = np.random.default_rng(0)
    for ax, (reward, title) in zip(axes, PANELS):
        for reports, (label, color, offset) in zip((eps02, v2), GROUPS):
            confirmations = [r["rewards"][reward]["confirmation"] for r in reports]
            exact = confirmations[0]["exact_belief_discounted_map"]["total_reward"]
            ax.axvline(exact, color=color, ls="--", lw=0.7)
            for i, (_, pick) in enumerate(ROWS):
                y = len(ROWS) - 1 - i + offset
                values = [pick(c)["total_reward"] for c in confirmations]
                ax.scatter(values, y + rng.uniform(-0.1, 0.1, len(values)), s=7, color=color,
                           alpha=0.7, lw=0, label=label if i == 0 else None)
                ax.plot([np.mean(values)] * 2, [y - 0.16, y + 0.16], color=color, lw=1.4)
        ax.set_title(f"{title} (dashed: exact belief, long-run map)", fontsize=7)
        ax.set_xlabel("Mean reward per step", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(ROWS))[::-1], [name for name, _ in ROWS])
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=6, frameon=False, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Long-run-map steering at block 4, per Joint Reward agent (256 episodes each)",
                 fontsize=8, y=1.02)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=STUDY_DIR / "results")
    args = parser.parse_args()
    load = lambda paths: [json.loads(p.read_text()) for p in sorted(paths)]
    plot(load(args.results.glob("eps02_ctx32__reward_both__seed*.json")),
         load(V2_RESULTS.glob("ctx32_2026_11__reward_both__seed*.json")),
         args.results / "eps02_xor_moving_steering_15_seeds.pdf")


if __name__ == "__main__":
    main()
