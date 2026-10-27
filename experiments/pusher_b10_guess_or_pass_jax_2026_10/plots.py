"""Render reward-vs-steps panels from the committed campaign training curves.

Usage: ``python -m experiments.pusher_b10_guess_or_pass_jax_2026_10.plots``
writes four PNGs under ``results/``: PPO checkpoints alone and PPO vs PPO+CE,
each for the full-trunk and frozen-trunk runs at one cost (default c=2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY_DIR = Path(__file__).resolve().parent
SEED = 42
COST_DIRS = {2.0: "ppo_c2", 1.5: "ppo_c1p5", 1.0: "ppo_c1"}
PPO_SOURCES = ("ppo_iter4", "ppo_iter8", "ppo_iter143")
CE_SOURCES = ("ppo_ce_iter4", "ppo_ce_iter8", "ppo_ce_iter211")
COLORS = ("tab:blue", "tab:orange", "tab:green")
MODES = (
    ("full", "full trunk (encoder fine-tuned)"),
    ("frozen", "frozen trunk (heads only)"),
)


def _run_dir(cost: float, source: str, mode: str) -> Path:
    leaf = COST_DIRS[cost]
    return (
        STUDY_DIR
        / leaf
        / "results"
        / f"guess-or-pass-jax-seed{SEED}-{leaf}-{source}-{mode}"
    )


def load_curve(cost: float, source: str, mode: str) -> tuple[list[float], list[float]]:
    path = _run_dir(cost, source, mode) / "training_curves.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [r["steps"] / 1e6 for r in rows], [r["reward_per_step"] for r in rows]


def bayes_reward(cost: float) -> float:
    summary = _run_dir(cost, "scratch", "scratch") / "summary.json"
    return json.loads(summary.read_text())["bayes_expected_reward"]


def _label(source: str) -> str:
    family, _, iteration = source.rpartition("_")
    return f"{'PPO+CE' if family == 'ppo_ce' else 'PPO'} {iteration}"


def render(cost: float, mode: str, title: str, *, with_ce: bool, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for ppo, ce, color in zip(PPO_SOURCES, CE_SOURCES, COLORS, strict=True):
        x, y = load_curve(cost, ppo, mode)
        ax.plot(x, y, color=color, lw=2, label=_label(ppo))
        if with_ce:
            x, y = load_curve(cost, ce, mode)
            ax.plot(x, y, color=color, lw=2, ls="--", label=_label(ce))
    x, y = load_curve(cost, "scratch", "scratch")
    ax.plot(x, y, color="gray", lw=1.5, ls=":", label="scratch (reference)")
    bayes = bayes_reward(cost)
    ax.axhline(bayes, color="k", lw=1, ls="-.", label=f"Bayes optimal ({bayes:.3f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(-0.4, 0.1)
    ax.set_xlabel("training env steps (millions)")
    ax.set_ylabel("reward per step")
    ax.set_title(f"Guess-or-pass, c={cost:g}, seed {SEED} — {title}")
    ax.legend(fontsize=8, loc="lower right", ncol=2 if with_ce else 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost", type=float, default=2.0, choices=sorted(COST_DIRS))
    parser.add_argument("--output-dir", type=Path, default=STUDY_DIR / "results")
    args = parser.parse_args()
    tag = COST_DIRS[args.cost].removeprefix("ppo_")
    for mode, title in MODES:
        render(
            args.cost,
            mode,
            title,
            with_ce=False,
            out=args.output_dir / f"{tag}_ppo_checkpoints_{mode}.png",
        )
        render(
            args.cost,
            mode,
            title,
            with_ce=True,
            out=args.output_dir / f"{tag}_ppo_vs_ppo_ce_{mode}.png",
        )
    print(f"wrote 4 panels to {args.output_dir}")


if __name__ == "__main__":
    main()
