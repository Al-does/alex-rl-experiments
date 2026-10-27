"""Render the grokking_in_rl figures from the compact data under ``data/``.

Usage (from this directory): ``uvx --python 3.13 --with matplotlib python plots.py``

Figure 1a/1b: task-1 greedy accuracy and probe 1-R^2 vs env steps for PPO and
PPO+CE, with checkpoints A/B/C marked. Figures 2-4: task-2 (guess-or-pass,
c=2) reward curves for the same checkpoints, full trunk and frozen trunk.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figures"
SEED = 42

CKPT_COLORS = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
CURVE_COLOR = "0.3"
SCRATCH_COLOR = "0.75"

# label -> (source name in the task-2 campaign, task-1 checkpoint label)
CHECKPOINTS = {
    "PPO": {"A": ("ppo_iter4", "iter 4"), "B": ("ppo_iter8", "iter 8"), "C": ("ppo_iter143", "iter 143")},
    "PPO+CE": {"A": ("ppo_ce_iter4", "iter 4"), "B": ("ppo_ce_iter8", "iter 8"), "C": ("ppo_ce_iter211", "iter 211")},
}


def task1_curve(family: str) -> list[dict]:
    if family == "PPO":
        base = json.loads((DATA / "probes" / "probe_curves.json").read_text())["runs"]
        points = base["rl_b10"] + base["rl_b10_continue"][1:]  # drop duplicated restore point
    else:
        points = json.loads((DATA / "probes" / "probe_aux_ce.json").read_text())["runs"]["rl_b10_aux_ce"]
    return points


def bayes_accuracy() -> float:
    return json.loads((DATA / "probes" / "probe_curves.json").read_text())["bayes"]["b10"]["greedy_expected"]


def _ckpt_point(points: list[dict], label: str) -> dict:
    return next(p for p in points if p["label"].startswith(label))


def render_task1(family: str, out: Path) -> None:
    points = task1_curve(family)
    bayes = bayes_accuracy()
    x = [p["env_steps"] / 1e6 for p in points]
    acc = [p["accuracy"] for p in points]
    err = [1 - p["r2"] for p in points]

    fig, (ax_acc, ax_err) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax_acc.plot(x, acc, "-o", color=CURVE_COLOR, lw=1.5, ms=3.5, label=f"{family} (seed {SEED})")
    ax_acc.axhline(bayes, color="k", ls=":", lw=1.2, label=f"Bayes-optimal greedy accuracy ({bayes:.3f})")
    ax_err.plot(x, err, "-o", color=CURVE_COLOR, lw=1.5, ms=3.5)

    # red box: task performance has plateaued at Bayes, probe error keeps falling
    plateau = next(p for p in points if p["accuracy"] >= bayes - 1e-3)
    best = min(points, key=lambda p: 1 - p["r2"])
    x0, x1 = plateau["env_steps"] / 1e6, best["env_steps"] / 1e6
    span = [1 - p["r2"] for p in points if x0 <= p["env_steps"] / 1e6 <= x1]
    y0, y1 = min(span) / 1.25, max(span) * 1.25
    ax_err.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", lw=1.5, zorder=4))

    offsets = {"A": (-14, 12), "B": (6, 12), "C": (-6, 12)}
    for letter, (_, ckpt_label) in CHECKPOINTS[family].items():
        p = _ckpt_point(points, ckpt_label)
        px = p["env_steps"] / 1e6
        for ax, py in ((ax_acc, p["accuracy"]), (ax_err, 1 - p["r2"])):
            ax.plot(px, py, "o", color=CKPT_COLORS[letter], ms=10, mec="k", mew=0.8, zorder=5)
            ax.annotate(letter, (px, py), textcoords="offset points", xytext=offsets[letter],
                        ha="center", fontsize=12, fontweight="bold", color=CKPT_COLORS[letter])
        ax_acc.plot([], [], "o", color=CKPT_COLORS[letter], ms=8, mec="k", mew=0.8,
                    label=f"{letter}: {family} {ckpt_label} ({px:.1f}M steps)")

    ax_acc.set_ylabel("greedy next-token accuracy (held-out)")
    ax_acc.set_ylim(0.45, 0.68)
    ax_acc.set_title(f"Task 1 (Pusher-B b=0.1), {family}: task performance and belief probe")
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(fontsize=8, loc="lower right")
    ax_err.set_yscale("log")
    ax_err.set_ylabel("probe $1-R^2$ (held-out)")
    ax_err.set_xlabel("PPO env steps (millions)")
    ax_err.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def task2_curve(source: str, mode: str) -> tuple[list[float], list[float]]:
    path = DATA / "guess_or_pass_c2" / f"guess-or-pass-jax-seed{SEED}-ppo_c2-{source}-{mode}" / "training_curves.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [r["steps"] / 1e6 for r in rows], [r["reward_per_step"] for r in rows]


def task2_bayes() -> float:
    path = DATA / "guess_or_pass_c2" / f"guess-or-pass-jax-seed{SEED}-ppo_c2-scratch-scratch" / "summary.json"
    return json.loads(path.read_text())["bayes_expected_reward"]


def render_task2(families: tuple[str, ...], mode: str, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    styles = {"PPO": "-", "PPO+CE": "--"}
    for family in families:
        for letter, (source, ckpt_label) in CHECKPOINTS[family].items():
            x, y = task2_curve(source, mode)
            ax.plot(x, y, styles[family] if len(families) > 1 else "-", color=CKPT_COLORS[letter], lw=2,
                    label=f"{letter}: {family} {ckpt_label}")
    x, y = task2_curve("scratch", "scratch")
    ax.plot(x, y, color=SCRATCH_COLOR, lw=1.8, label="scratch (trainable trunk)")
    bayes = task2_bayes()
    ax.axhline(bayes, color="k", lw=1, ls="-.", label=f"Bayes optimal ({bayes:.3f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(-0.4, 0.1)
    ax.set_xlabel("task-2 training env steps (millions)")
    ax.set_ylabel("reward per step")
    ax.set_title(f"Task 2 (guess-or-pass, c=2), seed {SEED}: {title}")
    ax.legend(fontsize=8, loc="lower right", ncol=2 if len(families) > 1 else 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    render_task1("PPO", FIGS / "fig1a_task1_ppo.png")
    render_task1("PPO+CE", FIGS / "fig1b_task1_ppo_ce.png")
    render_task2(("PPO",), "full", "full trunk, PPO checkpoints", FIGS / "fig2_full_ppo.png")
    render_task2(("PPO", "PPO+CE"), "full", "full trunk, PPO vs PPO+CE", FIGS / "fig3_full_ppo_vs_ppo_ce.png")
    render_task2(("PPO",), "frozen", "frozen trunk, PPO checkpoints", FIGS / "fig4a_frozen_ppo.png")
    render_task2(("PPO+CE",), "frozen", "frozen trunk, PPO+CE checkpoints", FIGS / "fig4b_frozen_ppo_ce.png")


if __name__ == "__main__":
    main()
