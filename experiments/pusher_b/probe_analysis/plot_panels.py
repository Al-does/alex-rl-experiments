"""Two stacked panels vs agent steps: (top) 1-R^2 belief-probe error, (bottom) greedy task accuracy.

x = agent steps. PPO: env steps (one agent decision per token). Supervised: sequences x 127
tokens, i.e. the number of labelled next-token predictions consumed. The PPO b=0.1 line
chains rl_b10 -> rl_b10_continue -> rl_b10_continue2 (warm starts).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "results" / "probe_curves.json").read_text())
EP = 127

SERIES = {
    "Supervised b=0.9": (["supervised_b90"], "#1f77b4", "-o"),
    "PPO b=0.9": (["rl_b90"], "#d62728", "-s"),
    "Supervised b=0.1": (["supervised_b10"], "#17becf", "-o"),
    "PPO b=0.1 (+ two warm-started continuations)": (
        ["rl_b10", "rl_b10_continue", "rl_b10_continue2"], "#ff7f0e", "-s"),
}


def points(leaves):
    seen, out = set(), []
    for leaf in leaves:
        for p in data["runs"][leaf]:
            x = p["sequences_seen"] * EP
            if x in seen:
                continue
            seen.add(x)
            out.append((x, 1 - p["r2"], p["accuracy"]))
    return sorted(out)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9.5), sharex=True)
for name, (leaves, color, style) in SERIES.items():
    pts = points(leaves)
    x = [p[0] for p in pts]
    ax1.plot(x, [p[1] for p in pts], style, color=color, lw=2, ms=5, label=name)
    ax2.plot(x, [p[2] for p in pts], style, color=color, lw=2, ms=5, label=name)

for key, v in data["bayes"].items():
    y = v["greedy_expected"]
    ax2.axhline(y, color="k", ls=":", lw=1.2, alpha=0.7)
    ax2.annotate(f"Bayes-optimal greedy accuracy, {key} = {y:.3f}", xy=(1, y),
                 xycoords=("axes fraction", "data"), xytext=(-8, 4),
                 textcoords="offset points", ha="right", fontsize=9, color="0.3")
ax2.axhline(0.5, color="0.5", ls="--", lw=1, alpha=0.6)
ax2.annotate("chance (constant guess) = 0.500", xy=(1, 0.5), xycoords=("axes fraction", "data"),
             xytext=(-8, 4), textcoords="offset points", ha="right", fontsize=9, color="0.4")

ax1.set_yscale("log")
ax1.set_ylabel(r"1 - R$^2$  (affine probe, last-layer residual -> Bayesian belief, held-out)")
ax1.set_title("Pusher-B: belief-probe error vs agent steps")
ax2.set_ylabel("greedy next-token accuracy (held-out)")
ax2.set_title("Pusher-B: task accuracy vs agent steps")
ax2.set_xlabel("agent steps  (PPO: env steps;  supervised: sequences x 127 labelled tokens)")
ax2.set_xlim(0, 6.7e8)
ax2.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
for ax in (ax1, ax2):
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right" if ax is ax1 else "lower right", fontsize=9)
fig.tight_layout()
out = ROOT / "results" / "pusher_b_probe_panels.png"
fig.savefig(out, dpi=150)
print(out)
