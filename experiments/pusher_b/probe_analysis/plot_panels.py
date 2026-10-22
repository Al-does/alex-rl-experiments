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


BAYES = {"b90": data["bayes"]["b90"]["greedy_expected"], "b10": data["bayes"]["b10"]["greedy_expected"]}
LINTHRESH = 2e-3  # held-out sampling noise on 2048 x 127 predictions is ~1e-3

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9.5), sharex=True)
for name, (leaves, color, style) in SERIES.items():
    pts = points(leaves)
    bayes = BAYES["b90" if "0.9" in name else "b10"]
    x = [p[0] for p in pts]
    ax1.plot(x, [p[1] for p in pts], style, color=color, lw=2, ms=5, label=name)
    ax2.plot(x, [bayes - p[2] for p in pts], style, color=color, lw=2, ms=5, label=name)

ax2.axhline(0, color="k", ls=":", lw=1.2, alpha=0.7)
ax2.annotate("Bayes-optimal greedy accuracy (b90 = 0.727, b10 = 0.638)", xy=(0, 0),
             xycoords=("axes fraction", "data"), xytext=(8, 4),
             textcoords="offset points", ha="left", fontsize=9, color="0.3")
for key, y in BAYES.items():
    ax2.axhline(y - 0.5, color="0.5", ls="--", lw=1, alpha=0.6)
    ax2.annotate(f"chance (constant guess), {key}: {y - 0.5:.3f} below Bayes", xy=(1, y - 0.5),
                 xycoords=("axes fraction", "data"), xytext=(-8, 4),
                 textcoords="offset points", ha="right", fontsize=9, color="0.4")

ax1.set_yscale("log")
ax1.set_ylabel(r"1 - R$^2$  (affine probe -> Bayesian belief, held-out)")
ax1.set_title("Pusher-B: belief-probe error vs agent steps")
ax2.set_yscale("symlog", linthresh=LINTHRESH)
ax2.set_ylim(-LINTHRESH, 0.3)
ax2.set_ylabel("Bayes-optimal acc. - greedy acc. (held-out; symlog)")
ax2.set_title("Pusher-B: task-accuracy shortfall from Bayes-optimal vs agent steps (lower is better)")
ax2.set_xlabel("agent steps  (PPO: env steps;  supervised: sequences x 127 labelled tokens)")
ax2.set_xscale("symlog", linthresh=1e5)
ax2.set_xlim(-2e4, 1e9)
for ax in (ax1, ax2):
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
out = ROOT / "results" / "pusher_b_probe_panels.png"
fig.savefig(out, dpi=150)
print(out)
