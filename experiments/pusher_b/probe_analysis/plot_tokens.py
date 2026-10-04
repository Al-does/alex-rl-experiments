"""Token-for-token comparison: x = tokens consumed (supervised: updates*512*127; PPO: env steps)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "results" / "probe_curves.json").read_text())
EP = 127

COLORS = {"supervised_b90": "#1f77b4", "supervised_b10": "#17becf",
          "rl_b90": "#d62728", "rl_b10": "#ff7f0e", "rl_b10_continue": "#ff7f0e"}
NAMES = {"supervised_b90": "Supervised b=0.9 (65k tokens/update)", "supervised_b10": "Supervised b=0.1 (65k tokens/update)",
         "rl_b90": "PPO b=0.9 (262k tokens/iter)", "rl_b10": "PPO b=0.1 (262k tokens/iter)",
         "rl_b10_continue": "PPO b=0.1 continuation (1.05M tokens/iter)"}
MARK = {"rl_b10_continue": ("^", "D")}

fig, ax = plt.subplots(figsize=(11, 6.5))
ax2 = ax.twinx()
for leaf, points in data["runs"].items():
    x = [p["sequences_seen"] * EP for p in points]
    m1, m2 = MARK.get(leaf, ("o", "s"))
    ax.plot(x, [1 - p["r2"] for p in points], "-" + m1, color=COLORS[leaf], lw=2, ms=5)
    ax2.plot(x, [p["accuracy"] for p in points], "--" + m2, color=COLORS[leaf], lw=1.4, ms=4, alpha=0.8)
for key, y in [(k, v["greedy_expected"]) for k, v in data["bayes"].items()]:
    ax2.axhline(y, color="k", ls=":", lw=1.2, alpha=0.7)
    ax2.annotate(f"Bayes-optimal greedy accuracy, {key} = {y:.3f}", xy=(1, y), xycoords=("axes fraction", "data"),
                 xytext=(-8, -12), textcoords="offset points", ha="right", fontsize=9, color="0.3")
ax.set_xscale("symlog", linthresh=1e5)
ax.set_xlim(-2e4, 1e9)
ax.set_yscale("log")
ax.set_ylabel(r"1 - R$^2$ (affine probe: last-layer residual -> Bayesian 3-state belief, held-out)")
ax2.set_ylabel("greedy next-token accuracy (held-out)")
ax.set_xlabel("tokens consumed (supervised: updates x 512 seq x 127 tok;  PPO: env steps = agent decisions)")
ax.set_title("Pusher-B, token-for-token: belief-probe error (left, solid) and task accuracy (right, dashed)")
ax.grid(True, which="both", alpha=0.3)
handles = [Line2D([], [], color=COLORS[k], lw=2, label=NAMES[k]) for k in data["runs"]]
handles += [Line2D([], [], color="k", marker="o", label=r"1 - R$^2$ (left axis)"),
            Line2D([], [], color="k", ls="--", marker="s", label="accuracy (right axis)"),
            Line2D([], [], color="k", ls=":", label="Bayes-optimal accuracy")]
ax.legend(handles=handles, loc="lower left", fontsize=9, ncol=2)
fig.tight_layout()
fig.savefig(ROOT / "results" / "pusher_b_probe_tokens.png", dpi=150)
print("saved")
