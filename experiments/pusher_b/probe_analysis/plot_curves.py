import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "results" / "probe_curves.json").read_text())

COLORS = {"supervised_b90": "#1f77b4", "supervised_b10": "#17becf",
          "rl_b90": "#d62728", "rl_b10": "#ff7f0e", "rl_b10_continue": "#ff7f0e"}
NAMES = {"supervised_b90": "Supervised b=0.9", "supervised_b10": "Supervised b=0.1",
         "rl_b90": "PPO b=0.9", "rl_b10": "PPO b=0.1",
         "rl_b10_continue": "PPO b=0.1 continuation (1M batch, 100M steps)"}
MARK = {"rl_b10_continue": ("^", "D")}

fig, ax = plt.subplots(figsize=(11, 6.5))
ax2 = ax.twinx()
for leaf, points in data["runs"].items():
    x = [p["sequences_seen"] for p in points]
    m1, m2 = MARK.get(leaf, ("o", "s"))
    ax.plot(x, [1 - p["r2"] for p in points], "-" + m1, color=COLORS[leaf], lw=2, ms=5)
    ax2.plot(x, [p["accuracy"] for p in points], "--" + m2, color=COLORS[leaf], lw=1.4, ms=4, alpha=0.8)

for preset, ls, txt in (("b90", "#d62728", "Bayes-optimal greedy accuracy, b=0.9"),
                        ("b10", "#ff7f0e", "Bayes-optimal greedy accuracy, b=0.1")):
    y = data["bayes"][preset]["greedy_expected"]
    ax2.axhline(y, color="0.35", ls=":", lw=1.5)
    ax2.text(5.2e6, y - 0.014, f"{txt} = {y:.3f}", color="0.25", fontsize=9, ha="right")

ax.set_xlim(-1e5, 5.3e6)
ax.ticklabel_format(axis="x", style="plain")
ax.set_yscale("log")
ax.set_xlabel("training sequences consumed (supervised: updates x 512; PPO: env steps / 127)")
ax.set_ylabel("1 - R$^2$  (affine probe: last-layer residual -> Bayesian 3-state belief, held-out)")
ax2.set_ylabel("greedy next-token accuracy (held-out)")
ax2.set_ylim(0.45, 0.78)
ax.grid(True, which="both", alpha=0.25)
ax.set_title("Pusher-B: belief-probe error (left, solid) and task accuracy (right, dashed) vs training")

handles = [Line2D([], [], color=COLORS[k], lw=2, marker=MARK.get(k, ("o",))[0], label=NAMES[k])
           for k in data["runs"]]
handles += [Line2D([], [], color="k", lw=2, marker="o", label="1 - R$^2$ (left axis)"),
            Line2D([], [], color="k", lw=1.4, ls="--", marker="s", label="accuracy (right axis)"),
            Line2D([], [], color="0.35", ls=":", lw=1.5, label="Bayes-optimal accuracy")]
ax.legend(handles=handles, loc="lower center", ncol=2, fontsize=9)
fig.tight_layout()
fig.savefig(ROOT / "results" / "pusher_b_probe_curves.png", dpi=160)
print("saved")
