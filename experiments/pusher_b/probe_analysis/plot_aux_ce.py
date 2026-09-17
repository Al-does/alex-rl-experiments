"""Compare rl_b10_aux_ce (next-token CE aux) with rl_b10 (+continuation): 1-R^2 and accuracy vs env steps."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
aux = json.loads((ROOT / "results" / "probe_aux_ce.json").read_text())
base = json.loads((ROOT / "results" / "probe_curves.json").read_text())

runs = {"rl_b10": base["runs"]["rl_b10"], "rl_b10_continue": base["runs"]["rl_b10_continue"],
        "rl_b10_aux_ce": aux["runs"]["rl_b10_aux_ce"]}
COLORS = {"rl_b10": "#ff7f0e", "rl_b10_continue": "#ff7f0e", "rl_b10_aux_ce": "#2ca02c"}
MARK = {"rl_b10": "o", "rl_b10_continue": "^", "rl_b10_aux_ce": "o"}
NAMES = {"rl_b10": "PPO b=0.1 (no aux)", "rl_b10_continue": "PPO b=0.1 continuation (no aux)",
         "rl_b10_aux_ce": "PPO b=0.1 + next-token CE aux (coef 1)"}
bayes = base["bayes"]["b10"]["greedy_expected"]

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
for leaf, points in runs.items():
    x = [max(p["env_steps"], 1e5) for p in points]  # env_steps=0 shown at 1e5 on the log axis
    ax.plot(x, [1 - p["r2"] for p in points], "-" + MARK[leaf], color=COLORS[leaf], lw=2, ms=5, label=NAMES[leaf])
    ax2.plot(x, [p["accuracy"] for p in points], "-" + MARK[leaf], color=COLORS[leaf], lw=2, ms=5, label=NAMES[leaf])

ax2.axhline(bayes, color="0.35", ls=":", lw=1.5, label=f"Bayes-optimal greedy accuracy = {bayes:.3f}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("1 - R$^2$  (affine probe: last residual -> 3-state belief, held-out)")
ax.set_title("Pusher-B b=0.1: next-token CE auxiliary vs plain PPO (init plotted at 1e5 steps)")
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=9)
ax2.set_xlabel("PPO env steps (log)")
ax2.set_ylabel("greedy next-token accuracy (held-out)")
ax2.set_ylim(0.45, 0.68)
ax2.grid(True, which="both", alpha=0.25)
ax2.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(ROOT / "results" / "pusher_b_probe_aux_ce.png", dpi=160)
print("saved")
