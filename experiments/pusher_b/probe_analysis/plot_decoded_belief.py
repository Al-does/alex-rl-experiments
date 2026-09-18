"""Target vs affine-decoded 3-state belief for the final rl_b10_continue and rl_b10_aux_ce checkpoints.

Run: uv run python -m experiments.pusher_b.probe_analysis.plot_decoded_belief
Top row: one held-out episode (target solid, decoded dashed). Bottom row: decoded vs target over all held-out positions.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiments.pusher_b.learning  # noqa: F401
from experiments.pusher_b.probe_analysis.probe_curves import (CKPTS, N_FIT, N_TEST, ROOT, SEEDS, load_module_only,
                          rl_forward, sample, targets)

PRESET = "b10"
RUNS = {"rl_b10_continue": "PPO b=0.1 (no aux, 116M steps)", "rl_b10_aux_ce": "PPO b=0.1 + next-token CE aux (116M steps)"}
EPISODE = 3
SHOW = 48  # positions shown in the time-series panel
COLORS = ("#1f77b4", "#d62728", "#2ca02c")


def fit_decode(a_fit, y_fit, a_test, ridge=1e-6):
    def design(a):
        a = a.reshape(-1, a.shape[-1])
        return np.concatenate([a, np.ones((len(a), 1))], axis=1)
    x = design(a_fit)
    w = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y_fit.reshape(-1, y_fit.shape[-1]))
    return (design(a_test) @ w).reshape(a_test.shape[0], a_test.shape[1], -1)


def main():
    fit_t = sample(PRESET, N_FIT, SEEDS["fit"])
    test_t = sample(PRESET, N_TEST, SEEDS["test"])
    fb, _ = targets(fit_t, PRESET)
    tb, _ = targets(test_t, PRESET)
    fig, axes = plt.subplots(2, len(RUNS), figsize=(7 * len(RUNS), 9))
    for col, (leaf, name) in enumerate(RUNS.items()):
        path = next((CKPTS / leaf / "tune").glob("*/checkpoint_*"))
        module = load_module_only(path).eval()
        a_fit, _ = rl_forward(module, fit_t)
        a_test, _ = rl_forward(module, test_t)
        dec = fit_decode(a_fit, fb, a_test)
        r2 = 1 - ((tb - dec) ** 2).sum() / ((tb - tb.reshape(-1, 3).mean(0)) ** 2).sum()

        ax = axes[0, col]
        for s in range(3):
            ax.plot(tb[EPISODE, :SHOW, s], color=COLORS[s], lw=2, label=f"target P(state {s})")
            ax.plot(dec[EPISODE, :SHOW, s], color=COLORS[s], lw=1.4, ls="--", label=f"decoded P(state {s})")
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("position in held-out episode")
        ax.set_ylabel("belief")
        ax.set_title(f"{name}\nheld-out episode {EPISODE}, first {SHOW} positions: target (solid) vs decoded (dashed); pooled R$^2$={r2:.4f}", fontsize=10)
        ax.grid(alpha=0.25)
        if col == 0:
            ax.legend(fontsize=8, ncol=3, loc="upper center")

        ax = axes[1, col]
        idx = np.random.default_rng(0).choice(tb.shape[0] * tb.shape[1], 20_000, replace=False)
        for s in range(3):
            ax.scatter(tb.reshape(-1, 3)[idx, s], dec.reshape(-1, 3)[idx, s], s=2, alpha=0.15, color=COLORS[s], label=f"state {s}")
        ax.plot([0, 1], [0, 1], "k-", lw=1)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.3, 1.3)
        ax.set_xlabel("target belief")
        ax.set_ylabel("decoded belief")
        ax.set_title("all held-out positions (20k sample)", fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, markerscale=6)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "pusher_b_decoded_belief.png", dpi=150)
    print("saved")


if __name__ == "__main__":
    main()
