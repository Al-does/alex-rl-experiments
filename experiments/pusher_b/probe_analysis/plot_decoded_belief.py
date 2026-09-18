"""Target vs affine-decoded 3-state belief on the simplex for the final checkpoint of each run in RUNS (RL and supervised).

Run: uv run python -m experiments.pusher_b.probe_analysis.plot_decoded_belief
Writes results/pusher_b_decoded_belief_<leaf>.png (one figure per run: true targets | raw predictions on the triangle).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import experiments.pusher_b.learning  # noqa: F401
from analysis.plots import plot_belief_comparison
from experiments.pusher_b.probe_analysis.probe_curves import (CKPTS, MODEL_CONFIG, N_FIT, N_TEST, ROOT, SEEDS,
                          PusherBTransformer, load_module_only, rl_forward, sample, supervised_forward, targets)

PRESET = "b10"
RUNS = {"rl_b10_continue": "PPO b=0.1 (no aux, 116M steps)", "rl_b10_aux_ce": "PPO b=0.1 + next-token CE aux (116M steps)",
        "supervised_b10": "supervised next-token b=0.1 (10k steps x 512 seqs, run 20260913T224010Z-054af3e0)"}
N_POINTS = 20_000


def fit_decode(a_fit, y_fit, a_test, ridge=1e-6):
    def design(a):
        a = a.reshape(-1, a.shape[-1])
        return np.concatenate([a, np.ones((len(a), 1))], axis=1)
    x = design(a_fit)
    w = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y_fit.reshape(-1, y_fit.shape[-1]))
    return design(a_test) @ w


def main():
    fit_t = sample(PRESET, N_FIT, SEEDS["fit"])
    test_t = sample(PRESET, N_TEST, SEEDS["test"])
    fb, _ = targets(fit_t, PRESET)
    tb, _ = targets(test_t, PRESET)
    flat_t = tb.reshape(-1, 3)
    idx = np.random.default_rng(0).choice(len(flat_t), N_POINTS, replace=False)
    for leaf, name in RUNS.items():
        if leaf.startswith("supervised"):
            state = torch.load(CKPTS / leaf / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
            model = PusherBTransformer(MODEL_CONFIG)
            model.load_state_dict(state["model_state"])
            forward, module = supervised_forward, model.eval()
        else:
            forward, module = rl_forward, load_module_only(next((CKPTS / leaf / "tune").glob("*/checkpoint_*"))).eval()
        a_fit, _ = forward(module, fit_t)
        a_test, _ = forward(module, test_t)
        dec = fit_decode(a_fit, fb, a_test)
        # affine probe output onto the belief plane: it sums to 1 only approximately, so project for the simplex check
        dec = dec - (dec.sum(1, keepdims=True) - 1) / 3
        fig = plot_belief_comparison(flat_t[idx], dec[idx], state_labels=["state 0", "state 1", "state 2"],
                                     title=f"{name}\nheld-out beliefs ({N_POINTS} positions): true vs affine-decoded")
        fig.savefig(ROOT / "results" / f"pusher_b_decoded_belief_{leaf}.png", dpi=150)
        plt.close(fig)
        print("saved", leaf)


if __name__ == "__main__":
    main()
