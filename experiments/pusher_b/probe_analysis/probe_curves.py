"""Belief-probe R^2 and greedy next-token accuracy across Pusher-B checkpoints.

Run: uv run python -m experiments.pusher_b.probe_analysis.probe_curves (checkpoints under artifacts/ckpts/<leaf>, pulled from B2).
Writes results/probe_curves.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

from analysis.checkpoints import load_module_only
from experiments.pusher_b.process import (
    BOS_TOKEN, EPISODE_LENGTH, TOKEN_COUNT, pusher_b_model,
)
from experiments.pusher_b.supervised import (
    MODEL_CONFIG, SequenceSampler, beliefs_from_tokens, next_token_distributions,
)
from experiments.pusher_b.supervised_model import PusherBTransformer

ROOT = Path(__file__).resolve().parent
CKPTS = ROOT / "artifacts" / "ckpts"
N_FIT = 2048
N_TEST = 2048
N_BAYES = 50_000
SUP_BATCH = 512
SEEDS = {"fit": 7_001, "test": 7_002, "bayes": 7_003}
torch.set_num_threads(8)


def sample(preset: str, n: int, seed: int) -> np.ndarray:
    return SequenceSampler(preset=preset, seed=seed).sample(n)  # (n, 128), BOS first


def targets(tokens: np.ndarray, preset: str):
    beliefs = beliefs_from_tokens(tokens, preset=preset)[:, :EPISODE_LENGTH]  # decision k sees tokens[:, :k+1]
    next_tokens = tokens[:, 1:]  # (n, 127)
    return beliefs, next_tokens


def r2_pooled(acts_fit, y_fit, acts_test, y_test, ridge: float = 1e-6) -> float:
    def design(a):
        a = a.reshape(-1, a.shape[-1])
        return np.concatenate([a, np.ones((len(a), 1))], axis=1)
    x = design(acts_fit)
    y = y_fit.reshape(-1, y_fit.shape[-1])
    gram = x.T @ x + ridge * np.eye(x.shape[1])
    w = np.linalg.solve(gram, x.T @ y)
    xt = design(acts_test)
    yt = y_test.reshape(-1, y_test.shape[-1])
    pred = xt @ w
    ss_res = ((yt - pred) ** 2).sum()
    ss_tot = ((yt - yt.mean(axis=0)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------- supervised
@torch.inference_mode()
def supervised_forward(model: PusherBTransformer, tokens: np.ndarray):
    acts, preds = [], []
    for i in range(0, len(tokens), SUP_BATCH):
        t = torch.as_tensor(tokens[i:i + SUP_BATCH], dtype=torch.long)
        hidden = model.token_embedding(t)
        for block in model.blocks:
            hidden = block(hidden)
        logits = model.unembedding(model.final_norm(hidden))
        acts.append(hidden[:, :EPISODE_LENGTH].double().numpy())
        preds.append(logits[:, :EPISODE_LENGTH, :TOKEN_COUNT].argmax(-1).numpy())
    return np.concatenate(acts), np.concatenate(preds)


def analyze_supervised(leaf: str, preset: str, fit, test):
    out = []
    for path in sorted((CKPTS / leaf / "checkpoints").glob("step_*.pt"),
                       key=lambda p: int(p.stem.split("_")[1])):
        step = int(path.stem.split("_")[1])
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = PusherBTransformer(MODEL_CONFIG)
        model.load_state_dict(state["model_state"])
        model.eval()
        a_fit, _ = supervised_forward(model, fit["tokens"])
        a_test, pred = supervised_forward(model, test["tokens"])
        r2 = r2_pooled(a_fit, fit["beliefs"], a_test, test["beliefs"])
        acc = float((pred == test["next"]).mean())
        out.append({"label": f"step {step}", "step": step,
                    "sequences_seen": step * 512, "r2": r2, "accuracy": acc})
        print(leaf, step, f"R2={r2:.6f} 1-R2={1-r2:.3e} acc={acc:.4f}", flush=True)
    return out


# ------------------------------------------------------------------------ RL
def rl_checkpoints(leaf: str):
    base = CKPTS / leaf
    items = [(base / "initial_checkpoint", 0, "init")]
    for d in sorted((base / "log_spaced_checkpoints").iterdir()):
        m = re.match(r"iteration_(\d+)_steps_(\d+)", d.name)
        if m is None or not d.is_dir():
            continue
        items.append((d, int(m.group(2)), f"iter {int(m.group(1))}"))
    final = next((base / "tune").glob("*/checkpoint_000000"))
    curves = [json.loads(l) for l in open(next(
        (ROOT.parent / leaf / "results")
        .glob("*/training_curves.jsonl")))]
    items.append((final, int(curves[-1]["steps"]), f"iter {curves[-1]['iteration']} (final)"))
    return items


@torch.inference_mode()
def rl_forward(module, tokens: np.ndarray):
    n = len(tokens)
    t = torch.as_tensor(tokens[:, 1:EPISODE_LENGTH], dtype=torch.long)
    obs = torch.nn.functional.one_hot(t, num_classes=TOKEN_COUNT).float()
    obs = torch.cat([torch.zeros(n, 1, TOKEN_COUNT), obs], dim=1)  # BOS = zeros, len 127
    acts, preds = [], []
    for i in range(0, n, SUP_BATCH):
        o = obs[i:i + SUP_BATCH]
        residual = module.encoder.forward_complete_episode(o, apply_final_norm=False)
        logits = module.action_distribution_inputs(module.encoder.final_norm(residual))
        acts.append(residual.double().numpy())
        preds.append(logits.argmax(-1).numpy())
    return np.concatenate(acts), np.concatenate(preds)


def analyze_rl(leaf: str, preset: str, fit, test):
    out = []
    for path, env_steps, label in rl_checkpoints(leaf):
        module = load_module_only(path).eval()
        a_fit, _ = rl_forward(module, fit["tokens"])
        a_test, pred = rl_forward(module, test["tokens"])
        r2 = r2_pooled(a_fit, fit["beliefs"], a_test, test["beliefs"])
        acc = float((pred == test["next"]).mean())
        out.append({"label": label, "env_steps": env_steps,
                    "sequences_seen": env_steps / EPISODE_LENGTH, "r2": r2, "accuracy": acc})
        print(leaf, label, f"R2={r2:.6f} 1-R2={1-r2:.3e} acc={acc:.4f}", flush=True)
    return out


# --------------------------------------------------------------------- Bayes
def bayes_accuracy(preset: str) -> dict:
    tokens = sample(preset, N_BAYES, SEEDS["bayes"])
    beliefs, nxt = targets(tokens, preset)
    p = next_token_distributions(beliefs, preset=preset)  # (n,127,2)
    greedy_pred = p.argmax(-1)
    return {
        "greedy_expected": float(p.max(-1).mean()),          # E[max_t P(t|h)]
        "greedy_empirical": float((greedy_pred == nxt).mean()),
        "sampling_expected": float((p ** 2).sum(-1).mean()),  # policy that samples from Bayes posterior
        "loss_floor_nats": float(-(np.log(p) * np.eye(2)[nxt]).sum(-1).mean()),
        "n_sequences": N_BAYES,
    }


def main():
    out_path = ROOT / "results" / "probe_curves.json"
    prior = json.loads(out_path.read_text())["runs"] if out_path.exists() else {}
    results = {"runs": {}, "bayes": {}, "settings": {
        "n_fit": N_FIT, "n_test": N_TEST, "positions": EPISODE_LENGTH, "seeds": SEEDS,
        "probe": "affine least squares, residual stream after last block (pre-final-norm) -> 3-state Bayesian belief; pooled R^2 on held-out sequences",
        "accuracy": "greedy argmax next-token prediction on held-out sequences",
    }}
    for preset in ("b90", "b10"):
        results["bayes"][preset] = bayes_accuracy(preset)
        print("bayes", preset, results["bayes"][preset], flush=True)
        fit_t = sample(preset, N_FIT, SEEDS["fit"])
        test_t = sample(preset, N_TEST, SEEDS["test"])
        fb, fn = targets(fit_t, preset)
        tb, tn = targets(test_t, preset)
        fit = {"tokens": fit_t, "beliefs": fb, "next": fn}
        test = {"tokens": test_t, "beliefs": tb, "next": tn}
        for leaf, analyze in ((f"supervised_{preset}", analyze_supervised), (f"rl_{preset}", analyze_rl)):
            results["runs"][leaf] = prior[leaf] if leaf in prior else analyze(leaf, preset, fit, test)
            out_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
