"""Effective rank and related grokking diagnostics across Pusher-B b10 PPO checkpoints.

Run: uv run python -m experiments.pusher_b.probe_analysis.effective_rank [--leaves rl_b10 ...] [--limit N]
Checkpoints are synced from B2 into artifacts/ckpts/<leaf>/ (module-only subtrees suffice):
  aws s3 sync --endpoint-url $B2_ENDPOINT --exclude "*" --include "*/rl_module/default_policy/*" \
      --include "compact-results/*" s3://$B2_BUCKET/experiments/pusher_b/<leaf>/<run-id>/ artifacts/ckpts/<leaf>/
Writes results/effective_rank.json.

Activation metrics use the same held-out sequences as the belief probe (SEEDS["test"], all 127
decision positions) and are reported for every residual-stream stage: the token embedding, the
output of each block (block_4 is the probe stream, pre-final-norm) and the post-final-norm stream.
Weight metrics cover every 2-D parameter of the restored module (qkv split per projection).
"""
from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path

import numpy as np
import torch

import experiments.pusher_b.learning  # noqa: F401  (registers ActorCriticWithNextTokenAux for restore)
from analysis.checkpoints import load_module_only
from experiments.pusher_b.probe_analysis.probe_curves import (
    CKPTS, EPISODE_LENGTH, N_TEST, ROOT, SEEDS, SUP_BATCH, sample, targets,
)
from experiments.pusher_b.process import TOKEN_COUNT
from experiments.pusher_b.supervised import next_token_distributions

PRESET = "b10"
LEAVES = ("rl_b10", "rl_b10_continue", "rl_b10_aux_ce")
OUT_PATH = ROOT / "results" / "effective_rank.json"


# ------------------------------------------------------------------ metrics
def spectrum_metrics(singular_values: np.ndarray) -> dict[str, float]:
    """Effective rank (Roy & Vetterli 2007), participation ratio and variance-coverage dims."""
    s = np.asarray(singular_values, dtype=np.float64)
    s = s[s > 0]
    if s.size == 0:
        return {"erank": 0.0, "participation_ratio": 0.0, "n90": 0, "n99": 0, "spectral_norm": 0.0}
    p = s / s.sum()
    erank = float(np.exp(-(p * np.log(p)).sum()))
    lam = s ** 2
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    cum = np.cumsum(np.sort(lam)[::-1]) / lam.sum()
    return {
        "erank": erank,
        "participation_ratio": pr,
        "n90": int(np.searchsorted(cum, 0.90) + 1),
        "n99": int(np.searchsorted(cum, 0.99) + 1),
        "spectral_norm": float(s.max()),
    }


class ScatterAccumulator:
    """Running sum / scatter matrix so the SVD never needs the full (N, d) activation matrix."""

    def __init__(self, width: int) -> None:
        self.n = 0
        self.sum = np.zeros(width, dtype=np.float64)
        self.scatter = np.zeros((width, width), dtype=np.float64)

    def add(self, rows: np.ndarray) -> None:
        rows = rows.reshape(-1, rows.shape[-1]).astype(np.float64)
        self.n += len(rows)
        self.sum += rows.sum(axis=0)
        self.scatter += rows.T @ rows

    def metrics(self) -> dict[str, object]:
        mean = self.sum / self.n
        centered = self.scatter - self.n * np.outer(mean, mean)
        centered_eig = np.clip(np.linalg.eigvalsh(centered), 0.0, None)
        raw_eig = np.clip(np.linalg.eigvalsh(self.scatter), 0.0, None)
        out = {f"centered_{k}": v for k, v in spectrum_metrics(np.sqrt(centered_eig)).items()}
        out.update({f"uncentered_{k}": v for k, v in spectrum_metrics(np.sqrt(raw_eig)).items()})
        out["mean_norm"] = float(np.linalg.norm(mean))
        out["rms_norm"] = float(np.sqrt(np.trace(self.scatter) / self.n))
        out["n_rows"] = int(self.n)
        return out


def weight_matrices(module) -> dict[str, torch.Tensor]:
    out = {}
    for name, tensor in module.state_dict().items():
        if tensor.ndim != 2:
            continue
        if name.endswith("attention.qkv.weight"):
            for part, chunk in zip(("q", "k", "v"), tensor.chunk(3, dim=0)):
                out[name.replace("qkv.weight", f"{part}.weight")] = chunk
        else:
            out[name] = tensor
    return out


def weight_metrics(module, init_state: dict[str, torch.Tensor] | None) -> dict[str, object]:
    per_matrix = {}
    for name, w in weight_matrices(module).items():
        s = torch.linalg.svdvals(w.double()).numpy()
        m = spectrum_metrics(s)
        m["frobenius_norm"] = float(np.linalg.norm(s))
        per_matrix[name] = m
    state = module.state_dict()
    encoder = {k: v for k, v in state.items() if k.startswith("encoder.")}
    total_sq = sum(float((v.double() ** 2).sum()) for v in state.values())
    enc_sq = sum(float((v.double() ** 2).sum()) for v in encoder.values())
    out: dict[str, object] = {
        "param_l2_norm": float(np.sqrt(total_sq)),
        "encoder_l2_norm": float(np.sqrt(enc_sq)),
        "encoder_mean_erank": float(np.mean([m["erank"] for n, m in per_matrix.items() if n.startswith("encoder.")])),
        "per_matrix": per_matrix,
    }
    if init_state is not None:
        shared = [k for k in encoder if k in init_state and init_state[k].shape == encoder[k].shape]
        dist_sq = sum(float(((encoder[k].double() - init_state[k].double()) ** 2).sum()) for k in shared)
        init_sq = sum(float((init_state[k].double() ** 2).sum()) for k in shared)
        out["encoder_distance_from_init"] = float(np.sqrt(dist_sq))
        out["encoder_relative_distance_from_init"] = float(np.sqrt(dist_sq / init_sq)) if init_sq > 0 else None
    return out


# ------------------------------------------------------------- checkpoints
def checkpoints(leaf: str) -> list[tuple[Path, int, str]]:
    base = CKPTS / leaf
    items = []
    if (base / "initial_checkpoint").is_dir():
        items.append((base / "initial_checkpoint", 0, "init"))
    for sub in ("log_spaced_checkpoints", "interval_checkpoints"):
        if not (base / sub).is_dir():
            continue
        for d in sorted((base / sub).iterdir()):
            m = re.match(r"iteration_(\d+)_steps_(\d+)", d.name)
            if m and d.is_dir():
                items.append((d, int(m.group(2)), f"iter {int(m.group(1))}"))
    final = next((base / "tune").glob("*/checkpoint_*"), None)
    curve_files = sorted((ROOT.parent / leaf / "results").glob("*/training_curves.jsonl"))
    curves = [json.loads(line) for line in curve_files[-1].read_text().splitlines() if line] if curve_files else []
    if final is not None and curves:
        last = curves[-1]
        items.append((final, int(last["steps"]), f"iter {last['iteration']} (final)"))
    if leaf.endswith("_continue"):
        # The continuation's initial checkpoint is the parent run's final checkpoint (restore point).
        parent_curves = sorted((ROOT.parent / leaf[: -len("_continue")] / "results").glob("*/training_curves.jsonl"))
        parent_last = json.loads(parent_curves[-1].read_text().splitlines()[-1])
        items = [(p, int(parent_last["steps"]) if lab == "init" else s, f"restore (iter {parent_last['iteration']})" if lab == "init" else lab)
                 for p, s, lab in items]
    seen, out = set(), []
    for path, steps, label in sorted(items, key=lambda t: t[1]):
        if steps not in seen:
            seen.add(steps)
            out.append((path, steps, label))
    return out


# ----------------------------------------------------------------- forward
@torch.inference_mode()
def analyze_activations(module, tokens: np.ndarray, beliefs: np.ndarray):
    n = len(tokens)
    t = torch.as_tensor(tokens[:, 1:EPISODE_LENGTH], dtype=torch.long)
    obs = torch.nn.functional.one_hot(t, num_classes=TOKEN_COUNT).float()
    obs = torch.cat([torch.zeros(n, 1, TOKEN_COUNT), obs], dim=1)  # BOS = zeros, len 127
    encoder = module.encoder
    stages = ["embed"] + [f"block_{i + 1}" for i in range(len(encoder.blocks))] + ["final_norm"]
    acc = {s: ScatterAccumulator(encoder.config.d_model) for s in stages}
    captured: dict[str, torch.Tensor] = {}
    handles = [
        encoder.blocks[0].register_forward_pre_hook(lambda mod, args: captured.__setitem__("embed", args[0])),
        *[
            blk.register_forward_hook(lambda mod, args, out, i=i: captured.__setitem__(f"block_{i + 1}", out))
            for i, blk in enumerate(encoder.blocks)
        ],
    ]
    aux = getattr(module, "next_token_aux_head", None)
    aux_nll, aux_rows = 0.0, 0
    next_tokens = torch.as_tensor(tokens[:, 1:], dtype=torch.long)  # (n, 127)
    try:
        for i in range(0, n, SUP_BATCH):
            captured.clear()
            residual = encoder.forward_complete_episode(obs[i:i + SUP_BATCH], apply_final_norm=False)
            normed = encoder.final_norm(residual)
            captured["final_norm"] = normed
            for s in stages:
                acc[s].add(captured[s].numpy())
            if aux is not None:
                logits = aux(normed)
                nll = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), next_tokens[i:i + SUP_BATCH].reshape(-1), reduction="sum")
                aux_nll += float(nll)
                aux_rows += logits.shape[0] * logits.shape[1]
    finally:
        for h in handles:
            h.remove()
    out = {s: acc[s].metrics() for s in stages}
    extras = {}
    if aux is not None and aux_rows:
        bayes_floor = float(-(np.log(next_token_distributions(beliefs, preset=PRESET)) * np.eye(2)[tokens[:, 1:]]).sum(-1).mean())
        extras["aux_head_nll"] = aux_nll / aux_rows
        extras["aux_head_excess_nll"] = aux_nll / aux_rows - bayes_floor
    return out, extras


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaves", nargs="*", default=list(LEAVES))
    parser.add_argument("--limit", type=int, default=None, help="only the first N checkpoints per leaf (smoke)")
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    test_tokens = sample(PRESET, N_TEST, SEEDS["test"])
    test_beliefs, _ = targets(test_tokens, PRESET)

    init_state = None
    init_path = CKPTS / "rl_b10" / "initial_checkpoint"
    if init_path.is_dir():
        init_state = {k: v.clone() for k, v in load_module_only(init_path).state_dict().items() if k.startswith("encoder.")}

    runs, skipped = {}, []
    for leaf in args.leaves:
        rows = []
        for path, env_steps, label in checkpoints(leaf)[: args.limit]:
            try:
                module = load_module_only(path).eval()
                activations, extras = analyze_activations(module, test_tokens, test_beliefs)
                weights = weight_metrics(module, init_state)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                skipped.append({"leaf": leaf, "label": label, "env_steps": env_steps, "error": repr(exc)})
                continue
            probe = activations["block_4"]
            row = {"label": label, "env_steps": env_steps, "path": str(path.relative_to(CKPTS)),
                   "activations": activations, "weights": weights, **extras}
            rows.append(row)
            print(f"{leaf} {label:>18} {env_steps:>12,d}  erank(block_4)={probe['centered_erank']:.2f} "
                  f"PR={probe['centered_participation_ratio']:.2f} n99={probe['centered_n99']} "
                  f"|theta|={weights['encoder_l2_norm']:.1f} "
                  f"dist0={weights.get('encoder_distance_from_init', float('nan')):.1f} "
                  f"W-erank={weights['encoder_mean_erank']:.1f}"
                  + (f" auxXS={extras['aux_head_excess_nll']:.2e}" if extras else ""), flush=True)
        runs[leaf] = rows
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "runs": runs, "skipped": skipped,
            "settings": {
                "preset": PRESET, "n_test": N_TEST, "positions": EPISODE_LENGTH, "seed": SEEDS["test"],
                "activations": "held-out probe sequences (same as probe_curves); scatter-matrix SVD over all "
                               "sequences x positions; centered_* subtracts the mean activation, uncentered_* does not",
                "erank": "Roy & Vetterli 2007: exp(entropy of L1-normalised singular values)",
                "participation_ratio": "(sum lambda)^2 / sum lambda^2 over covariance eigenvalues",
                "n90/n99": "principal components needed for 90% / 99% of variance",
                "weights": "per 2-D parameter (qkv split into q/k/v); encoder_* aggregates encoder.* params; "
                           "distance_from_init is the L2 distance of encoder.* from rl_b10/initial_checkpoint",
                "aux_head_excess_nll": "rl_b10_aux_ce only: held-out next-token NLL of the aux head minus the Bayes floor",
            },
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
