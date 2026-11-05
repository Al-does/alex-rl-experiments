"""Offline checkpoint diagnostics; linear accessibility is not causal use."""

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from analysis.probes import fit_affine_probe, global_mse_metrics, probe_predict

from experiments.pusher_b_jax_2026_10 import model as tm
from experiments.mrrxor_grokking_jax_2026_11 import process as process


def fit_pca(depth=16):
    """Exact process weighting over all stationary length-depth histories."""
    b = process.stationary_distribution(process.transition_tensor())[None]
    weights = np.ones(1)
    edges = process.transition_tensor()
    for _ in range(depth):
        joint = np.einsum("bs,xst->bxt", b, edges)
        mass = joint.sum(-1)
        b = (joint / mass[..., None]).reshape(-1, 5)
        weights = (weights[:, None] * mass).reshape(-1)
    mean = weights @ b
    centered = b - mean
    values, vectors = np.linalg.eigh((centered * weights[:, None]).T @ centered)
    components = vectors[:, np.argsort(values)[::-1]][:, :4]
    pcs = centered @ components
    target = process.predictive(b)[:, 1]
    correlations = ((pcs * weights[:, None]).T @ (target - weights @ target)) ** 2
    correlations /= (pcs ** 2 * weights[:, None]).sum(0) * np.sum(weights * (target - weights @ target) ** 2)
    ntp = int(np.argmax(correlations))
    return {"mean": mean.tolist(), "components": components.tolist(),
            "eigenvalues": values[::-1][:4].tolist(), "ntp_pc_index": ntp + 1,
            "ntp_pc_predictive_r2": float(correlations[ntp]),
            "weighting": f"exact probabilities of all 2^{depth} stationary histories"}


@lru_cache(maxsize=8)
def _forward(spec):
    return jax.jit(lambda p, x: (tm.encode(spec, p, x),
                               tm.policy_value(p, tm.encode(spec, p, x))[0]))


def evaluate(params, pools, spec, *, features=False, batch_size=64):
    forward = _forward(spec)
    result, representations, odds = {}, {}, {}
    for split, raw in pools.items():
        embeddings, logits = [], []
        for start in range(0, len(raw), batch_size):
            h, z = forward(params, jnp.asarray(process.policy_inputs(raw[start:start + batch_size])))
            if features:
                embeddings.append(np.asarray(h).reshape(-1, spec.d_model))
            logits.append(np.asarray(z))
        z = np.concatenate(logits).astype(np.float64)
        logq = z - np.logaddexp(z[..., :1], z[..., 1:])
        q = np.exp(logq)
        bayes = process.predictive(process.beliefs(raw)[:, :-1])
        greedy = z.argmax(-1) == raw
        random = bayes.max(-1) < 0.6
        empirical_ce = -np.take_along_axis(logq, raw[..., None], -1)[..., 0].mean()
        floor = -np.take_along_axis(np.log(bayes), raw[..., None], -1)[..., 0].mean()
        result[split] = {
            "sampled_policy_mean_reward": float(np.take_along_axis(q, raw[..., None], -1).mean()),
            "greedy_accuracy": float(greedy.mean()),
            "random_bit_accuracy": float(greedy[random].mean()) if random.any() else None,
            "random_bit_count": int(random.sum()),
            "ce_nats": float(empirical_ce), "bayes_floor_nats": float(floor),
            "excess_ce_nats": float(empirical_ce - floor),
            "bayes_to_policy_kl_nats": float((bayes * (np.log(bayes) - logq)).sum(-1).mean()),
        }
        if features:
            representations[split] = np.concatenate(embeddings)
            odds[split] = (z[..., 1] - z[..., 0]).reshape(-1, 1)
    return result, representations, odds


def normalized_error(x, y, test_x, test_y, ridge=1e-6):
    w, b = fit_affine_probe(x, y, ridge=ridge)
    return float(global_mse_metrics(probe_predict(w, b, test_x), test_y)["global_mse_ratio"])


def load_params(path, spec):
    template = tm.init_params(spec, jax.random.PRNGKey(0), actor_critic=True)
    with np.load(path, allow_pickle=False) as archive:
        return jax.tree_util.tree_map_with_path(
            lambda k, _: jnp.asarray(archive[jax.tree_util.keystr(k)]), template)


def analyze(run_dir, output):
    root, output = Path(run_dir), Path(output)
    recipe = json.loads((root / "recipe.json").read_text())
    spec = tm.ModelSpec(**recipe["model"])
    with np.load(root / "pools.npz", allow_pickle=False) as archive:
        pools = {k: archive[k] for k in archive.files}
    pca = fit_pca()
    b = {k: process.beliefs(v)[:, :-1].reshape(-1, 5) for k, v in pools.items()}
    pcs = {k: (v - pca["mean"]) @ np.asarray(pca["components"]) for k, v in b.items()}
    ntp = pca["ntp_pc_index"] - 1
    records = []
    for entry in json.loads((root / "checkpoint_index.json").read_text()):
        path = root / entry["path"]
        params = load_params(path, spec)
        metrics, h, odds = evaluate(params, pools, spec, features=True)
        probes = {"belief": normalized_error(h["train"], b["train"], h["heldout"], b["heldout"])}
        for i in range(4):
            probes[f"pc{i + 1}"] = normalized_error(h["train"], pcs["train"][:, i:i+1],
                                                    h["heldout"], pcs["heldout"][:, i:i+1])
        logodds = {}
        for name, target in (("ntp_pc", {k: v[:, ntp:ntp+1] for k, v in pcs.items()}), ("belief", b)):
            scores = {k: 1 - normalized_error(target["train"], odds["train"], target[k], odds[k])
                      for k in pools}
            logodds[name] = {**scores, "train_minus_heldout": scores["train"] - scores["heldout"]}
        records.append({**entry, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "metrics": metrics, "probe_1_minus_r2": probes, "policy_logodds_r2": logodds})
    report = {"recipe": recipe, "pca": pca, "representation": "post-final-RMSNorm residual stream",
              "probe": "affine ridge 1e-6, fit train sequences, score heldout sequences; all prefixes incl BOS",
              "checkpoints": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    plot(report, output.with_suffix(".png"))
    print(json.dumps({"output": str(output), "checkpoint_count": len(records),
                      "ntp_pc_index": pca["ntp_pc_index"], "ntp_pc_predictive_r2": pca["ntp_pc_predictive_r2"],
                      "final_metrics": records[-1]["metrics"]}, indent=2))
    return report


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["checkpoints"]
    x = [max(1, r["env_steps"]) for r in rows]
    fig, (ax, accuracy) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    other = ax.twinx()
    for key in ("belief", f"pc{report['pca']['ntp_pc_index']}"):
        ax.plot(x, [max(1e-12, r["probe_1_minus_r2"][key]) for r in rows], label=key)
    for split in ("train", "heldout"):
        other.plot(x, [r["metrics"][split]["excess_ce_nats"] for r in rows], "--", label=f"{split} excess CE")
        accuracy.plot(x, [r["metrics"][split]["random_bit_accuracy"] for r in rows], label=split)
    ax.set(yscale="log", ylabel="Held-out probe 1−R²")
    other.set_ylabel("Excess CE (nats; may be negative on finite data)")
    accuracy.set(xscale="log", xlabel="Environment steps (initialization shown at 1)", ylabel="Random-bit greedy accuracy")
    accuracy.axhline(0.5, color="gray", linestyle=":")
    ax.legend(loc="upper left")
    other.legend(loc="upper right")
    accuracy.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Artifacts run directory containing recipe.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze(args.run_dir, args.output)
