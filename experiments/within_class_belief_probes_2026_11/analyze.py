"""Offline conditional accessibility analysis of frozen passive predictors."""

import json
import hashlib
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.checkpoints import load_module_only
from analysis.rollouts import _episode_seed, _ROLLOUT_STREAM_KEYS
from envs.hmm import HMMEnv
from harness.seeding import named_seed_sequences
from experiments.mess3_token_guess_cycle_2.shared import ENV_CONFIG
from experiments.mess3_token_guess_cycle_2.analysis import _STREAM_KEYS
from .gate import ROOT, SOURCE, RUNS
from .metrics import affine, score, refit, null_basis, scalar_targets, bootstrap


def control_windows(data, seed):
    """Reconstruct complete original episodes, then perturb token sequences."""
    streams = named_seed_sequences(seed, _ROLLOUT_STREAM_KEYS)
    rng = np.random.default_rng(1267)
    histories = {}
    env = HMMEnv(ENV_CONFIG)
    try:
        for cluster in np.unique(data["episode"]):
            env_index, episode_index = int(cluster % 16), int(cluster // 16)
            obs, _ = env.reset(seed=_episode_seed(streams["episode_seeds"], env_index, episode_index))
            history = [obs]
            while True:
                obs, _, terminated, truncated, _ = env.step(0)
                if terminated or truncated:
                    break
                history.append(obs)
            histories[cluster] = np.asarray(history)
    finally:
        env.close()
    pool = np.concatenate([h[1:] for h in histories.values()])
    controls = {}
    for mode in ("original", "history_shuffle", "random_input"):
        output = np.empty_like(data["window"])
        for cluster, history in histories.items():
            changed = history.copy()
            if mode == "history_shuffle":
                changed[1:] = rng.permutation(history[1:])
            elif mode == "random_input":
                changed[1:] = pool[rng.integers(len(pool), size=len(history)-1)]
            rows = np.flatnonzero(data["episode"] == cluster)
            positions = data["episode_step"][rows]
            output[rows] = changed[positions[:, None]+np.arange(-9, 1)]
        controls[mode] = output
    assert np.array_equal(controls.pop("original"), data["window"]), "history reconstruction mismatch"
    return controls


@torch.no_grad()
def encode(module, windows):
    values = []
    for start in range(0, len(windows), 1024):
        batch = torch.as_tensor(windows[start:start+1024], dtype=torch.float32)
        h = module.encoder(batch[:, :-1], torch.full((len(batch),), 9), batch[:, -1:])[:, 0]
        values.append(h.numpy())
    return np.concatenate(values).astype(np.float64)


def summarize(y, predictions, classes, episodes, uncertainty=False):
    valid = np.isfinite(predictions).all(1)
    result = score(y[valid], predictions[valid], classes[valid])
    result["n_undefined_predictions"] = int((~valid).sum())
    result["classes"] = {str(k): score(y[(classes == k) & valid], predictions[(classes == k) & valid]) for k in range(3)}
    if uncertainty:
        result["bootstrap"] = bootstrap(y[valid], predictions[valid], classes[valid], episodes[valid])
    return result


def analyze_point(check, reference_beliefs):
    arm, steps = check["arm"], check["steps"]
    base = ROOT / "artifacts" / RUNS[arm]
    train, test = [dict(np.load(base / f"{steps}_{split}.npz")) for split in ("fit", "test")]
    for i, data in enumerate((train, test)):
        if reference_beliefs[i] is None:
            reference_beliefs[i] = data["b"]
        assert np.array_equal(reference_beliefs[i], data["b"]), "passive targets changed"
        assert np.array_equal(data["logits"].argmax(1), data["action"].reshape(-1))
    module_dir = (base / check["hashes"][0]["path"]).parent
    module = load_module_only(module_dir).to("cpu").eval()
    head = module.heads.policy
    weight, bias = head.weight.detach().numpy().astype(np.float64), head.bias.detach().numpy().astype(np.float64)
    np.savez(base / f"{steps}_head.npz", weight=weight, bias=bias)
    basis = null_basis(weight)
    assert np.max(np.abs((weight[:2]-weight[2]) @ basis)) < 1e-10
    for data in (train, test):
        assert np.max(np.abs(data["h"] @ weight.T + bias-data["logits"])) < 1e-5
    streams = named_seed_sequences(45, _STREAM_KEYS)
    perturbed = [control_windows(train, streams["probe_train"]), control_windows(test, streams["probe_test"])]
    features = {
        "full_h": (train["h"], test["h"]),
        "head_blind": (train["h"] @ basis, test["h"] @ basis),
        "logits": (train["logits"], test["logits"]),
        "window": (train["window"].reshape(len(train["b"]), -1), test["window"].reshape(len(test["b"]), -1)),
    }
    for mode in ("history_shuffle", "random_input"):
        cache = base / f"{steps}_{mode}.npz"
        if cache.exists():
            with np.load(cache) as saved:
                features[mode] = (saved["fit"], saved["test"])
        else:
            values = tuple(encode(module, d[mode]) for d in perturbed)
            features[mode] = values
            np.savez_compressed(cache, fit=values[0], test=values[1])
    uncertainty = steps in (0, 662029, 626197)
    result = dict(arm=arm, steps=steps, head_null_dimension=basis.shape[1],
                  checkpoint_hashes=check["hashes"], n_fit=len(train["b"]), n_test=len(test["b"]),
                  global_r2=check["reproduced"], classes={})
    for class_name in ("bayes", "network"):
        fc, tc = [(data["b"] if class_name == "bayes" else data["logits"]).argmax(1) for data in (train, test)]
        class_result = {"fit_counts": np.bincount(fc, minlength=3).tolist(), "test_counts": np.bincount(tc, minlength=3).tolist()}
        predictions = {}
        for name, (fit, held) in features.items():
            outcome = {}
            if name not in ("history_shuffle", "random_input"):
                pred = affine(fit, train["b"], held)
                outcome["global_probe_global_score"] = score(test["b"], pred)
                if uncertainty and name in ("head_blind", "logits"):
                    outcome["global_probe_global_score"]["bootstrap"] = bootstrap(test["b"], pred, np.zeros(len(pred), dtype=int), test["episode"])
                outcome["global_probe_class_score"] = summarize(test["b"], pred, tc, test["episode"], uncertainty and name in ("head_blind", "logits"))
            pred = refit(fit, train["b"], held, fc, tc)
            predictions[name] = pred
            outcome["within_refit"] = summarize(test["b"], pred, tc, test["episode"], uncertainty and name in ("full_h", "head_blind", "logits", "window"))
            class_result[name] = outcome
        permuted = refit(train["h"], train["b"], test["h"], fc, tc, permute_seed=811)
        class_result["label_permutation"] = summarize(test["b"], permuted, tc, test["episode"])
        indicator = affine(np.eye(3)[fc], train["b"], np.eye(3)[tc])
        class_result["class_indicator_global"] = score(test["b"], indicator)
        if class_name == "bayes":
            fit_scalars, test_scalars = scalar_targets(train["b"]), scalar_targets(test["b"])
            margin = test_scalars[0][:, 0]
            bin_index = np.digitize(margin, [.05, .2, .5], right=False)
            class_result["margin_bins"] = []
            for index, label in enumerate(("[0,.05)", "[.05,.2)", "[.2,.5)", "[.5,1]")):
                mask = bin_index == index
                counts = np.bincount(tc[mask], minlength=3)
                bins = {"bin": label, "too_few_rows": bool(mask.sum() < 50), "class_counts": counts.tolist(), "class_too_few_rows": (counts < 50).tolist()}
                for name in ("full_h", "head_blind", "window"):
                    bins[name] = summarize(test["b"][mask], predictions[name][mask], tc[mask], test["episode"][mask])
                class_result["margin_bins"].append(bins)
            class_result["scalar_probes"] = {}
            for i, target in enumerate(("margin", "runner_up_split")):
                scalar = {}
                for name in ("full_h", "head_blind", "window"):
                    fit, held = features[name]
                    pred = refit(fit, fit_scalars[i], held, fc, tc)
                    scalar[name] = summarize(test_scalars[i], pred, tc, test["episode"], uncertainty)
                class_result["scalar_probes"][target] = scalar
        result["classes"][class_name] = class_result
    return result


def render(report):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    lines = ["# Passive within-class belief decoding", "", "R² denominators are centred within test class; pooled R² = 1 − pooled SSE / pooled within-class SST.", "", "| Objective | Steps | Global | Within Bayes class | Head-blind | Logits | Window | Margin | Split |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, color in (("ppo", "#2266aa"), ("supervised_ce", "#cc6633")):
        points = [p for p in report["checkpoints"] if p["arm"] == arm]
        objective = "PPO" if arm == "ppo" else "CE"
        for feature, style in (("full_h", "-o"), ("history_shuffle", ":"), ("random_input", "--"), ("label_permutation", ":x"), ("window", "-.")):
            values = [p["classes"]["bayes"][feature] for p in points]
            axes[0].plot([p["steps"] for p in points], [v.get("within_refit", v)["r2"] for v in values], style, color=color, label=f"{objective}: {feature.replace('_', ' ')}")
        init = points[0]["classes"]["bayes"]["full_h"]["within_refit"]["r2"]
        if arm == "ppo":
            axes[0].axhline(init, color="black", alpha=.5, linewidth=.7, label="Shared initialization level")
        paper = next(p for p in points if p["steps"] in (662029, 626197))
        for label, point, style in (("init", points[0], ":"), ("paper", paper, "-"), ("final", points[-1], "--")):
            bins = point["classes"]["bayes"]["margin_bins"]
            axes[1].plot(range(4), [b["full_h"]["r2"] for b in bins], style+"o", color=color, label=f"{objective} {label}")
            b = point["classes"]["bayes"]
            values = [point["global_r2"]]+[b[k]["within_refit"]["r2"] for k in ("full_h", "head_blind", "logits", "window")]+[b["scalar_probes"][k]["full_h"]["r2"] for k in ("margin", "runner_up_split")]
            lines.append(f"| {arm} {label} | {point['steps']} | "+" | ".join(f"{v:.6f}" for v in values)+" |")
    axes[0].set(xlabel="Training tokens sampled (environment steps)", ylabel="Pooled within-Bayes-class R²")
    axes[1].set(xticks=range(4), xticklabels=["<.05", ".05–.2", ".2–.5", "≥.5"], xlabel="Bayes margin bin", ylabel="Within-class-refit R² (class-centred in bin)")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "passive_within_class.png", dpi=180)
    plt.close(fig)
    lines += ["", "One training seed; episode-bootstrap intervals condition on frozen probes and are not training-seed uncertainty. Input controls preserve original targets/classes, refit probes separately, and alter histories rather than activations. All decoding is affine accessibility, not causal use. The 62-dimensional head-blind basis removes logit-difference directions but retains any common-logit-shift direction.", "", "Margin bins use half-open boundaries, with the final bin including 0.5. Flags mark fewer than 50 total rows; per-class counts are included. Undefined scores are null, not zero. See JSON protocol and provenance; raw arrays/checkpoints are ignored under artifacts/."]
    (ROOT / "results" / "README.md").write_text("\n".join(lines)+"\n")


def main():
    torch.set_num_threads(2)
    gate = json.loads((ROOT / "results" / "reproduction_gate.json").read_text())
    expected = sum(len(json.loads((SOURCE / arm / "results" / run / "checkpoint_probe_curve.json").read_text())["checkpoints"]) for arm, run in RUNS.items())
    assert len(gate["checks"]) == expected and all(abs(c["delta"]) <= 1e-6 for c in gate["checks"])
    report = {"provenance": gate, "source_runs": RUNS, "protocol": {
        "fit_rows": 60000, "test_rows": 80000, "ridge": 1e-6, "warmup": 64, "n_envs": 16,
        "seed": 45, "stream_keys": _STREAM_KEYS, "representation": "post-final-LN 64d", "cpu_only": True,
        "history_shuffle": "one permutation of all non-reset tokens within each complete original episode; recompute strict10 windows; original targets and classes",
        "random_input": "one replacement draw per non-reset position from empirical non-reset token pool in that split, preserving reset; original targets/classes",
        "controls_seed": 1267, "label_permutation_seed": 811, "control_repetitions": 1,
        "bootstrap": "1000 episode-cluster resamples; probes fixed; recompute test-class means; init and paper checkpoints",
        "pooled_r2": "1 - sum class SSE / sum class SST (test-class means)",
        "scalar_split": "b_j/(b_j+b_l), j<l are the two non-Bayes-winner coordinates",
        "minimum_bin_rows": 50, "null_space": "null of W[0]-W[2], W[1]-W[2], rank2, dimension62"}, "checkpoints": []}
    report["runtime"] = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}
    report["source_provenance"] = {}
    for arm, run in RUNS.items():
        directory = SOURCE / arm / "results" / run
        manifest = json.loads((directory / "run_manifest.json").read_text())
        report["source_provenance"][arm] = {
            "run_id": run, "git": manifest["git"], "remote_artifacts": manifest["remote_artifacts"],
            "resolved_recipe": json.loads((directory / "resolved_recipe.json").read_text()),
            "source_results_path": str(directory.relative_to(ROOT.parent.parent)),
            "curve_sha256": hashlib.sha256((directory / "checkpoint_probe_curve.json").read_bytes()).hexdigest(),
            "durability_manifest_sha256": hashlib.sha256((ROOT / "artifacts" / run / "durability_manifest.json").read_bytes()).hexdigest()}
    references = [None, None]
    for check in gate["checks"]:
        cache = ROOT / "artifacts" / f"analysis_v2_{check['arm']}_{check['steps']}.json"
        if cache.exists():
            result = json.loads(cache.read_text())
        else:
            result = analyze_point(check, references)
            cache.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        report["checkpoints"].append(result)
        print(f"analyzed {check['arm']} {check['steps']}: within={result['classes']['bayes']['full_h']['within_refit']['r2']:.9f}", flush=True)
    (ROOT / "results" / "passive_within_class.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    render(report)


if __name__ == "__main__":
    main()
