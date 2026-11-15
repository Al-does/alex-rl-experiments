"""Diagnostics for the early rise of within-class b1-b2 resolution in the v2 symmetry-dose runs.

All measurements replay saved checkpoints on the final policy's cached fit/test histories (the
``probe.py`` trajectory protocol), so inputs, classes and targets are identical across checkpoints
and only the weights change. Per checkpoint this scores, inside the final greedy-action classes:

* b1-b2 (delta), b3 (rho) and the conditional skew (b1-b2)/(b1+b2), which removes the 1-b3 factor
  from delta, from the residual stream, its policy-head-blind part and the logits;
* a degree-2 polynomial probe on the leading principal components (is the information already
  present nonlinearly at initialization?) and a ridge-free probe (is the rise a conditioning effect?);
* how far back token identity is linearly readable: A-versus-B and non-C-versus-C at fixed lags;
* the input-embedding separation of tokens A and B relative to their separation from C.

Per run it also scores affine probes on the raw one-hot input window of several lengths.
Probe R2 is descriptive affine accessibility, not evidence of causal use.
"""
import argparse
import json
from pathlib import Path

import jax
import numpy as np

from experiments.within_class_belief_probes_2026_11.metrics import null_basis
from experiments.within_class_belief_probes_2026_11.temp15 import histories, restore
from experiments.within_class_belief_probes_2026_11.temp15_analyze import replay
from .probe import COUNTS, RAW, completed_runs

HERE = Path(__file__).resolve().parent
WARMUP = 64
LAGS = (0, 1, 2, 4, 8, 16, 24, 32, 48)
WINDOWS = (1, 2, 4, 8, 16, 32, 64)
SKEW_FLOOR = .1  # the conditional skew (b1-b2)/(b1+b2) is scored only where b1+b2 exceeds this
STEPS = (0, 20e6, 60e6, 100e6, 160e6, 220e6, 300e6, 500e6, 1e9, 2e9)
SELECTED = {("variant_1_transformer", 42), ("variant_2_transformer", 42), ("variant_3_transformer", 42),
            ("lam_p100", 43), ("lam_p100", 44), ("lam_m100", 43), ("lam_m100", 44),
            ("lam_p075", 43), ("kappa_050", 43)}


def flat(array, count):
    array = np.asarray(array)[:, WARMUP:]
    return array.reshape(-1, *array.shape[2:])[:count]


def lagged(obs, lag, count):
    """Rows of ``obs`` shifted back by ``lag`` steps, aligned with the post-warmup probe rows."""
    return np.asarray(obs)[:, WARMUP - lag:obs.shape[1] - lag].reshape(-1, obs.shape[-1])[:count]


def r2(fit_x, fit_y, test_x, test_y, classes, ridge=1e-6):
    """Per-column within-class held-out R2 of class-specific affine ridge probes (normal equations)."""
    sse, sst = np.zeros(fit_y.shape[1]), np.zeros(fit_y.shape[1])
    for k in np.unique(classes[1]):
        fit, held = classes[0] == k, classes[1] == k
        mean_x, mean_y = fit_x[fit].mean(0), fit_y[fit].mean(0)
        x = fit_x[fit] - mean_x
        gram = x.T @ x + ridge * np.eye(x.shape[1])
        weight = np.linalg.lstsq(gram, x.T @ (fit_y[fit] - mean_y), rcond=None)[0]
        y = test_y[held]
        sse += ((y - (test_x[held] - mean_x) @ weight - mean_y) ** 2).sum(0)
        sst += ((y - y.mean(0)) ** 2).sum(0)
    return [float(v) for v in 1 - sse / sst]


def named(names, values):
    return dict(zip(names, values))


def quadratic(fit, test, components=24):
    mean = fit.mean(0)
    _, _, right = np.linalg.svd(fit - mean, full_matrices=False)
    out = []
    for x in (fit, test):
        z = (x - mean) @ right[:components].T
        i, j = np.triu_indices(components)
        out.append(np.concatenate([x, z[:, i] * z[:, j]], axis=1))
    return out


def analyze(run, lam, kappa):
    raw = RAW / run.name
    points = json.loads((run / "condition_summary.json").read_text())["checkpoint_reports"]
    steps = np.array([p["checkpoint_step"] for p in points])
    final = int(steps[-1])
    spec, process, params, _ = restore(run, final, raw_root=raw)
    data = histories(run, final, spec, process, params, raw_root=raw)
    classes = [flat(d["action"], n) for d, n in zip(data, COUNTS)]
    belief = [flat(d["belief"], n).astype(float) for d, n in zip(data, COUNTS)]
    names = ("delta", "rho")
    targets = [np.concatenate([b[:, 0:1] - b[:, 1:2], b[:, 2:3]], axis=1) for b in belief]
    past = {lag: [lagged(d["obs"], lag, n) for d, n in zip(data, COUNTS)] for lag in LAGS}
    token_names = [f"{kind}_lag{lag}" for lag in LAGS for kind in ("a_minus_b", "non_c")]
    tokens = [np.concatenate([np.stack([past[lag][i][:, 0] - past[lag][i][:, 1], 1 - past[lag][i][:, 2]], axis=1)
                              for lag in LAGS], axis=1).astype(float) for i in range(2)]
    keep = [b[:, 2] < 1 - SKEW_FLOOR for b in belief]
    skew = [((b[:, 0] - b[:, 1]) / (b[:, 0] + b[:, 1]))[m, None] for b, m in zip(belief, keep)]
    kept_classes = [c[m] for c, m in zip(classes, keep)]

    def skew_r2(fit, test):
        return r2(fit[keep[0]], skew[0], test[keep[1]], skew[1], kept_classes)[0]

    window = {}
    for k in WINDOWS:
        x = [np.concatenate([lagged(d["obs"], lag, n) for lag in range(k)], axis=1).astype(float)
             for d, n in zip(data, COUNTS)]
        window[str(k)] = named(names, r2(x[0], targets[0], x[1], targets[1], classes)) | {"skew": skew_r2(*x)}
    out = []
    for index in sorted({int(np.abs(steps - target).argmin()) for target in STEPS}):
        step = int(steps[index])
        _, _, checkpoint, _ = restore(run, step, raw_root=raw)
        h = [flat(replay(spec, checkpoint, d["obs"]), n).astype(float) for d, n in zip(data, COUNTS)]
        weight = np.asarray(checkpoint["policy"]["w"], dtype=float)
        bias = np.asarray(checkpoint["policy"]["b"], dtype=float)
        basis = null_basis(weight.T)
        features = {"full_h": h, "head_blind": [x @ basis for x in h], "logits": [x @ weight + bias for x in h]}
        row = {"step": step, "reward_per_step": points[index]["policy"]["reward_per_step"],
               "action_fractions": points[index]["policy"]["action_fractions"]}
        for feature, (fit, test) in features.items():
            row[feature] = named(names, r2(fit, targets[0], test, targets[1], classes)) | {"skew": skew_r2(fit, test)}
        row["full_h_no_ridge"] = named(names, r2(h[0], targets[0], h[1], targets[1], classes, ridge=0.))
        poly = quadratic(*h)
        row["quadratic"] = named(names, r2(poly[0], targets[0], poly[1], targets[1], classes)) | {"skew": skew_r2(*poly)}
        row["token_lags"] = named(token_names, r2(h[0], tokens[0], h[1], tokens[1], classes))
        embed = np.asarray(checkpoint["input"]["w"], dtype=float)[:3]
        row["embedding_a_b_over_ab_c"] = float(np.linalg.norm(embed[0] - embed[1])
                                               / np.linalg.norm((embed[0] + embed[1]) / 2 - embed[2]))
        centred = np.concatenate([h[1][classes[1] == k] - h[1][classes[1] == k].mean(0) for k in np.unique(classes[1])])
        spectrum = np.linalg.svd(centred, compute_uv=False) ** 2
        row["within_class_participation_ratio"] = float(spectrum.sum() ** 2 / (spectrum ** 2).sum())
        out.append(row)
        jax.clear_caches()
    return {"run_id": run.name, "arm": run.parent.parent.name, "lam": lam, "kappa": kappa,
            "seed": json.loads((run / "resolved_recipe.json").read_text())["seed"],
            "test_class_counts": np.bincount(classes[1], minlength=3).tolist(),
            "skew_test_rows": int(keep[1].sum()),
            "raw_window_affine": window, "points": out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--collect", action="store_true", help="write results/early_rise.json from finished runs")
    args = parser.parse_args()
    selected = [(run, lam, kappa) for run, lam, kappa in completed_runs()
                if (run.parent.parent.name, json.loads((run / "resolved_recipe.json").read_text())["seed"]) in SELECTED]
    if args.collect:
        rows = [json.loads((RAW / run.name / "early_rise.json").read_text()) for run, _, _ in selected]
        (HERE / "results/early_rise.json").write_text(json.dumps(rows, indent=1, allow_nan=False) + "\n")
        return
    for index, (run, lam, kappa) in enumerate(selected):
        if index % args.shards != args.shard:
            continue
        output = RAW / run.name / "early_rise.json"
        if not output.exists():
            output.write_text(json.dumps(analyze(run, lam, kappa), indent=2, allow_nan=False) + "\n")
        print("DONE", run.parent.parent.name, run.name, flush=True)


if __name__ == "__main__":
    main()
