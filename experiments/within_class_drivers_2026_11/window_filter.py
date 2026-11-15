"""Is within-class b1-b2 a short-window token statistic? Probes against a truncated Bayes filter.

The truncated filter restarts from the uniform reset prior ``k`` steps back and applies the exact
token/action update over only the last ``k`` observations (``k=32`` is one attention band of the
4-layer transformer, whose total lookback is 128). On the final policy's fixed histories this
scores, inside the final greedy-action classes:

* how well the truncated filter itself reproduces the full filter (no network involved);
* affine probes from the residual stream of saved checkpoints for the full coordinate, its
  ``k=32`` truncation, and the remainder ``full - truncated`` that needs evidence older than 32 steps;
* the same remainder from features of the last 32 raw inputs plus the truncated belief and its
  squares, as the level reachable without any older evidence.

Probe R2 is descriptive affine accessibility, not evidence of causal use.
"""
import argparse
import json

import jax
import numpy as np

from experiments.within_class_belief_probes_2026_11.temp15 import histories, restore
from experiments.within_class_belief_probes_2026_11.temp15_analyze import replay
from .early_rise import SELECTED, STEPS, WARMUP, flat, lagged, named, r2
from .probe import COUNTS, HERE, RAW, completed_runs

WINDOWS = (8, 16, 32, 64, 128)
BAND = 32


def truncated_filter(obs, transitions, emission, window, prior, initial):
    """Belief at every step from only the last ``window`` observations.

    The filter starts from ``prior`` over the state ``window`` steps back and applies the exact
    update for each of the last ``window`` (token, previous action) pairs. Steps with fewer than
    ``window`` observations since reset use the exact reset belief, so they equal the full filter.
    """
    obs = np.asarray(obs, dtype=float)
    token, previous = obs[..., :3].argmax(-1), obs[..., 3:].argmax(-1)
    likelihood = np.asarray(emission, dtype=float).T[token]
    step = np.asarray(transitions, dtype=float)[previous] * likelihood[..., None, :]
    reset = np.asarray(initial, dtype=float) * likelihood[:, 0]
    reset /= reset.sum(-1, keepdims=True)
    episodes, length = token.shape
    belief = np.broadcast_to(np.asarray(prior, dtype=float), (episodes, length, 3)).copy()
    for back in range(window - 1, -1, -1):
        index = np.arange(length) - back
        valid = index >= 0
        update = np.einsum("eti,etij->etj", belief[:, valid], step[:, index[valid]])
        update /= update.sum(-1, keepdims=True)
        if back < length:
            update[:, 0] = reset  # index 0 is the reset step, which discards the prior
        belief[:, valid] = update
    return belief


def coordinates(belief):
    return np.stack([belief[..., 0] - belief[..., 1], belief[..., 2]], axis=-1)


def analyze(run, lam, kappa):
    raw = RAW / run.name
    recipe = json.loads((run / "resolved_recipe.json").read_text())
    points = json.loads((run / "condition_summary.json").read_text())["checkpoint_reports"]
    steps = np.array([p["checkpoint_step"] for p in points])
    final = int(steps[-1])
    spec, process, params, _ = restore(run, final, raw_root=raw)
    data = histories(run, final, spec, process, params, raw_root=raw)
    transitions, emission = np.asarray(recipe["process"]["transitions"]), np.asarray(recipe["process"]["emission"])
    initial = np.asarray(recipe["process"]["initial"])
    classes = [flat(d["action"], n) for d, n in zip(data, COUNTS)]
    full = [coordinates(flat(d["belief"], n).astype(float)) for d, n in zip(data, COUNTS)]
    occupancy = np.asarray(data[0]["belief"], dtype=float).reshape(-1, 3).mean(0)  # mean filtered belief
    length = data[1]["obs"].shape[1]
    exact = truncated_filter(data[1]["obs"], transitions, emission, length, initial, initial)
    reproduction = float(np.abs(flat(exact, COUNTS[1]) - flat(data[1]["belief"], COUNTS[1])).max())
    assert reproduction < 1e-4, reproduction
    names = ("delta", "rho")

    def within(error, reference):
        sse, sst = np.zeros(2), np.zeros(2)
        for k in np.unique(classes[1]):
            mask = classes[1] == k
            sse += (error[mask] ** 2).sum(0)
            sst += ((reference[mask] - reference[mask].mean(0)) ** 2).sum(0)
        return sse / sst

    filter_error, band = {}, {}
    for label, prior in (("uniform_prior", initial), ("occupancy_prior", occupancy)):
        filter_error[label] = {}
        for window in WINDOWS:
            beliefs = [truncated_filter(d["obs"], transitions, emission, window, prior, initial) for d in data]
            if window == BAND:
                band[label] = [flat(b, n) for b, n in zip(beliefs, COUNTS)]
            error = full[1] - coordinates(flat(beliefs[1], COUNTS[1]))
            filter_error[label][str(window)] = {
                "within_class": named(names, within(error, full[1]).tolist()),
                "global": named(names, ((error ** 2).sum(0) / ((full[1] - full[1].mean(0)) ** 2).sum(0)).tolist())}
    near = [coordinates(b) for b in band["occupancy_prior"]]
    far = [f - n for f, n in zip(full, near)]
    target_names = [f"{part}_{name}" for part in ("full", "near", "far") for name in names]
    targets = [np.concatenate([f, n, r], axis=1) for f, n, r in zip(full, near, far)]
    window_features = [np.concatenate(
        [lagged(d["obs"], lag, n) for lag in range(BAND)] + [b, b ** 2, b[:, :1] * b[:, 1:2]], axis=1).astype(float)
        for d, b, n in zip(data, band["occupancy_prior"], COUNTS)]
    out = []
    for index in sorted({int(np.abs(steps - target).argmin()) for target in STEPS}):
        step = int(steps[index])
        _, _, checkpoint, _ = restore(run, step, raw_root=raw)
        h = [flat(replay(spec, checkpoint, d["obs"]), n).astype(float) for d, n in zip(data, COUNTS)]
        out.append({"step": step, "reward_per_step": points[index]["policy"]["reward_per_step"],
                    "full_h": named(target_names, r2(h[0], targets[0], h[1], targets[1], classes))})
        jax.clear_caches()
    far_share = named(names, within(far[1] - far[1].mean(0), full[1]).tolist())
    return {"run_id": run.name, "arm": run.parent.parent.name, "lam": lam, "kappa": kappa, "seed": recipe["seed"],
            "band": BAND, "state_occupancy": occupancy.tolist(),
            "near_far_prior": "occupancy_prior", "full_filter_reproduction_max_abs": reproduction,
            "truncated_filter_one_minus_r2": filter_error,
            "far_variance_over_full_variance_within_class": far_share,
            "window_only_features": named(target_names, r2(window_features[0], targets[0], window_features[1],
                                                           targets[1], classes)),
            "points": out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--collect", action="store_true", help="write results/window_filter.json from finished runs")
    args = parser.parse_args()
    selected = [(run, lam, kappa) for run, lam, kappa in completed_runs()
                if (run.parent.parent.name, json.loads((run / "resolved_recipe.json").read_text())["seed"]) in SELECTED]
    if args.collect:
        rows = [json.loads((RAW / run.name / "window_filter.json").read_text()) for run, _, _ in selected]
        (HERE / "results/window_filter.json").write_text(json.dumps(rows, indent=1, allow_nan=False) + "\n")
        return
    for index, (run, lam, kappa) in enumerate(selected):
        if index % args.shards != args.shard:
            continue
        output = RAW / run.name / "window_filter.json"
        if not output.exists():
            output.write_text(json.dumps(analyze(run, lam, kappa), indent=2, allow_nan=False) + "\n")
        print("DONE", run.parent.parent.name, run.name, flush=True)


if __name__ == "__main__":
    main()
