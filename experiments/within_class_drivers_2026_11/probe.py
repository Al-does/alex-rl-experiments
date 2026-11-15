"""Matched-initialization within-class probes for the v2 symmetry-dose runs.

For each completed run this restores the hash-verified initial and final checkpoints from B2,
rolls out the final greedy policy on fixed fit/test streams, replays the initialized network on
those exact histories, and scores class-specific affine probes on held-out rows with
class-centred denominators (the ``within_class_belief_probes_2026_11`` protocol). Beyond the
belief coordinates it probes the belief-MDP action-value contrasts Q(b, Tilt X) - Q(b, Noop) and
Q(b, Tilt Y) - Q(b, Noop), which is the quantity a policy gradient is driven by.

Probe R2 is descriptive affine accessibility, not evidence of causal use.
"""
import argparse
import json
from pathlib import Path

import jax
import numpy as np

from experiments.within_class_belief_probes_2026_11.metrics import bootstrap, null_basis, refit, score
from experiments.within_class_belief_probes_2026_11.temp15 import histories, restore, rows
from experiments.within_class_belief_probes_2026_11.temp15_analyze import perturb, replay
from .theory import emission_for, kernels_for, solve

HERE = Path(__file__).resolve().parent
RAW = HERE / "artifacts/probe"
DOSE = HERE.parent / "belief_quotient_v2_symmetry_dose_2026_11"
BASELINE = HERE.parent / "belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11"
COUNTS = (60000, 80000)
TARGETS = {"belief": slice(0, 3), "delta": slice(3, 4), "rho": slice(4, 5), "q_contrast": slice(5, 7)}
FEATURES = ("full_h", "head_blind", "logits")
TRAJECTORY_STEPS = (0, 20e6, 40e6, 60e6, 100e6, 160e6, 220e6, 300e6, 400e6, 500e6, 700e6, 1e9, 1.5e9, 2e9)


def completed_runs():
    """(run directory, lam, kappa) for dose runs plus the PR #219 seed-42 epsilon-0 endpoints."""
    found = []
    for run in sorted(DOSE.glob("*/results/*")):
        if (run / "run_manifest.json").exists() and (run / "condition_summary.json").exists():
            recipe = json.loads((run / "resolved_recipe.json").read_text())
            found.append((run, recipe["lam"], recipe["kappa"]))
    for variant, lam in ((2, 1.), (3, -1.)):
        for run in sorted(BASELINE.glob(f"variant_{variant}_transformer/results/*")):
            if (run / "run_manifest.json").exists():
                found.append((run, lam, 0.))
    return found


def validate(recipe, lam, kappa):
    training, process = recipe["training"], recipe["process"]
    if (training["temperature"] != 1.5 or training["entropy_schedule"] != [[0, 0.]]
            or training["exploration_epsilon"] != 0 or training["total_env_steps"] != 2_000_000_000
            or recipe["model"]["kind"] != "transformer"
            or not np.allclose(process["transitions"], kernels_for(lam))
            or not np.allclose(process["emission"], emission_for(kappa))
            or not np.allclose(process["initial"], [1 / 3] * 3)):
        raise ValueError("run is not the 2B-step T=1.5 entropy-0 epsilon-0 v2 transformer recipe for this dose")


def encode(spec, params, data, embeddings, classes, targets):
    h = [rows({"h": value}, "h", n) for value, n in zip(embeddings, COUNTS)]
    weight = np.asarray(params["policy"]["w"], dtype=float)
    bias = np.asarray(params["policy"]["b"], dtype=float)
    basis = null_basis(weight.T)
    features = {"full_h": h, "head_blind": [x @ basis for x in h], "logits": [x @ weight + bias for x in h]}
    predictions = {name: refit(fit, targets[0], test, *classes) for name, (fit, test) in features.items()}
    for index, mode in enumerate(("history_shuffle", "random_input")):
        changed = [perturb(d["obs"], mode, 20261106 + 10 * index + i) for i, d in enumerate(data)]
        values = [rows({"h": np.asarray(replay(spec, params, obs))}, "h", n) for obs, n in zip(changed, COUNTS)]
        predictions[mode] = refit(values[0], targets[0], values[1], *classes)
    predictions["label_permutation"] = refit(h[0], targets[0], h[1], *classes, permute_seed=853)
    return predictions, features["logits"][1]


def summarize(y, prediction, classes):
    valid = np.isfinite(prediction).all(1)
    return {name: score(y[valid][:, sl], prediction[valid][:, sl], classes[valid])["r2"] for name, sl in TARGETS.items()}


def margin(values):
    ordered = np.sort(values, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def rank_correlation(x, y, classes):
    """Class-size-weighted Spearman correlation computed inside each class."""
    total, weight = 0., 0
    for k in np.unique(classes):
        mask = classes == k
        if mask.sum() < 3 or np.ptp(x[mask]) == 0 or np.ptp(y[mask]) == 0:
            continue
        a, b = (np.argsort(np.argsort(v[mask])).astype(float) for v in (x, y))
        total += mask.sum() * float(np.corrcoef(a, b)[0, 1])
        weight += mask.sum()
    return total / weight if weight else None


def boundary_bins(y, trained, initial, q_margin, classes, bins=3):
    """Squared probe error by within-class tercile of the optimal action-value margin."""
    out = []
    cell = np.zeros(len(y), dtype=int)
    for k in np.unique(classes):
        mask = classes == k
        edges = np.quantile(q_margin[mask], np.linspace(0, 1, bins + 1)[1:-1])
        cell[mask] = np.searchsorted(edges, q_margin[mask])
    valid = np.isfinite(trained).all(1) & np.isfinite(initial).all(1)
    for j in range(bins):
        mask = (cell == j) & valid
        out.append({"mse_trained": float(((y[mask] - trained[mask]) ** 2).mean()),
                    "mse_matched_init": float(((y[mask] - initial[mask]) ** 2).mean()),
                    "mean_q_margin": float(q_margin[mask].mean())})
    return out


def analyze_run(run, lam, kappa, *, bootstrap_draws=2000):
    recipe = json.loads((run / "resolved_recipe.json").read_text())
    validate(recipe, lam, kappa)
    raw = RAW / run.name
    output = raw / "analysis.json"
    if output.exists():
        return json.loads(output.read_text())
    points = json.loads((run / "condition_summary.json").read_text())["checkpoint_reports"]
    step = points[-1]["checkpoint_step"]
    spec, process, init, init_provenance = restore(run, 0, raw_root=raw)
    _, _, params, provenance = restore(run, step, raw_root=raw)
    data = histories(run, step, spec, process, params, raw_root=raw)
    classes = [rows(d, "action", n) for d, n in zip(data, COUNTS)]
    belief = [rows(d, "belief", n).astype(float) for d, n in zip(data, COUNTS)]
    q = solve(kernels_for(lam), emission_for(kappa))
    values = [q(b) for b in belief]
    targets = [np.concatenate([b, b[:, 0:1] - b[:, 1:2], b[:, 2:3], v[:, 1:] - v[:, :1]], axis=1)
               for b, v in zip(belief, values)]
    y, test_class = targets[1], classes[1]
    own = np.asarray(replay(spec, params, data[1]["obs"]))
    replay_error = float(np.max(np.abs(own - data[1]["embedding"])))
    assert replay_error < 1e-4, replay_error
    trained, logits = encode(spec, params, data, [d["embedding"] for d in data], classes, targets)
    matched, init_logits = encode(spec, init, data, [np.asarray(replay(spec, init, d["obs"])) for d in data],
                                  classes, targets)
    episodes = np.arange(COUNTS[1]) // 960
    differences = {}
    for feature in FEATURES:
        differences[feature] = {}
        for name, sl in TARGETS.items():
            p, m = trained[feature][:, sl], matched[feature][:, sl]
            valid = np.isfinite(p).all(1) & np.isfinite(m).all(1)
            interval = bootstrap(y[valid][:, sl], p[valid], test_class[valid], episodes[valid],
                                 n=bootstrap_draws, comparison=m[valid])
            differences[feature][name] = {
                "difference": score(y[valid][:, sl], p[valid], test_class[valid])["r2"]
                - score(y[valid][:, sl], m[valid], test_class[valid])["r2"],
                "ci95": interval["paired_difference_ci95"]}
    q_margin = margin(values[1])
    h_fit, h_test = (rows({"h": d["embedding"]}, "h", n) for d, n in zip(data, COUNTS))
    zeros = [np.zeros(n, dtype=int) for n in COUNTS]
    rho = y[:, 4]
    per_episode = np.array([rho[episodes == e].mean() for e in np.unique(episodes)])
    result = {
        "run_id": run.name, "arm": run.parent.parent.name, "study": recipe["study"], "seed": recipe["seed"],
        "lam": lam, "kappa": kappa, "final_step": step, "self_replay_max_abs": replay_error,
        "provenance": {"init": init_provenance, "final": provenance},
        "test_class_counts": np.bincount(test_class, minlength=3).tolist(),
        "reward": {"expected_per_step": float(rho.mean()),
                   "se": float(per_episode.std(ddof=1) / np.sqrt(len(per_episode))),
                   "archived_gpu_per_step": points[-1]["policy"]["reward_per_step"],
                   "definition": "mean filtered reward-state probability at decision time on the 80,000 greedy test rows"},
        "greedy_agrees_with_belief_mdp": float((values[1].argmax(1) == test_class).mean()),
        "trained": {name: summarize(y, trained[name], test_class) for name in trained},
        "matched_init": {name: summarize(y, matched[name], test_class) for name in matched},
        "paired_differences": differences,
        "global_trained": summarize(y, refit(h_fit, targets[0], h_test, *zeros), zeros[1]),
        "logit_margin_vs_q_margin_spearman": {
            "trained": rank_correlation(margin(logits), q_margin, test_class),
            "matched_init": rank_correlation(margin(init_logits), q_margin, test_class)},
        "by_q_margin_tercile": {name: boundary_bins(y[:, sl], trained["full_h"][:, sl], matched["full_h"][:, sl],
                                                    q_margin, test_class)
                                for name, sl in TARGETS.items() if name in ("delta", "rho")},
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    jax.clear_caches()
    return result


def trajectory(run, lam, kappa):
    """Within-final-class resolution of saved checkpoints replayed on the final policy's histories."""
    raw = RAW / run.name
    output = raw / "trajectory.json"
    if output.exists():
        return json.loads(output.read_text())
    points = json.loads((run / "condition_summary.json").read_text())["checkpoint_reports"]
    steps = np.array([p["checkpoint_step"] for p in points])
    final = int(steps[-1])
    spec, process, params, _ = restore(run, final, raw_root=raw)
    data = histories(run, final, spec, process, params, raw_root=raw)
    classes = [rows(d, "action", n) for d, n in zip(data, COUNTS)]
    belief = [rows(d, "belief", n).astype(float) for d, n in zip(data, COUNTS)]
    q = solve(kernels_for(lam), emission_for(kappa))
    targets = [np.concatenate([b, b[:, 0:1] - b[:, 1:2], b[:, 2:3], v[:, 1:] - v[:, :1]], axis=1)
               for b, v in zip(belief, (q(b) for b in belief))]
    out = []
    for index in sorted({int(np.abs(steps - target).argmin()) for target in TRAJECTORY_STEPS}):
        step = int(steps[index])
        _, _, checkpoint, _ = restore(run, step, raw_root=raw)
        h = [rows({"h": np.asarray(replay(spec, checkpoint, d["obs"]))}, "h", n) for d, n in zip(data, COUNTS)]
        prediction = refit(h[0], targets[0], h[1], *classes)
        out.append({"step": step, "archived_policy": points[index]["policy"],
                    "within_final_class_r2": summarize(targets[1], prediction, classes[1])})
    result = {"run_id": run.name, "arm": run.parent.parent.name, "lam": lam, "kappa": kappa,
              "history_policy_step": final, "points": out}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    jax.clear_caches()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--trajectory", action="store_true")
    args = parser.parse_args()
    for index, (run, lam, kappa) in enumerate(completed_runs()):
        if index % args.shards == args.shard:
            result = analyze_run(run, lam, kappa)
            if args.trajectory:
                trajectory(run, lam, kappa)
            print("DONE", result["arm"], result["seed"], result["reward"]["expected_per_step"],
                  result["paired_differences"]["full_h"]["delta"]["difference"], flush=True)


if __name__ == "__main__":
    main()
