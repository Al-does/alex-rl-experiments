"""Constrain only error>1 five-site predictions; preserve original reports."""
from __future__ import annotations

from copy import deepcopy
import csv
import io
import json
from pathlib import Path

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import five_site_probes as probe
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import five_site_probes_stage2 as stage2
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import five_site_hashes
from experiments.algorithm_generality_probes_eps02_2026_11.paper_constraints import constrain_predictions
from analysis.probes.linear import fit_affine_probe, global_mse_metrics, r2_score

import numpy as np

OUTPUT = stage2.OUTPUT
REMEDY = probe.ROOT.parent / "deep-beliefs/experiments/persistent_nuisance_v2/constrain_degenerate.py"
REMEDY_SOURCE_SHA256 = "809466500dda8c42f5f6155c42cb8025ab483bef8e182f6452a2fda8ef39d55f"
TARGETS = ("b", "c", "rho", "delta")
PAIRS = ((2, "b"), (2, "c"), (2, "delta"), (3, "b"), (3, "delta"))


def projector():
    """Return the in-repo projection and its historical upstream-source digest.

    REMEDY identifies the original source in archived reports; it is not read.
    The function extract in paper_constraints is unchanged from that source.
    """
    return constrain_predictions, REMEDY_SOURCE_SHA256


def metrics(prediction, target):
    result = global_mse_metrics(prediction, target)
    result["r_squared"] = r2_score(prediction, target)
    result["error"] = 1 - result["r_squared"]
    return result


def selected(reports):
    return {(d["variant"], d["seed"], d["checkpoint_step"], site, target)
            for d in reports for site, scores in d["scores"].items()
            for target, score in scores.items() if score["error"] > 1}


def apply_replacements(reports, replacements):
    corrected = deepcopy(reports)
    lookup = {(d["variant"], d["seed"], d["checkpoint_step"]): d for d in corrected}
    keys = [(r["variant"], r["seed"], r["checkpoint_step"], r["site"], r["target"])
            for r in replacements]
    if len(keys) != len(set(keys)) or set(keys) != selected(reports):
        raise ValueError("replacements must cover exactly the original error>1 scores")
    for key, row in zip(keys, replacements):
        variant, seed, step, site, target = key
        original = lookup[variant, seed, step]["scores"][site][target]
        if original != row["old"]:
            raise ValueError("replacement original differs")
        lookup[variant, seed, step]["scores"][site][target] = row["new"]
    return corrected


def collect(base, core, report):
    directory = probe.ROOT / report["results_path"]
    recipe = base.read(directory / "resolved_recipe.json")
    path, provenance = base.verified_checkpoint(directory, report["checkpoint_step"])
    assert provenance["checkpoint_sha256"] == report["checkpoint_sha256"]
    spec = probe.models.ModelSpec(**recipe["model"])
    params = probe.load_params(path, spec)
    keys = probe.jax.random.split(probe.jax.random.fold_in(probe.jax.random.PRNGKey(42), 2))
    raw = [probe._probe_rollout(spec, probe.make_process(**recipe["process"]), params, key,
                              -(-n // 960), 1024) for key, n in zip(keys, (60000, 80000))]
    features = []
    for rows, n in zip(raw, (60000, 80000)):
        sites = np.asarray(probe.replay_sites(spec, params, rows["obs"],
                                             core.instrumented_step, core.empty_maps))
        np.testing.assert_array_equal(sites[:, :, -1], rows["embedding"])
        features.append(base.flatten(sites, n))
    beliefs = [base.flatten(r["belief"], n) for r, n in zip(raw, (60000, 80000))]
    coarse = [base.flatten(base.merged_filter(r["obs"], recipe["process"]), n)
              for r, n in zip(raw, (60000, 80000))]
    return features, {"b": beliefs, "c": coarse, "rho": [b[:, 2:3] for b in beliefs],
                      "delta": [b[:, :1] - b[:, 1:2] for b in beliefs]}


def fit(features, targets):
    weight, bias = fit_affine_probe(features[0], targets[0], ridge=1e-6)
    return np.asarray(features[1], dtype=np.float64) @ weight + bias, weight, bias


def diagnose(features, targets, prediction, weight):
    train, test = [np.asarray(x, dtype=np.float64) for x in features]
    center = train.mean(axis=0)
    _, singular, vt = np.linalg.svd(train - center, full_matrices=False)
    train_directions = (train - center) @ vt.T
    test_directions = (test - center) @ vt.T
    train_max = np.max(np.abs(train_directions), axis=0)
    test_max = np.max(np.abs(test_directions), axis=0)
    gains = np.linalg.norm((vt @ weight), axis=1)
    bound = test_max * gains
    direction = int(np.argmax(bound))
    squared = np.square(prediction - targets[1]).sum(axis=1)
    return {"singular_values": singular.tolist(), "condition_number": float(singular[0] / singular[-1]),
            "ridge": 1e-6, "weight_norm": float(np.linalg.norm(weight)),
            "max_abs_prediction": float(np.abs(prediction).max()),
            "fraction_rows_outside_simplex": float(np.mean(np.any((prediction < 0) | (prediction > 1), axis=1))),
            "worst_10_rows_fraction_squared_error": float(np.sort(squared)[-10:].sum() / squared.sum()),
            "largest_direction_test_contribution": {"direction_index": direction,
                "singular_value": float(singular[direction]), "coefficient_norm": float(gains[direction]),
                "train_max_abs_coordinate": float(train_max[direction]),
                "test_max_abs_coordinate": float(test_max[direction]),
                "test_to_train_max_ratio": float(test_max[direction] / train_max[direction])},
            "train_target_variance": float(np.var(targets[0], axis=0).mean()),
            "test_target_variance": float(np.var(targets[1], axis=0).mean())}


def replace(reports):
    projection, remedy_hash = projector()
    base, _ = probe.frozen_module(probe.ARCH_REV, probe.ARCH_PATH)
    core, _ = probe.frozen_module(probe.STEER_REV, probe.CORE_PATH)
    candidates = selected(reports)
    replacements, diagnostics = [], []
    groups = sorted({k[:3] for k in candidates})
    lookup = {(d["variant"], d["seed"], d["checkpoint_step"]): d for d in reports}
    with probe.jax.default_matmul_precision("highest"):
        for group in groups + [(2, 50, 0)]:
            report = lookup[group]
            features, targets = collect(base, core, report)
            sites = sorted({k[3] for k in candidates if k[:3] == group}, key=probe.SITES.index)
            if group not in groups:
                sites = ["block_1"]
            for site in sites:
                x = [f[:, probe.SITES.index(site)] for f in features]
                belief_prediction, weight, _ = fit(x, targets["b"])
                np.testing.assert_allclose(metrics(belief_prediction, targets["b"][1])["error"],
                                           report["scores"][site]["b"]["error"], rtol=1e-10, atol=1e-12)
                if site == "block_1":
                    diagnostics.append({"variant": group[0], "seed": group[1], "checkpoint_step": group[2],
                                        "site": site, **diagnose(x, targets["b"], belief_prediction, weight)})
                projected_belief = projection(belief_prediction, "belief")
                for target in TARGETS:
                    if (*group, site, target) not in candidates:
                        continue
                    prediction = belief_prediction if target == "b" else fit(x, targets[target])[0]
                    old = report["scores"][site][target]
                    reproduced = metrics(prediction, targets[target][1])
                    np.testing.assert_allclose(reproduced["error"], old["error"], rtol=1e-10, atol=1e-12)
                    if target == "b":
                        bounded, method = projected_belief, "Euclidean projection of original b predictions onto 3-state simplex"
                    elif target == "c":
                        bounded = projection(np.concatenate([1 - prediction, prediction], axis=1), "belief")[:, 1:2]
                        method = "Euclidean projection of (1-c,c) onto its own 2-state simplex; score reward coordinate"
                    elif target == "rho":
                        bounded, method = projected_belief[:, 2:3], "derive rho from projected b (not independent scalar clipping)"
                    else:
                        bounded = projected_belief[:, :1] - projected_belief[:, 1:2]
                        method = "derive delta from projected b (not independent scalar clipping)"
                    new = metrics(bounded, targets[target][1])
                    row = dict(variant=group[0], seed=group[1], checkpoint_step=group[2], site=site,
                               target=target, checkpoint_sha256=report["checkpoint_sha256"], old=old,
                               ridge_reproduced=reproduced, new=new, method=method)
                    replacements.append(row)
                    print(group, site, target, old["error"], "->", new["error"], flush=True)
    return {"selection_rule": "all original error > 1, across all 90 reports and all sites/targets",
            "selection_warning": "retrospective test-error screening; selected scores are constrained but others remain raw",
            "protocol": reports[0]["protocol"], "projector_source": str(REMEDY),
            "projector_source_sha256": remedy_hash, "replacements": replacements,
            "diagnostics": diagnostics, "diagnostic_control": "Reward-State initialization seed 50, block 1; not replaced"}


def summarize(reports, replacement_data):
    corrected = apply_replacements(reports, replacement_data["replacements"])
    lookup = {(d["variant"], d["seed"], d["checkpoint_step"]): d for d in corrected}
    indices = np.random.default_rng(42).integers(0, 15, (100000, 15))
    aggregates, rows, pairs = [], [], []
    text = ["# Corrected five-site probes", "", "Cells: mean 1−R² [95% seed-bootstrap percentile CI].",
            "100,000 seed resamples; RNG seed 42; original error>1 scores only constrained.",
            "Full-State c is the merged-token control, not an exact quotient.", ""]
    for step in stage2.STEPS:
        text += [f"## {'Initialization' if step == 0 else 'Final'}", "",
                 "| Condition | Site | b | c | ρ | δ |", "|---|---|---:|---:|---:|---:|"]
        for variant in (1, 2, 3):
            for site in probe.SITES:
                cells = []
                for target in TARGETS:
                    values = []
                    for seed in stage2.SEEDS:
                        score = lookup[variant, seed, step]["scores"][site][target]
                        values.append(score["error"])
                        rows.append(dict(condition=stage2.NAMES[variant], seed=seed, step=step,
                                         site=site, target=target, **score))
                    cell = stage2.bootstrap_mean_ci(values, indices)
                    aggregates.append(dict(condition=stage2.NAMES[variant], site=site, step=step, target=target, **cell))
                    cells.append(f"{cell['mean']:.4f} [{cell['ci95_low']:.4f}, {cell['ci95_high']:.4f}]")
                text.append("| " + " | ".join([stage2.NAMES[variant], site, *cells]) + " |")
        text.append("")
    text += ["## Paired counts: final error below initialization error", "",
             "| Condition | Target | Block 1 | Block 2 | Block 3 | Block 4 | Final norm |",
             "|---|---|---:|---:|---:|---:|---:|"]
    counts = []
    for variant, target in PAIRS:
        cells = []
        for site in probe.SITES:
            subset = []
            for seed in stage2.SEEDS:
                init, final = [lookup[variant, seed, step]["scores"][site][target]["error"] for step in stage2.STEPS]
                row = dict(condition=stage2.NAMES[variant], target=target, site=site, seed=seed,
                           init_error=init, final_error=final, final_minus_init=final-init, final_below_init=final<init)
                pairs.append(row)
                subset.append(row)
            count = sum(r["final_below_init"] for r in subset)
            counts.append(dict(condition=stage2.NAMES[variant], target=target, site=site, count=count, n=15))
            cells.append(f"{count}/15")
        text.append("| " + " | ".join([stage2.NAMES[variant], target, *cells]) + " |")
    result = {"aggregates": aggregates, "paired_counts": counts, "pairs": pairs,
              "bootstrap": {"replicates": 100000, "rng_seed": 42, "method": "percentile",
                            "quantile_method": "linear", "shared_seed_indices": True, "seeds": stage2.SEEDS},
              "replacement_count": len(replacement_data["replacements"])}
    (OUTPUT / "corrected_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUTPUT / "corrected_tables.md").write_text("\n".join(text) + "\n")
    for name, data in [("corrected_per_seed.csv", rows), ("corrected_paired.csv", pairs)]:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
        (OUTPUT / name).write_text(buffer.getvalue())
    print("\n".join(text))


def main():
    names = ["degenerate_fit_replacements.json", "corrected_summary.json", "corrected_tables.md",
             "corrected_per_seed.csv", "corrected_paired.csv", "corrected_hashes.json"]
    for name in names:
        if (OUTPUT / name).exists():
            raise FileExistsError(OUTPUT / name)
    paths = sorted(OUTPUT.glob("variant_*.json"))
    reports = [json.loads(p.read_text()) for p in paths]
    assert len(reports) == 90
    assert {(d["variant"],d["seed"],d["checkpoint_step"]) for d in reports} == {
        (v,s,t) for v in (1,2,3) for s in stage2.SEEDS for t in stage2.STEPS}
    raw_hashes = {p.name: stage2.digest(p) for p in OUTPUT.iterdir() if p.is_file()}
    replacements = replace(reports)
    replacements["raw_input_sha256"] = raw_hashes
    (OUTPUT / names[0]).write_text(json.dumps(replacements, indent=2) + "\n")
    summarize(reports, replacements)
    assert all(stage2.digest(OUTPUT / name) == h for name,h in raw_hashes.items())
    sources = paths + [OUTPUT / n for n in names[:-1]] + [Path(__file__)]
    five_site_hashes.write_manifest(OUTPUT / names[-1], sources)
    print(f"Validated {len(replacements['replacements'])} replacements; all original scores reproduced; all raw files unchanged.")


if __name__ == "__main__":
    main()
