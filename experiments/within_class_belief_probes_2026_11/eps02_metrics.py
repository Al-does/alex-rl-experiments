"""Variance decomposition and equal-training-seed summaries for eps02."""

import numpy as np

from .metrics import score, seed_mean_intervals


def class_variance(belief, classes, fit_belief, fit_classes):
    belief, fit_belief = np.asarray(belief, float), np.asarray(fit_belief, float)
    classes, fit_classes = np.asarray(classes), np.asarray(fit_classes)
    if (belief.ndim != 2 or belief.shape[1] != 3 or fit_belief.ndim != 2
            or fit_belief.shape[1] != 3 or classes.shape != (len(belief),)
            or fit_classes.shape != (len(fit_belief),)
            or not np.isfinite(belief).all() or not np.isfinite(fit_belief).all()):
        raise ValueError("aligned finite three-state beliefs and class vectors required")
    total = score(belief, belief)["sst"]
    within = score(belief, belief, classes)["sst"]
    predicted = np.full_like(belief, np.nan)
    for k in np.unique(classes):
        if np.any(fit_classes == k):
            predicted[classes == k] = fit_belief[fit_classes == k].mean(0)
    valid = np.isfinite(predicted).all(1)
    return {"between_class_fraction": 1-within/total if total > 1e-20 else None,
            "global_sst": total, "within_class_sst": within,
            "class_only_held_out": score(belief[valid], predicted[valid]),
            "class_only_excluded_rows": int((~valid).sum())}


def values(record):
    final, own, matched = record["final"], record["init"], record["matched_init"]
    result = {"global_trained": final["global"]["r2"],
              "global_init": own["global"]["r2"],
              "global_matched_init": matched["global"]["r2"],
              "between_class_fraction": final["variance"]["between_class_fraction"],
              "between_class_fraction_init": own["variance"]["between_class_fraction"],
              "class_only_held_out": final["variance"]["class_only_held_out"]["r2"],
              "within_trained": final["metrics"]["full_h"]["belief"]["r2"],
              "within_matched_init": matched["metrics"]["full_h"]["belief"]["r2"],
              "within_own_init": own["metrics"]["full_h"]["belief"]["r2"]}
    a, b = result["within_trained"], result["within_matched_init"]
    result["final_minus_matched_init"] = a-b if a is not None and b is not None else None
    for name in ("head_blind", "logits"):
        result[name] = final["metrics"][name]["belief"]["r2"]
        result[name+"_matched_init"] = matched["metrics"][name]["belief"]["r2"]
    for target in ("delta", "rho"):
        result[target] = final["metrics"]["full_h"][target]["r2"]
        result[target+"_matched_init"] = matched["metrics"]["full_h"][target]["r2"]
    for name in ("history_shuffle", "random_input", "label_permutation"):
        for stage, data in (("", final), ("_matched_init", matched), ("_own_init", own)):
            result[name+stage] = data["controls"][name]["belief"]["r2"]
    return result


def summarize_seeds(records, *, eligible_only, n=10000, seed=20261118):
    if n < 1:
        raise ValueError("positive bootstrap resample count required")
    seeds = [r["seed"] for r in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError("one record per training seed required")
    selected = [r for r in records if not eligible_only or r["within_class_eligible"]]
    rows = [values(r) for r in selected]
    keys = list(values(records[0])) if records else []
    metrics = {}
    for key in keys:
        finite = [(r["seed"], row[key]) for r, row in zip(selected, rows)
                  if row[key] is not None and np.isfinite(row[key])]
        array = np.asarray([v for _, v in finite], float)
        if len(array) > 1:
            summary = seed_mean_intervals(array[:, None], n=n, seed=seed)
            mean, ci = summary["mean"][0], summary["ci95"][0]
        else:
            mean = float(array[0]) if len(array) else None
            ci = None
        metrics[key] = {"mean": mean, "ci95": ci, "n_seeds": len(array),
                        "seeds": [s for s, _ in finite],
                        "excluded_seeds": [r["seed"] for r, row in zip(selected, rows)
                                           if row[key] is None or not np.isfinite(row[key])]}
    differences = [row["final_minus_matched_init"] for row in rows
                   if row["final_minus_matched_init"] is not None]
    return {"n_runs": len(records), "n_seeds": len(selected),
            "seeds": [r["seed"] for r in selected],
            "one_class_seeds": [r["seed"] for r in records if r["final"]["one_class"]],
            "ineligible_seeds": [r["seed"] for r in records if not r["within_class_eligible"]],
            "positive_final_minus_matched_init": int(np.sum(np.asarray(differences) > 0)),
            "n_defined_differences": len(differences),
            "n_resamples": n, "bootstrap_seed": seed, "metrics": metrics}
