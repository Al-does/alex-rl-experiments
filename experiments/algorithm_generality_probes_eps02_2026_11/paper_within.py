"""Unmodified function extracts: alex-rl-experiments@c351a35032d6530daeaf42264ec32c64707517e9 experiments/within_class_belief_probes_2026_11/metrics.py."""

import numpy as np


def affine(x, y, test, ridge=1e-6):
    x, y, test = [np.asarray(a, dtype=np.float64) for a in (x, y, test)]
    if x.ndim != 2 or y.ndim != 2 or test.ndim != 2 or len(x) != len(y) or x.shape[1] != test.shape[1] or len(x) < 2:
        raise ValueError("aligned nonempty matrices and at least two fit rows required")
    xm, ym = x.mean(0), y.mean(0)
    xc = x-xm
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    design = np.concatenate([xc, np.sqrt(ridge)*np.eye(x.shape[1])])
    response = np.concatenate([y-ym, np.zeros((x.shape[1], y.shape[1]))])
    weight = np.linalg.lstsq(design, response, rcond=None)[0]
    return (test-xm) @ weight + ym


def score(y, pred, classes=None):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    if y.ndim != 2 or y.shape != pred.shape:
        raise ValueError("targets and predictions must be aligned matrices")
    if classes is None:
        classes = np.zeros(len(y), dtype=int)
    classes = np.asarray(classes)
    if classes.shape != (len(y),):
        raise ValueError("classes must align with rows")
    sse = float(np.sum((y-pred)**2))
    sst = sum(float(np.sum((y[classes == k]-y[classes == k].mean(0))**2)) for k in np.unique(classes))
    return {"n": len(y), "r2": 1-sse/sst if sst > 1e-20 else None,
            "rmse": float(np.sqrt(sse/y.size)) if y.size else None,
            "sse": sse, "sst": sst}


def refit(x, y, test, fit_class, test_class, *, permute_seed=None):
    result = np.full((len(test), y.shape[1]), np.nan)
    rng = np.random.default_rng(permute_seed)
    for k in np.unique(test_class):
        fit = fit_class == k
        held = test_class == k
        if fit.sum() < 2:
            continue
        targets = y[fit]
        if permute_seed is not None:
            targets = rng.permutation(targets)
        result[held] = affine(x[fit], targets, test[held])
    return result


def null_basis(weight):
    weight = np.asarray(weight, dtype=float)
    if weight.ndim != 2 or weight.shape[0] != 3 or weight.shape[1] < 3:
        raise ValueError("expected a three-action affine head with at least three features")
    contrasts = weight[:2]-weight[2]
    _, singular, right = np.linalg.svd(contrasts, full_matrices=True)
    rank = int(np.sum(singular > singular[0]*1e-12)) if singular[0] else 0
    if rank != 2:
        raise ValueError("action-difference head does not have rank two")
    return right[rank:].T
