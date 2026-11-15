"""NumPy-only affine accessibility metrics with class-centred denominators."""

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


def scalar_targets(b):
    b = np.asarray(b)
    if b.ndim != 2 or b.shape[1] != 3:
        raise ValueError("beliefs must have three coordinates")
    if len(b) == 0:
        return np.empty((0, 1)), np.empty((0, 1))
    winners = b.argmax(1)
    sorted_b = np.sort(b, axis=1)
    others = np.stack([np.flatnonzero(np.arange(3) != k) for k in winners])
    runner = b[np.arange(len(b))[:, None], others]
    return (sorted_b[:, -1]-sorted_b[:, -2])[:, None], (runner[:, 0]/runner.sum(1))[:, None]


def bootstrap(y, pred, classes, episodes, n=1000, seed=20261106, comparison=None):
    """Resample whole episodes, recomputing class means with probes held fixed."""
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    classes, episodes = np.asarray(classes), np.asarray(episodes)
    if y.ndim != 2 or y.shape != pred.shape or classes.shape != (len(y),) or episodes.shape != (len(y),):
        raise ValueError("aligned target, prediction, class and episode rows required")
    if len(y) == 0 or n < 1:
        raise ValueError("bootstrap requires rows and positive resample count")
    if comparison is not None:
        comparison = np.asarray(comparison, dtype=float)
        if comparison.shape != y.shape:
            raise ValueError("comparison predictions must align with targets")
    _, group = np.unique(episodes, return_inverse=True)
    groups = int(group.max()+1)
    labels = np.unique(classes)
    counts = np.zeros((groups, len(labels)))
    sums = np.zeros((groups, len(labels), y.shape[1]))
    squares = np.zeros((groups, len(labels)))
    errors = np.zeros((groups, len(labels)))
    other_errors = np.zeros_like(errors)
    for j, k in enumerate(labels):
        mask = classes == k
        np.add.at(counts[:, j], group[mask], 1)
        np.add.at(sums[:, j], group[mask], y[mask])
        np.add.at(squares[:, j], group[mask], (y[mask]**2).sum(1))
        np.add.at(errors[:, j], group[mask], ((y[mask]-pred[mask])**2).sum(1))
        if comparison is not None:
            np.add.at(other_errors[:, j], group[mask], ((y[mask]-comparison[mask])**2).sum(1))
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(groups, np.full(groups, 1/groups), size=n)
    count, total = weights @ counts, np.einsum("rg,gkd->rkd", weights, sums)
    sst = (weights @ squares - (total**2).sum(2)/np.maximum(count, 1)).sum(1)
    sse = (weights @ errors).sum(1)
    valid = sst > 1e-20
    r2 = 1-sse[valid]/sst[valid]
    rmse = np.sqrt(sse/(count.sum(1)*y.shape[1]))
    result = {"n_resamples": n, "n_episodes": groups, "valid_r2_resamples": int(valid.sum()),
            "r2_ci95": np.quantile(r2, [.025, .975]).tolist() if len(r2) else None,
            "r2_se": float(r2.std(ddof=1)) if len(r2) > 1 else None,
            "rmse_ci95": np.quantile(rmse, [.025, .975]).tolist(), "seed": seed}
    if comparison is not None:
        differences = ((weights @ other_errors).sum(1)[valid]-sse[valid])/sst[valid]
        result["paired_difference_ci95"] = np.quantile(differences, [.025, .975]).tolist() if len(differences) else None
        result["paired_difference_se"] = float(differences.std(ddof=1)) if len(differences) > 1 else None
    return result


def seed_mean_intervals(values, n=10000, seed=20261106):
    """Columnwise mean intervals, resampling complete training-seed rows."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all() or n < 1:
        raise ValueError("finite metrics for at least two seeds and positive resamples required")
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(values), size=(n, len(values)))
    means = values[indices].mean(axis=1)
    return {"n_seeds": len(values), "n_resamples": n, "seed": seed,
            "mean": values.mean(axis=0).tolist(),
            "ci95": np.quantile(means, [.025, .975], axis=0).T.tolist()}
