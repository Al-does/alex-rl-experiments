"""Unmodified function extracts: deep-beliefs@177ba8d9081933b83a3834b7a1f92ac31bb3f02b experiments/persistent_nuisance_v2/constrain_degenerate.py."""

import numpy as np


def constrain_predictions(predictions, target):
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim != 2 or not len(predictions) or not np.isfinite(predictions).all():
        raise ValueError("expected finite nonempty prediction matrix")
    if target == "belief":
        if predictions.shape[1] < 2:
            raise ValueError("belief requires at least two components")
        ordered = np.sort(predictions, axis=1)[:, ::-1]
        cumulative = ordered.cumsum(axis=1) - 1
        index = np.arange(1, predictions.shape[1] + 1)
        rho = (ordered - cumulative / index > 0).sum(axis=1) - 1
        threshold = cumulative[np.arange(len(predictions)), rho] / (rho + 1)
        return np.maximum(predictions - threshold[:, None], 0)
    if target not in ("rho", "delta", "coarse_b") or predictions.shape[1] != 1:
        raise ValueError("expected a scalar rho, delta or coarse_b target")
    return np.clip(predictions, -1 if target == "delta" else 0, 1)


def constraint_method(target):
    return {"belief": "Euclidean projection onto probability simplex",
            "rho": "clip to [0,1]", "coarse_b": "clip to [0,1]",
            "delta": "clip to [-1,1]"}[target]
