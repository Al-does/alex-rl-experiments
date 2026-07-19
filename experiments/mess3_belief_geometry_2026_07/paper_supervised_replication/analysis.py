"""Layer-wise affine probes and paper-style simplex visualizations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .mess3 import bayesian_beliefs
from .model import PaperTransformer


TRIANGLE = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]],
    dtype=np.float64,
)


def grouped_probe_split(
    sequence_count: int,
    *,
    seed: int,
    fit_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Split whole sequences so positions from one context never leak."""
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be between zero and one")
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(sequence_count)
    fit_count = max(1, min(sequence_count - 1, round(
        fit_fraction * sequence_count
    )))
    return permutation[:fit_count], permutation[fit_count:]


@torch.no_grad()
def extract_activations(
    model: PaperTransformer,
    contexts: torch.Tensor,
    *,
    activation_name: str,
    batch_size: int,
) -> np.ndarray:
    """Extract one residual location in bounded inference batches."""
    device = next(model.parameters()).device
    chunks = []
    model.eval()
    for start in range(0, len(contexts), batch_size):
        batch = contexts[start : start + batch_size].to(device)
        _, activations = model(batch, return_activations=True)
        chunks.append(
            activations[activation_name]
            .reshape(-1, model.config.d_model)
            .float()
            .cpu()
            .numpy()
        )
    return np.concatenate(chunks, axis=0)


def fit_affine_ols(
    activations: np.ndarray,
    beliefs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``belief = activation @ weight + bias`` by plain OLS."""
    design = np.concatenate(
        [
            np.asarray(activations, dtype=np.float64),
            np.ones((len(activations), 1), dtype=np.float64),
        ],
        axis=1,
    )
    solution, _, _, _ = np.linalg.lstsq(
        design,
        np.asarray(beliefs, dtype=np.float64),
        rcond=None,
    )
    return solution[:-1], solution[-1]


def probe_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    residual = np.asarray(target) - np.asarray(prediction)
    squared_error = np.square(residual)
    mse_per_coordinate = squared_error.mean(axis=0)
    centered = target - target.mean(axis=0, keepdims=True)
    denominator = np.square(centered).sum(axis=0)
    numerator = squared_error.sum(axis=0)
    r2_per_coordinate = 1.0 - numerator / denominator
    return {
        "mse": float(squared_error.mean()),
        "mse_per_coordinate": mse_per_coordinate.tolist(),
        "r2": float(r2_per_coordinate.mean()),
        "r2_per_coordinate": r2_per_coordinate.tolist(),
    }


def run_layer_probes(
    model: PaperTransformer,
    contexts: torch.Tensor,
    *,
    seed: int,
    batch_size: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Fit on 20% of contexts and evaluate every layer on held-out 80%."""
    beliefs = bayesian_beliefs(contexts).cpu().numpy()
    fit_indices, test_indices = grouped_probe_split(
        len(contexts),
        seed=seed,
    )
    fit_contexts = contexts[fit_indices]
    test_contexts = contexts[test_indices]
    fit_beliefs = beliefs[fit_indices].reshape(-1, beliefs.shape[-1])
    test_beliefs = beliefs[test_indices].reshape(-1, beliefs.shape[-1])

    result: dict[str, Any] = {
        "fit_fraction": 0.2,
        "n_fit_sequences": len(fit_indices),
        "n_test_sequences": len(test_indices),
        "n_fit_positions": len(fit_beliefs),
        "n_test_positions": len(test_beliefs),
        "split_unit": "whole_context",
        "layers": {},
    }
    headline_prediction = None
    for activation_name in model.activation_names:
        fit_activations = extract_activations(
            model,
            fit_contexts,
            activation_name=activation_name,
            batch_size=batch_size,
        )
        weight, bias = fit_affine_ols(fit_activations, fit_beliefs)
        del fit_activations
        test_activations = extract_activations(
            model,
            test_contexts,
            activation_name=activation_name,
            batch_size=batch_size,
        )
        prediction = test_activations @ weight + bias
        del test_activations
        metrics = probe_metrics(prediction, test_beliefs)
        result["layers"][activation_name] = metrics
        if activation_name == f"block_{model.config.n_layers - 1}":
            headline_prediction = prediction

    if headline_prediction is None:
        raise RuntimeError("final pre-LN residual was not probed")
    result["headline_layer"] = f"block_{model.config.n_layers - 1}"
    return result, test_beliefs, headline_prediction


def _draw_triangle(axis) -> None:
    outline = np.vstack([TRIANGLE, TRIANGLE[0]])
    axis.plot(outline[:, 0], outline[:, 1], color="black", linewidth=0.6)
    axis.set_aspect("equal")
    axis.axis("off")


def plot_belief_comparison(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    path: Path,
    mse: float,
    r2: float,
    seed: int,
    max_points: int = 150_000,
) -> None:
    """Create the true-vs-decoded simplex figure colored by true belief."""
    if len(target) > max_points:
        indices = np.random.default_rng(seed).choice(
            len(target),
            size=max_points,
            replace=False,
        )
        target = target[indices]
        prediction = prediction[indices]
    true_xy = target @ TRIANGLE
    decoded_xy = prediction @ TRIANGLE
    colors = np.clip(target, 0.0, 1.0)
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    axes[0].scatter(
        true_xy[:, 0],
        true_xy[:, 1],
        c=colors,
        s=0.35,
        alpha=0.35,
        linewidths=0,
        rasterized=True,
    )
    axes[0].set_title("Exact Bayesian beliefs")
    axes[1].scatter(
        decoded_xy[:, 0],
        decoded_xy[:, 1],
        c=colors,
        s=0.35,
        alpha=0.35,
        linewidths=0,
        rasterized=True,
    )
    axes[1].set_title(f"Decoded final residual\nMSE={mse:.5g}, R²={r2:.4f}")
    for axis in axes:
        _draw_triangle(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_training_curve(
    history: list[dict[str, Any]],
    *,
    floor_nats: float,
    path: Path,
) -> None:
    training = [
        record for record in history if record["kind"] == "training"
    ]
    validation = [
        record for record in history if record["kind"] == "validation"
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    if training:
        axis.plot(
            [max(1, record["step"]) for record in training],
            [record["training_loss_nats"] for record in training],
            label="sampled train CE",
            linewidth=1.0,
        )
    axis.plot(
        [max(1, record["step"]) for record in validation],
        [record["validation_loss_nats"] for record in validation],
        "o-",
        label="exact validation CE",
        markersize=3,
    )
    axis.axhline(
        floor_nats,
        color="black",
        linestyle="--",
        linewidth=0.9,
        label="exact Bayesian floor",
    )
    axis.set_xscale("log")
    axis.set_xlabel("optimizer update")
    axis.set_ylabel("cross-entropy (nats)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_probe_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2) + "\n")
