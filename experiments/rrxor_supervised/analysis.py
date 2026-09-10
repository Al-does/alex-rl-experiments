"""Paper-style layer probes with held-out belief-geometry controls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.belief_geometry import (
    ProbeBatteryResult,
    evaluate_belief_geometry,
    prediction_null_basis,
)

from .model import RRXORPaperTransformer
from .process import bayesian_beliefs, labeled_operators, next_token_probabilities


def grouped_context_split(
    sequence_count: int,
    *,
    seed: int,
    fit_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Split complete contexts so positions from one sequence never leak."""

    if sequence_count < 2:
        raise ValueError("at least two contexts are required")
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be between zero and one")
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(sequence_count)
    fit_count = max(
        1,
        min(sequence_count - 1, round(fit_fraction * sequence_count)),
    )
    return permutation[:fit_count], permutation[fit_count:]


@torch.no_grad()
def extract_representations(
    model: RRXORPaperTransformer,
    contexts: torch.Tensor,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Extract each residual block, their concatenation, and final LayerNorm."""

    device = next(model.parameters()).device
    chunks: dict[str, list[np.ndarray]] = {
        **{
            f"layer_{index + 1}": []
            for index in range(model.config.n_layers)
        },
        "concatenated_layers": [],
        "post_final_layer_norm": [],
    }
    model.eval()
    for start in range(0, len(contexts), batch_size):
        batch = contexts[start : start + batch_size].to(device)
        _, activations = model(batch, return_activations=True)
        residuals = [
            activations[f"block_{index}"]
            for index in range(model.config.n_layers)
        ]
        for index, residual in enumerate(residuals):
            chunks[f"layer_{index + 1}"].append(
                residual.reshape(-1, model.config.d_model)
                .double()
                .cpu()
                .numpy()
            )
        chunks["concatenated_layers"].append(
            torch.cat(residuals, dim=-1)
            .reshape(-1, model.config.n_layers * model.config.d_model)
            .double()
            .cpu()
            .numpy()
        )
        chunks["post_final_layer_norm"].append(
            activations["final_ln"]
            .reshape(-1, model.config.d_model)
            .double()
            .cpu()
            .numpy()
        )
    return {
        name: np.concatenate(values, axis=0)
        for name, values in chunks.items()
    }


def suffix_features(contexts: torch.Tensor, width: int = 3) -> np.ndarray:
    """Encode the latest tokens with a distinct left-padding category."""

    if width <= 0:
        raise ValueError("suffix width must be positive")
    values = contexts.detach().cpu().numpy()
    rows = []
    padding_token = 2
    for sequence in values:
        for position in range(sequence.shape[0]):
            suffix = sequence[max(0, position - width + 1) : position + 1]
            padded = np.full(width, padding_token, dtype=np.int64)
            padded[-len(suffix) :] = suffix
            rows.append(padded)
    keys = np.asarray(rows, dtype=np.int64)
    return np.eye(3, dtype=np.float64)[keys].reshape(len(keys), -1)


def suffix_keys(contexts: torch.Tensor, width: int = 3) -> np.ndarray:
    """Return categorical suffix rows for nuisance-matched feature controls."""

    encoded = suffix_features(contexts, width=width)
    return encoded.reshape(len(encoded), width, 3).argmax(axis=-1)


def _prediction_map() -> np.ndarray:
    operators = labeled_operators().cpu().numpy()
    return operators.sum(axis=-1).T


def _features_for_indices(
    features: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    context_length: int,
) -> dict[str, np.ndarray]:
    shaped = {
        name: values.reshape(-1, context_length, values.shape[-1])
        for name, values in features.items()
    }
    return {
        name: values[indices].reshape(-1, values.shape[-1])
        for name, values in shaped.items()
    }


def _rows_for_indices(
    values: np.ndarray,
    indices: np.ndarray,
    *,
    context_length: int,
) -> np.ndarray:
    shaped = values.reshape(-1, context_length, values.shape[-1])
    return shaped[indices].reshape(-1, values.shape[-1])


def _distance_r2(predictor: np.ndarray, target: np.ndarray) -> float | None:
    predictor = np.asarray(predictor, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    design = np.column_stack([predictor, np.ones(len(predictor))])
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    prediction = design @ coefficients
    denominator = np.square(target - target.mean()).sum()
    if denominator <= 0.0:
        return None
    return float(1.0 - np.square(target - prediction).sum() / denominator)


def pairwise_geometry_metrics(
    beliefs: np.ndarray,
    decoded_beliefs: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compare decoded distances with beliefs and next-token predictions."""

    rounded = np.round(np.asarray(beliefs, dtype=np.float64), decimals=12)
    unique_beliefs, inverse = np.unique(rounded, axis=0, return_inverse=True)
    centers = np.zeros_like(unique_beliefs)
    counts = np.zeros(len(unique_beliefs), dtype=np.int64)
    np.add.at(centers, inverse, decoded_beliefs)
    np.add.at(counts, inverse, 1)
    centers = centers / counts[:, None]
    next_tokens = unique_beliefs @ _prediction_map()
    first, second = np.triu_indices(len(unique_beliefs), k=1)
    belief_distances = np.linalg.norm(
        unique_beliefs[first] - unique_beliefs[second],
        axis=1,
    )
    decoded_distances = np.linalg.norm(
        centers[first] - centers[second],
        axis=1,
    )
    next_token_distances = np.linalg.norm(
        next_tokens[first] - next_tokens[second],
        axis=1,
    )
    metrics = {
        "n_belief_states": int(len(unique_beliefs)),
        "n_pairs": int(len(first)),
        "belief_distance_to_decoded_distance_r_squared": _distance_r2(
            belief_distances,
            decoded_distances,
        ),
        "next_token_distance_to_decoded_distance_r_squared": _distance_r2(
            next_token_distances,
            decoded_distances,
        ),
        "aggregation": "mean decoded belief per exact belief state",
    }
    arrays = {
        "unique_beliefs": unique_beliefs,
        "decoded_centers": centers,
        "belief_distances": belief_distances,
        "decoded_distances": decoded_distances,
        "next_token_distances": next_token_distances,
    }
    return metrics, arrays


def run_belief_probes(
    model: RRXORPaperTransformer,
    initialization_model: RRXORPaperTransformer,
    contexts: torch.Tensor,
    *,
    seed: int,
    batch_size: int,
    smoke: bool,
) -> tuple[dict[str, Any], ProbeBatteryResult, np.ndarray, dict[str, np.ndarray]]:
    """Fit on 20% of contexts and evaluate on the held-out 80%."""

    context_length = contexts.shape[1]
    beliefs = bayesian_beliefs(contexts).double().cpu().numpy()
    flattened_beliefs = beliefs.reshape(-1, beliefs.shape[-1])
    exact_next_tokens = next_token_probabilities(
        torch.from_numpy(flattened_beliefs),
    ).numpy()
    history = suffix_features(contexts)
    history_keys = suffix_keys(contexts)
    current_tokens = np.eye(2, dtype=np.float64)[
        contexts.detach().cpu().numpy()
    ].reshape(-1, 2)
    trained_features = extract_representations(
        model,
        contexts,
        batch_size=batch_size,
    )
    initialization_features = extract_representations(
        initialization_model,
        contexts,
        batch_size=batch_size,
    )
    fit_indices, test_indices = grouped_context_split(
        len(contexts),
        seed=seed,
    )
    train_beliefs = beliefs[fit_indices].reshape(-1, beliefs.shape[-1])
    test_beliefs = beliefs[test_indices].reshape(-1, beliefs.shape[-1])
    train_features = _features_for_indices(
        trained_features,
        fit_indices,
        context_length=context_length,
    )
    test_features = _features_for_indices(
        trained_features,
        test_indices,
        context_length=context_length,
    )
    initial_train = _features_for_indices(
        initialization_features,
        fit_indices,
        context_length=context_length,
    )
    initial_test = _features_for_indices(
        initialization_features,
        test_indices,
        context_length=context_length,
    )
    train_groups = np.repeat(fit_indices, context_length)
    test_groups = np.repeat(test_indices, context_length)
    train_next_tokens = _rows_for_indices(
        exact_next_tokens,
        fit_indices,
        context_length=context_length,
    )
    test_next_tokens = _rows_for_indices(
        exact_next_tokens,
        test_indices,
        context_length=context_length,
    )
    train_history = _rows_for_indices(
        history,
        fit_indices,
        context_length=context_length,
    )
    test_history = _rows_for_indices(
        history,
        test_indices,
        context_length=context_length,
    )
    train_current = _rows_for_indices(
        current_tokens,
        fit_indices,
        context_length=context_length,
    )
    test_current = _rows_for_indices(
        current_tokens,
        test_indices,
        context_length=context_length,
    )
    train_keys = _rows_for_indices(
        history_keys,
        fit_indices,
        context_length=context_length,
    )
    test_keys = _rows_for_indices(
        history_keys,
        test_indices,
        context_length=context_length,
    )
    null_basis = prediction_null_basis(_prediction_map())
    battery = evaluate_belief_geometry(
        train_features,
        test_features,
        train_beliefs,
        test_beliefs,
        train_groups=train_groups,
        test_groups=test_groups,
        nuisance_features={
            "current_token": (train_current, test_current),
            "exact_next_token": (train_next_tokens, test_next_tokens),
            "log_exact_next_token": (
                np.log(np.clip(train_next_tokens, 1e-12, 1.0)),
                np.log(np.clip(test_next_tokens, 1e-12, 1.0)),
            ),
            "three_token_suffix": (train_history, test_history),
        },
        contrasts={
            f"next_token_invisible_{index + 1}": null_basis[:, index]
            for index in range(null_basis.shape[1])
        },
        initialization_features={
            name: (initial_train[name], initial_test[name])
            for name in train_features
        },
        matched_keys=(train_keys, test_keys),
        seed=seed,
        n_null_repeats=1 if smoke else 5,
        n_resamples=20 if smoke else 200,
    )
    concatenated_prediction = battery.predictions["concatenated_layers"]
    pairwise, pairwise_arrays = pairwise_geometry_metrics(
        test_beliefs,
        concatenated_prediction,
    )
    report = {
        "target": {
            "name": "exact source-state posterior after the current token",
            "timing": (
                "activation at position t consumes token x_t and is paired "
                "with p(source_state_after_x_t | x_0:t)"
            ),
            "n_states": 5,
            "reachable_belief_states": 36,
        },
        "dataset": {
            "all_positive_probability_contexts": int(len(contexts)),
            "context_length": int(context_length),
            "positions_per_context": int(context_length),
            "fit_fraction": 0.2,
            "fit_contexts": int(len(fit_indices)),
            "test_contexts": int(len(test_indices)),
            "split_unit": "complete length-10 context",
            "weighting": (
                "unweighted all positive-probability input-position pairs"
            ),
        },
        "geometry": battery.report,
        "paper_pairwise_distance_control": pairwise,
        "headline_representation": "concatenated_layers",
    }
    return report, battery, test_beliefs, pairwise_arrays


@torch.no_grad()
def exact_task_metrics(
    model: RRXORPaperTransformer,
    paths: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    batch_size: int,
) -> dict[str, float]:
    """Compute exact path-weighted next-token CE and greedy accuracy."""

    was_training = model.training
    model.eval()
    weighted_loss = torch.zeros(
        (),
        dtype=probabilities.dtype,
        device=paths.device,
    )
    weighted_accuracy = torch.zeros_like(weighted_loss)
    normalization = probabilities.sum()
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        logits = model.forward(batch_paths[:, :-1])
        targets = batch_paths[:, 1:]
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape(len(batch_paths), -1)
        accuracies = logits.argmax(dim=-1).eq(targets).double()
        weights = probabilities[start : start + len(batch_paths)]
        weighted_loss += (weights * losses.mean(dim=-1)).sum()
        weighted_accuracy += (weights * accuracies.mean(dim=-1)).sum()
    if was_training:
        model.train()
    return {
        "cross_entropy_nats": float((weighted_loss / normalization).cpu()),
        "greedy_accuracy": float((weighted_accuracy / normalization).cpu()),
    }


def plot_belief_geometry(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    path: Path,
) -> None:
    """Project the five-state simplex to two dimensions for inspection."""

    centered = target - target.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    projection = right[:2].T
    target_xy = centered @ projection
    prediction_xy = (prediction - target.mean(axis=0, keepdims=True)) @ projection
    rounded = np.round(target, decimals=12)
    _, labels = np.unique(rounded, axis=0, return_inverse=True)
    colors = plt.get_cmap("hsv")(labels / max(1, labels.max() + 1))
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
    axes[0].scatter(
        target_xy[:, 0],
        target_xy[:, 1],
        c=colors,
        s=2.0,
        alpha=0.35,
        linewidths=0,
        rasterized=True,
    )
    axes[0].set_title("Exact RRXOR beliefs")
    axes[1].scatter(
        prediction_xy[:, 0],
        prediction_xy[:, 1],
        c=colors,
        s=2.0,
        alpha=0.25,
        linewidths=0,
        rasterized=True,
    )
    for label in np.unique(labels):
        center = prediction_xy[labels == label].mean(axis=0)
        axes[1].scatter(
            center[0],
            center[1],
            color=plt.get_cmap("hsv")(label / max(1, labels.max() + 1)),
            edgecolor="black",
            linewidth=0.25,
            s=24,
        )
    axes[1].set_title("Affine decode from concatenated layers")
    for axis in axes:
        axis.set_aspect("equal")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_layer_mse(report: dict[str, Any], *, path: Path) -> None:
    """Plot the paper's layerwise versus concatenated affine-probe comparison."""

    representations = report["geometry"]["representations"]
    names = [
        "layer_1",
        "layer_2",
        "layer_3",
        "layer_4",
        "concatenated_layers",
    ]
    values = [representations[name]["metrics"]["mse"] for name in names]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(
        ["L1", "L2", "L3", "L4", "Concat"],
        values,
        color=["#8fb9d4"] * 4 + ["#d5845a"],
    )
    axis.set_ylabel("Held-out affine-probe MSE")
    axis.set_title("RRXOR belief geometry across residual layers")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_pairwise_distances(
    arrays: dict[str, np.ndarray],
    metrics: dict[str, Any],
    *,
    path: Path,
) -> None:
    """Recreate the belief-distance versus next-token-distance control."""

    decoded = arrays["decoded_distances"]
    predictors = [
        (
            arrays["belief_distances"],
            "Ground-truth belief distance",
            metrics["belief_distance_to_decoded_distance_r_squared"],
        ),
        (
            arrays["next_token_distances"],
            "Ground-truth next-token distance",
            metrics["next_token_distance_to_decoded_distance_r_squared"],
        ),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for axis, (predictor, label, r_squared) in zip(axes, predictors):
        axis.scatter(predictor, decoded, s=5, alpha=0.35, linewidths=0)
        design = np.column_stack([predictor, np.ones(len(predictor))])
        coefficients, _, _, _ = np.linalg.lstsq(design, decoded, rcond=None)
        grid = np.linspace(predictor.min(), predictor.max(), 100)
        axis.plot(grid, coefficients[0] * grid + coefficients[1], color="black")
        axis.set_xlabel(label)
        axis.set_ylabel("Decoded belief distance")
        axis.set_title(f"R²={r_squared:.3f}")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_training_curve(
    history: list[dict[str, Any]],
    *,
    floor_nats: float,
    path: Path,
) -> None:
    """Plot sampled training CE and exact path-weighted validation CE."""

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
        label="exact finite-context Bayesian floor",
    )
    axis.set_xscale("log")
    axis.set_xlabel("optimizer update")
    axis.set_ylabel("cross-entropy (nats)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")
