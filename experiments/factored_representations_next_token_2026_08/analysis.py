"""Longitudinal geometry probes for pure next-token checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.probes import (
    center_within_groups,
    dimension_additivity,
    global_mse_metrics,
    pairwise_subspace_overlaps,
    readout_subspace,
    representation_dimension_predictions,
    r2_score,
    variance_geometry,
)

from .model import FactoredNextTokenTransformer, NextTokenModelConfig
from .process import (
    FACTOR_CARDINALITY,
    SEQUENCE_LENGTH,
    decode_joint_tokens,
    encode_joint_tokens,
    product_beliefs,
    sample_sequences,
)
from .training import language_model_io, load_checkpoint


RCOND_VALUES = (1e-15, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
INTRINSIC_FACTOR_RANK = 2
FULL_PROBE_SEQUENCES = 4_096
SMOKE_PROBE_SEQUENCES = 256
FULL_FROZEN_CONTEXTS = 10
FULL_REALIZATIONS_PER_CONTEXT = 100
SMOKE_FROZEN_CONTEXTS = 2
SMOKE_REALIZATIONS_PER_CONTEXT = 4


@dataclass(frozen=True, slots=True)
class ProbeData:
    activations: np.ndarray
    factor_beliefs: np.ndarray
    joint_beliefs: np.ndarray
    loss_nats: float
    accuracy: float
    bayes_loss_nats: float


def _json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _generator(device: torch.device, seed: int) -> torch.Generator | None:
    try:
        result = torch.Generator(device=device)
        result.manual_seed(seed)
        return result
    except RuntimeError:
        torch.manual_seed(seed)
        return None


@torch.inference_mode()
def collect_probe_data(
    model: FactoredNextTokenTransformer,
    *,
    factor_count: int,
    sequences: int,
    seed: int,
    device: torch.device,
) -> ProbeData:
    """Collect process-weighted pre-final-LN activations and exact beliefs."""

    batch = sample_sequences(
        batch_size=sequences,
        factor_count=factor_count,
        sequence_length=SEQUENCE_LENGTH,
        device=device,
        generator=_generator(device, seed),
    )
    inputs, targets = language_model_io(batch, model)
    logits, residuals = model(inputs, return_activations=True)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )
    accuracy = (logits.argmax(dim=-1) == targets).float().mean()
    bayes_loss = -batch.target_probabilities.log().mean()

    # Training inputs are BOS,x1,...,x7. Excluding BOS leaves activations
    # aligned with posterior beliefs after x1,...,x7.
    factor_beliefs = batch.factor_beliefs[:, 1:-1]
    joint_beliefs = product_beliefs(factor_beliefs)
    return ProbeData(
        activations=(
            residuals[:, 1:]
            .reshape(-1, model.config.d_model)
            .cpu()
            .double()
            .numpy()
        ),
        factor_beliefs=(
            factor_beliefs.reshape(-1, factor_count, FACTOR_CARDINALITY)
            .cpu()
            .double()
            .numpy()
        ),
        joint_beliefs=(
            joint_beliefs.reshape(-1, joint_beliefs.shape[-1])
            .cpu()
            .double()
            .numpy()
        ),
        loss_nats=float(loss.cpu()),
        accuracy=float(accuracy.cpu()),
        bayes_loss_nats=float(bayes_loss.cpu()),
    )


def _fit_svd_affine(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.concatenate([np.ones((len(features), 1)), features], axis=1)
    coefficients, _, _, _ = np.linalg.lstsq(design, targets, rcond=rcond)
    return coefficients[1:], coefficients[0]


def _predict(
    weight: np.ndarray,
    bias: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    return features @ weight + bias


def cross_validated_svd_affine(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    folds: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Choose the paper's SVD cutoff on fit data, then refit."""

    if features.ndim != 2 or targets.ndim != 2 or len(features) != len(targets):
        raise ValueError("features and targets must be aligned matrices")
    if len(features) < 4:
        raise ValueError("cross-validation requires at least four samples")
    fold_count = min(folds, len(features))
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(features))
    assignments = np.arange(len(features)) % fold_count
    assignments = assignments[np.argsort(permutation)]
    mean_errors: dict[str, float] = {}
    for rcond in RCOND_VALUES:
        errors = []
        for fold in range(fold_count):
            test = assignments == fold
            train = ~test
            weight, bias = _fit_svd_affine(
                features[train],
                targets[train],
                rcond=rcond,
            )
            residual = _predict(weight, bias, features[test]) - targets[test]
            errors.append(float(np.mean(np.square(residual))))
        mean_errors[str(rcond)] = float(np.mean(errors))
    selected = min(RCOND_VALUES, key=lambda value: mean_errors[str(value)])
    weight, bias = _fit_svd_affine(features, targets, rcond=selected)
    return weight, bias, {
        "method": "10_fold_cv_svd_pseudoinverse",
        "rcond_candidates": list(RCOND_VALUES),
        "mean_validation_mse": mean_errors,
        "selected_rcond": selected,
        "folds": fold_count,
    }


def _factor_names(factor_count: int) -> tuple[str, ...]:
    return tuple(f"factor_{index}" for index in range(factor_count))


def _concatenated_factors(data: ProbeData) -> np.ndarray:
    return data.factor_beliefs.reshape(len(data.factor_beliefs), -1)


def _regression_report(
    train: ProbeData,
    test: ProbeData,
    *,
    factor_count: int,
    seed: int,
) -> dict[str, Any]:
    train_targets = _concatenated_factors(train)
    test_targets = _concatenated_factors(test)
    weight, bias, fit = cross_validated_svd_affine(
        train.activations,
        train_targets,
        seed=seed,
    )
    prediction = _predict(weight, bias, test.activations)
    report: dict[str, Any] = {
        "target": "concatenated_exact_factor_predictive_vectors",
        "fit": fit,
        **global_mse_metrics(prediction, test_targets),
        "rmse": float(np.sqrt(np.mean(np.square(prediction - test_targets)))),
        "r_squared": r2_score(prediction, test_targets),
        "per_factor": {},
    }
    subspaces = {}
    for factor, name in enumerate(_factor_names(factor_count)):
        columns = slice(
            factor * FACTOR_CARDINALITY,
            (factor + 1) * FACTOR_CARDINALITY,
        )
        local_prediction = prediction[:, columns]
        local_target = test_targets[:, columns]
        report["per_factor"][name] = {
            **global_mse_metrics(local_prediction, local_target),
            "rmse": float(
                np.sqrt(np.mean(np.square(local_prediction - local_target)))
            ),
            "r_squared": r2_score(local_prediction, local_target),
        }
        subspaces[name] = readout_subspace(
            weight[:, columns],
            rank=INTRINSIC_FACTOR_RANK,
        )
    overlaps = pairwise_subspace_overlaps(subspaces)
    report["regression_subspaces"] = {
        "rank_per_factor": INTRINSIC_FACTOR_RANK,
        "pairwise_overlap": overlaps,
        "mean_pairwise_overlap": float(np.mean(tuple(overlaps.values()))),
    }
    return report


def _cev_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_curve = np.asarray(left["cumulative_explained_variance"])
    right_curve = np.asarray(right["cumulative_explained_variance"])
    width = max(len(left_curve), len(right_curve))
    left_curve = np.pad(left_curve, (0, width - len(left_curve)), constant_values=1)
    right_curve = np.pad(
        right_curve,
        (0, width - len(right_curve)),
        constant_values=1,
    )
    return float(np.sqrt(np.mean(np.square(left_curve - right_curve))))


def _variance_report(data: ProbeData, factor_count: int) -> dict[str, Any]:
    activation = variance_geometry(data.activations)
    factored = variance_geometry(_concatenated_factors(data))
    joint = variance_geometry(data.joint_beliefs)
    return {
        "activation": activation,
        "factored_target": factored,
        "joint_target": joint,
        "algebraic_dimension_predictions": representation_dimension_predictions(
            [FACTOR_CARDINALITY] * factor_count
        ),
        "activation_cev_rmse_to_factored_target": _cev_distance(
            activation,
            factored,
        ),
        "activation_cev_rmse_to_joint_target": _cev_distance(
            activation,
            joint,
        ),
    }


def _top_basis(centered: np.ndarray, rank: int) -> np.ndarray:
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    return right[:rank].T


@torch.inference_mode()
def _collect_vary_one(
    model: FactoredNextTokenTransformer,
    *,
    factor_count: int,
    frozen_contexts: int,
    realizations: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    generator = _generator(device, seed)
    activations: dict[str, np.ndarray] = {}
    groups: dict[str, np.ndarray] = {}
    positions = SEQUENCE_LENGTH - 1
    for varied_factor in range(factor_count):
        chunks = []
        for _ in range(frozen_contexts):
            batch = sample_sequences(
                batch_size=realizations,
                factor_count=factor_count,
                sequence_length=SEQUENCE_LENGTH,
                device=device,
                generator=generator,
            )
            subtokens = decode_joint_tokens(batch.tokens, factor_count)
            for factor in range(factor_count):
                if factor != varied_factor:
                    subtokens[:, :, factor] = subtokens[0:1, :, factor]
            tokens = encode_joint_tokens(subtokens)
            bos = torch.full(
                (realizations, 1),
                model.config.bos_token,
                dtype=torch.long,
                device=device,
            )
            inputs = torch.cat([bos, tokens[:, :-1]], dim=1)
            _, residuals = model(inputs, return_activations=True)
            chunks.append(
                residuals[:, 1:]
                .reshape(-1, model.config.d_model)
                .cpu()
                .double()
                .numpy()
            )
        name = f"factor_{varied_factor}"
        activations[name] = np.concatenate(chunks)
        fixed_context = np.repeat(
            np.arange(frozen_contexts),
            realizations * positions,
        )
        position = np.tile(
            np.arange(positions),
            frozen_contexts * realizations,
        )
        groups[name] = fixed_context * positions + position
    return activations, groups


def _vary_one_report(
    model: FactoredNextTokenTransformer,
    train: ProbeData,
    test: ProbeData,
    *,
    factor_count: int,
    frozen_contexts: int,
    realizations: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    activations, groups = _collect_vary_one(
        model,
        factor_count=factor_count,
        frozen_contexts=frozen_contexts,
        realizations=realizations,
        seed=seed,
        device=device,
    )
    centered = {
        name: center_within_groups(values, groups[name])
        for name, values in activations.items()
    }
    bases = {
        name: _top_basis(values, INTRINSIC_FACTOR_RANK)
        for name, values in centered.items()
    }
    overlaps = pairwise_subspace_overlaps(bases)
    overlap_curves = {}
    max_rank = min(8, model.config.d_model)
    for left, right in combinations(_factor_names(factor_count), 2):
        left_basis = _top_basis(centered[left], max_rank)
        right_basis = _top_basis(centered[right], max_rank)
        overlap_curves[f"{left}_vs_{right}"] = [
            float(
                np.square(
                    left_basis[:, :rank].T @ right_basis[:, :rank]
                ).sum()
                / rank
            )
            for rank in range(1, max_rank + 1)
        ]

    projected = {}
    for factor, name in enumerate(_factor_names(factor_count)):
        basis = bases[name]
        weight, bias, fit = cross_validated_svd_affine(
            train.activations @ basis,
            train.factor_beliefs[:, factor, :],
            seed=seed + factor + 1,
        )
        target = test.factor_beliefs[:, factor, :]
        prediction = _predict(weight, bias, test.activations @ basis)
        projected[name] = {
            "projection_rank": INTRINSIC_FACTOR_RANK,
            "fit": fit,
            **global_mse_metrics(prediction, target),
            "rmse": float(np.sqrt(np.mean(np.square(prediction - target)))),
            "r_squared": r2_score(prediction, target),
        }
    return {
        "centering": "within_frozen_context_and_sequence_position",
        "frozen_contexts": frozen_contexts,
        "realizations_per_context": realizations,
        "positions_excluding_bos": SEQUENCE_LENGTH - 1,
        "per_factor_cev": {
            name: variance_geometry(values) for name, values in centered.items()
        },
        "dimension_additivity": dimension_additivity(centered),
        "intrinsic_rank": INTRINSIC_FACTOR_RANK,
        "intrinsic_pairwise_overlap": overlaps,
        "intrinsic_mean_pairwise_overlap": float(
            np.mean(tuple(overlaps.values()))
        ),
        "overlap_curve_rank": list(range(1, max_rank + 1)),
        "pairwise_overlap_curves": overlap_curves,
        "projected_belief_regression": projected,
    }


def _embedding_report(
    model: FactoredNextTokenTransformer,
    factor_count: int,
) -> dict[str, Any]:
    embeddings = (
        model.token_embedding_matrix().detach().cpu().double().numpy()
    )
    tokens = torch.arange(model.config.base_vocab_size, dtype=torch.long)
    subtokens = decode_joint_tokens(tokens, factor_count).numpy()
    centered_by_factor = {}
    bases = {}
    for factor, name in enumerate(_factor_names(factor_count)):
        others = np.delete(subtokens, factor, axis=1)
        _, groups = np.unique(others, axis=0, return_inverse=True)
        centered = center_within_groups(embeddings, groups)
        centered_by_factor[name] = centered
        bases[name] = _top_basis(centered, INTRINSIC_FACTOR_RANK)
    overlaps = pairwise_subspace_overlaps(bases)
    return {
        "token_embedding_cev": variance_geometry(embeddings),
        "predicted_factored_dimension": factor_count * INTRINSIC_FACTOR_RANK,
        "vary_one_dimension_additivity": dimension_additivity(centered_by_factor),
        "intrinsic_pairwise_overlap": overlaps,
        "intrinsic_mean_pairwise_overlap": float(
            np.mean(tuple(overlaps.values()))
        ),
    }


def analyze_checkpoint(
    *,
    checkpoint: Path,
    factor_count: int,
    update: int,
    seed: int,
    smoke: bool,
    device: torch.device,
    results_dir: Path,
) -> dict[str, Any]:
    """Load one pure-PyTorch checkpoint and run the paper-inspired battery."""

    model = FactoredNextTokenTransformer(
        NextTokenModelConfig(factor_count=factor_count)
    ).to(device)
    load_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        generator=None,
        device=device,
    )
    model.eval()
    sequences = SMOKE_PROBE_SEQUENCES if smoke else FULL_PROBE_SEQUENCES
    frozen = SMOKE_FROZEN_CONTEXTS if smoke else FULL_FROZEN_CONTEXTS
    realizations = (
        SMOKE_REALIZATIONS_PER_CONTEXT
        if smoke
        else FULL_REALIZATIONS_PER_CONTEXT
    )
    train = collect_probe_data(
        model,
        factor_count=factor_count,
        sequences=sequences,
        seed=seed + 101,
        device=device,
    )
    test = collect_probe_data(
        model,
        factor_count=factor_count,
        sequences=sequences,
        seed=seed + 202,
        device=device,
    )
    report = {
        "objective": "pure_shifted_next_token_cross_entropy",
        "factor_count": factor_count,
        "checkpoint": checkpoint.name,
        "update": update,
        "is_initialization": update == 0,
        "representation": (
            "final_transformer_block_residual_before_final_layer_norm"
        ),
        "sampling_distribution": (
            "independent_MESS3_length_8_sequences; BOS excluded from geometry"
        ),
        "n_fit_activations": len(train.activations),
        "n_test_activations": len(test.activations),
        "next_token_prediction": {
            "loss_nats": test.loss_nats,
            "accuracy": test.accuracy,
            "bayes_loss_nats": test.bayes_loss_nats,
            "gap_nats": test.loss_nats - test.bayes_loss_nats,
        },
        "factor_decodability": _regression_report(
            train,
            test,
            factor_count=factor_count,
            seed=seed + 303,
        ),
        "variance_geometry": _variance_report(test, factor_count),
        "vary_one": _vary_one_report(
            model,
            train,
            test,
            factor_count=factor_count,
            frozen_contexts=frozen,
            realizations=realizations,
            seed=seed + 404,
            device=device,
        ),
        "token_embedding": _embedding_report(model, factor_count),
    }
    _json_write(results_dir / "probe_battery.json", report)
    return report


def plot_probe_trajectory(
    reports: list[dict[str, Any]],
    *,
    factor_count: int,
    path: Path,
) -> None:
    updates = np.asarray([report["update"] for report in reports])
    dimensions = np.asarray(
        [
            report["variance_geometry"]["activation"]["cev95_dimension"]
            for report in reports
        ]
    )
    rmse = np.asarray(
        [report["factor_decodability"]["rmse"] for report in reports]
    )
    overlap = np.asarray(
        [
            report["vary_one"]["intrinsic_mean_pairwise_overlap"]
            for report in reports
        ]
    )
    gap = np.asarray(
        [report["next_token_prediction"]["gap_nats"] for report in reports]
    )
    factored_dimension = 2 * factor_count
    joint_dimension = 3**factor_count - 1
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), squeeze=False)
    axes[0, 0].plot(updates, dimensions, marker="o")
    axes[0, 0].axhline(
        factored_dimension,
        linestyle="--",
        color="#2d7d46",
        label=f"factored ({factored_dimension})",
    )
    axes[0, 0].axhline(
        joint_dimension,
        linestyle=":",
        color="#9a3c3c",
        label=f"joint ({joint_dimension})",
    )
    axes[0, 0].set_ylabel("Dimensions for 95% CEV")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(updates, rmse, marker="o")
    axes[0, 1].set_ylabel("Held-out factor RMSE")
    axes[1, 0].plot(updates, overlap, marker="o")
    axes[1, 0].set_ylabel("Mean rank-2 overlap")
    axes[1, 1].plot(updates, gap, marker="o")
    axes[1, 1].set_ylabel("Validation gap to Bayes (nats)")
    for axis in axes.flat:
        axis.set_xlabel("Optimizer updates")
        axis.set_xscale("symlog", linthresh=1)
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"Pure next-token training, {factor_count} independent MESS3 factors"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
