"""Paper-faithful joint-versus-factored representation probe battery."""

from __future__ import annotations

from itertools import combinations
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.checkpoints import load_algorithm
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
from envs.hmm import HMMEnv
from harness.context import RunContext
from harness.seeding import named_seed_sequences, seed_sequence_to_int

from experiments.factored_representations_reproduction_2026_08.probe import (
    FactorProbeData,
    collect_probe_data,
    collect_vary_one_data,
)
from experiments.factored_representations_reproduction_2026_08.process import (
    FACTOR_CARDINALITY,
    decode_joint_tokens,
    environment_config,
    joint_token_count,
)


RCOND_VALUES = (1e-15, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
INTRINSIC_FACTOR_RANK = 2
FULL_PROBE_TRAIN_STEPS = 20_000
FULL_PROBE_TEST_STEPS = 20_000
SMOKE_PROBE_STEPS = 512
FULL_FROZEN_CONTEXTS = 10
FULL_REALIZATIONS_PER_CONTEXT = 100
SMOKE_FROZEN_CONTEXTS = 3
SMOKE_REALIZATIONS_PER_CONTEXT = 8
SEQUENCE_LENGTH = 8
_STREAM_KEYS = {
    "probe_train": (600,),
    "probe_test": (601,),
    "vary_one": (602,),
    "regression_cv": (603,),
}


def _array_summary(values: Any) -> dict[str, Any]:
    array = np.asarray(values)
    finite = np.isfinite(array)
    finite_values = array[finite]
    bad_indices = np.argwhere(~finite)[:12]
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "nan_count": int(np.isnan(array).sum()),
        "posinf_count": int(np.isposinf(array).sum()),
        "neginf_count": int(np.isneginf(array).sum()),
        "finite_min": float(finite_values.min()) if finite_values.size else None,
        "finite_max": float(finite_values.max()) if finite_values.size else None,
        "first_bad_indices": bad_indices.tolist(),
        "first_bad_values": [str(array[tuple(index)]) for index in bad_indices],
    }


# region agent log
def _debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
) -> None:
    with open("/opt/cursor/logs/debug.log", "a", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {
                    "hypothesisId": hypothesis_id,
                    "location": location,
                    "message": message,
                    "data": data,
                    "timestamp": int(time.time() * 1000),
                }
            )
            + "\n"
        )
# endregion


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fit_svd_affine(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    design = np.concatenate([np.ones((len(x), 1)), x], axis=1)
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=rcond)
    return coefficients[1:], coefficients[0]


def _predict(
    weight: np.ndarray,
    bias: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    return np.asarray(features, dtype=np.float64) @ weight + bias


def cross_validated_svd_affine(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    folds: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select the paper's SVD threshold on fit data, then refit once."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("features and targets must be aligned matrices")
    if len(x) < 4:
        raise ValueError("cross-validated regression requires at least four samples")
    fold_count = min(int(folds), len(x))
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(x))
    assignments = np.arange(len(x)) % fold_count
    assignments = assignments[np.argsort(permutation)]
    mean_errors: dict[str, float] = {}
    for rcond in RCOND_VALUES:
        errors = []
        for fold in range(fold_count):
            test = assignments == fold
            train = ~test
            weight, bias = _fit_svd_affine(
                x[train],
                y[train],
                rcond=rcond,
            )
            residual = _predict(weight, bias, x[test]) - y[test]
            errors.append(float(np.mean(np.square(residual))))
        mean_errors[str(rcond)] = float(np.mean(errors))
    selected = min(RCOND_VALUES, key=lambda value: mean_errors[str(value)])
    weight, bias = _fit_svd_affine(x, y, rcond=selected)
    return weight, bias, {
        "method": "10_fold_cv_svd_pseudoinverse",
        "rcond_candidates": list(RCOND_VALUES),
        "mean_validation_mse": mean_errors,
        "selected_rcond": selected,
        "folds": fold_count,
    }


def _factor_names(factor_count: int) -> tuple[str, ...]:
    return tuple(f"factor_{index}" for index in range(factor_count))


def _concatenated_factors(data: FactorProbeData) -> np.ndarray:
    return data.factor_beliefs.reshape(len(data.factor_beliefs), -1)


def _regression_report(
    train: FactorProbeData,
    test: FactorProbeData,
    *,
    factor_count: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train_target = _concatenated_factors(train)
    test_target = _concatenated_factors(test)
    weight, bias, selection = cross_validated_svd_affine(
        train.activations,
        train_target,
        seed=seed,
    )
    predicted = _predict(weight, bias, test.activations)
    # region agent log
    _debug_log(
        "C",
        "analysis.py:_regression_report",
        "Regression inputs and outputs",
        {
            "train_activations": _array_summary(train.activations),
            "train_target": _array_summary(train_target),
            "weight": _array_summary(weight),
            "bias": _array_summary(bias),
            "predicted": _array_summary(predicted),
        },
    )
    # endregion
    report: dict[str, Any] = {
        "target": "concatenated_exact_factor_predictive_vectors",
        "fit": selection,
        **global_mse_metrics(predicted, test_target),
        "rmse": float(np.sqrt(np.mean(np.square(predicted - test_target)))),
        "r_squared": r2_score(predicted, test_target),
        "per_factor": {},
    }
    subspaces = {}
    for factor, name in enumerate(_factor_names(factor_count)):
        columns = slice(
            factor * FACTOR_CARDINALITY,
            (factor + 1) * FACTOR_CARDINALITY,
        )
        factor_prediction = predicted[:, columns]
        factor_target = test_target[:, columns]
        report["per_factor"][name] = {
            **global_mse_metrics(factor_prediction, factor_target),
            "rmse": float(
                np.sqrt(np.mean(np.square(factor_prediction - factor_target)))
            ),
            "r_squared": r2_score(factor_prediction, factor_target),
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
        "union_rank": int(
            np.linalg.matrix_rank(np.concatenate(tuple(subspaces.values()), axis=1))
        ),
    }
    return report, subspaces


def _cev_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_curve = np.asarray(left["cumulative_explained_variance"], dtype=np.float64)
    right_curve = np.asarray(
        right["cumulative_explained_variance"],
        dtype=np.float64,
    )
    width = max(len(left_curve), len(right_curve))
    left_curve = np.pad(left_curve, (0, width - len(left_curve)), constant_values=1.0)
    right_curve = np.pad(
        right_curve,
        (0, width - len(right_curve)),
        constant_values=1.0,
    )
    return float(np.sqrt(np.mean(np.square(left_curve - right_curve))))


def _variance_report(
    test: FactorProbeData,
    *,
    factor_count: int,
) -> dict[str, Any]:
    # region agent log
    _debug_log(
        "A,B",
        "analysis.py:_variance_report",
        "Variance geometry inputs",
        {
            "activations": _array_summary(test.activations),
            "factor_beliefs": _array_summary(_concatenated_factors(test)),
            "joint_beliefs": _array_summary(test.joint_beliefs),
        },
    )
    # endregion
    activation = variance_geometry(test.activations)
    factored_target = variance_geometry(_concatenated_factors(test))
    joint_target = variance_geometry(test.joint_beliefs)
    return {
        "activation": activation,
        "factored_target": factored_target,
        "joint_target": joint_target,
        "algebraic_dimension_predictions": representation_dimension_predictions(
            [FACTOR_CARDINALITY] * factor_count
        ),
        "activation_cev_rmse_to_factored_target": _cev_distance(
            activation,
            factored_target,
        ),
        "activation_cev_rmse_to_joint_target": _cev_distance(
            activation,
            joint_target,
        ),
        "warning": (
            "CEV identifies effective dimensionality, not factor identity or "
            "orthogonality."
        ),
    }


def _top_basis(centered: np.ndarray, rank: int) -> np.ndarray:
    _, _, right = np.linalg.svd(
        np.asarray(centered, dtype=np.float64),
        full_matrices=False,
    )
    return right[:rank].T


def _vary_one_report(
    module: Any,
    train: FactorProbeData,
    test: FactorProbeData,
    *,
    factor_count: int,
    frozen_contexts: int,
    realizations_per_context: int,
    seed: int,
) -> dict[str, Any]:
    varied = collect_vary_one_data(
        module,
        factor_count=factor_count,
        frozen_contexts=frozen_contexts,
        realizations_per_context=realizations_per_context,
        sequence_length=SEQUENCE_LENGTH,
        seed=seed,
    )
    # region agent log
    _debug_log(
        "D",
        "analysis.py:_vary_one_report",
        "Vary-one activation inputs",
        {
            name: _array_summary(values)
            for name, values in varied.activations.items()
        },
    )
    # endregion
    centered = {
        name: center_within_groups(values, varied.groups[name])
        for name, values in varied.activations.items()
    }
    geometries = {
        name: variance_geometry(values) for name, values in centered.items()
    }
    bases = {
        name: _top_basis(values, INTRINSIC_FACTOR_RANK)
        for name, values in centered.items()
    }
    max_k = min(8, *(values.shape[1] for values in centered.values()))
    overlap_curves: dict[str, list[float]] = {}
    for left, right in combinations(_factor_names(factor_count), 2):
        left_all = _top_basis(centered[left], max_k)
        right_all = _top_basis(centered[right], max_k)
        overlap_curves[f"{left}_vs_{right}"] = [
            float(
                np.square(
                    left_all[:, :rank].T @ right_all[:, :rank]
                ).sum()
                / rank
            )
            for rank in range(1, max_k + 1)
        ]

    projected_regression = {}
    for factor, name in enumerate(_factor_names(factor_count)):
        basis = bases[name]
        train_projection = train.activations @ basis
        test_projection = test.activations @ basis
        train_target = train.factor_beliefs[:, factor, :]
        test_target = test.factor_beliefs[:, factor, :]
        weight, bias, selection = cross_validated_svd_affine(
            train_projection,
            train_target,
            seed=seed + factor + 1,
        )
        prediction = _predict(weight, bias, test_projection)
        projected_regression[name] = {
            "projection_rank": INTRINSIC_FACTOR_RANK,
            "fit": selection,
            **global_mse_metrics(prediction, test_target),
            "rmse": float(
                np.sqrt(np.mean(np.square(prediction - test_target)))
            ),
            "r_squared": r2_score(prediction, test_target),
        }

    intrinsic_overlaps = pairwise_subspace_overlaps(bases)
    return {
        "centering": "within_frozen_context_and_sequence_position",
        "frozen_contexts": frozen_contexts,
        "realizations_per_context": realizations_per_context,
        "sequence_length_excluding_bos": SEQUENCE_LENGTH,
        "per_factor_cev": geometries,
        "dimension_additivity": dimension_additivity(centered),
        "intrinsic_rank": INTRINSIC_FACTOR_RANK,
        "intrinsic_pairwise_overlap": intrinsic_overlaps,
        "intrinsic_mean_pairwise_overlap": float(
            np.mean(tuple(intrinsic_overlaps.values()))
        ),
        "overlap_curve_k": list(range(1, max_k + 1)),
        "pairwise_overlap_curves": overlap_curves,
        "projected_belief_regression": projected_regression,
        "interpretation": (
            "Projected regression checks that the identified two-dimensional "
            "vary-one subspace carries the anticipated factor geometry."
        ),
    }


def _embedding_report(module: Any, *, factor_count: int) -> dict[str, Any]:
    embeddings = (
        module.encoder.token_embedding_matrix()
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
    )
    # region agent log
    _debug_log(
        "E",
        "analysis.py:_embedding_report",
        "Token embedding inputs",
        {"embeddings": _array_summary(embeddings)},
    )
    # endregion
    subtokens = decode_joint_tokens(
        np.arange(joint_token_count(factor_count)),
        factor_count,
    )
    centered_by_factor = {}
    bases = {}
    for factor, name in enumerate(_factor_names(factor_count)):
        others = np.delete(subtokens, factor, axis=1)
        _, groups = np.unique(others, axis=0, return_inverse=True)
        centered = center_within_groups(embeddings, groups)
        centered_by_factor[name] = centered
        bases[name] = _top_basis(centered, INTRINSIC_FACTOR_RANK)

    centered = embeddings - embeddings.mean(axis=0)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    component_count = min(
        embeddings.shape[0] - 1,
        embeddings.shape[1],
        2 * factor_count + 4,
    )
    scores = centered @ right[:component_count].T
    attribution = np.zeros((component_count, factor_count), dtype=np.float64)
    for component in range(component_count):
        total = float(np.mean(np.square(scores[:, component])))
        for factor in range(factor_count):
            group_means = np.array(
                [
                    scores[subtokens[:, factor] == subtoken, component].mean()
                    for subtoken in range(FACTOR_CARDINALITY)
                ]
            )
            explained = group_means[subtokens[:, factor]]
            attribution[component, factor] = (
                0.0
                if total <= 0.0
                else float(np.mean(np.square(explained)) / total)
            )

    overlaps = pairwise_subspace_overlaps(bases)
    return {
        "token_embedding_cev": variance_geometry(embeddings),
        "predicted_factored_dimension": factor_count * INTRINSIC_FACTOR_RANK,
        "vary_one_dimension_additivity": dimension_additivity(centered_by_factor),
        "intrinsic_pairwise_overlap": overlaps,
        "intrinsic_mean_pairwise_overlap": float(
            np.mean(tuple(overlaps.values()))
        ),
        "pc_factor_attribution": {
            "definition": (
                "between-subtoken-ID variance divided by total score variance"
            ),
            "rows": [f"pc_{index + 1}" for index in range(component_count)],
            "columns": list(_factor_names(factor_count)),
            "values": attribution.tolist(),
        },
    }


def analyze_checkpoint(
    context: RunContext,
    *,
    checkpoint: Path,
    factor_count: int,
    condition: str,
    checkpoint_label: str,
    agent_steps: int,
    training_iteration: int,
) -> dict[str, Any]:
    """Run the full complementary analysis battery on one checkpoint."""

    # region agent log
    _debug_log(
        "A,B,C,D,E",
        "analysis.py:analyze_checkpoint",
        "Checkpoint analysis entry",
        {
            "checkpoint": str(checkpoint),
            "factor_count": factor_count,
            "checkpoint_label": checkpoint_label,
            "agent_steps": agent_steps,
            "training_iteration": training_iteration,
            "smoke": context.smoke,
        },
    )
    # endregion
    if context.seed is None:
        raise ValueError("factor analysis requires a resolved seed")
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    probe_steps = (
        SMOKE_PROBE_STEPS
        if context.smoke
        else FULL_PROBE_TRAIN_STEPS
    )
    test_steps = SMOKE_PROBE_STEPS if context.smoke else FULL_PROBE_TEST_STEPS
    frozen_contexts = (
        SMOKE_FROZEN_CONTEXTS if context.smoke else FULL_FROZEN_CONTEXTS
    )
    realizations = (
        SMOKE_REALIZATIONS_PER_CONTEXT
        if context.smoke
        else FULL_REALIZATIONS_PER_CONTEXT
    )
    config = environment_config(factor_count)
    config["diagnostics"] = {"belief": True, "tokens": True}

    with load_algorithm(checkpoint) as algorithm:
        module = algorithm.get_module()
        if module is None:
            raise KeyError("checkpoint has no default RLModule")

        def make_environment():
            return HMMEnv(config)

        common = {
            "module": module,
            "env_factory": make_environment,
            "factor_count": factor_count,
            "n_envs": 8,
            "warmup": 1,
        }
        train = collect_probe_data(
            n_steps=probe_steps,
            seed=streams["probe_train"],
            **common,
        )
        test = collect_probe_data(
            n_steps=test_steps,
            seed=streams["probe_test"],
            **common,
        )
        # region agent log
        _debug_log(
            "A,B",
            "analysis.py:analyze_checkpoint",
            "Collected train and test probe data",
            {
                "train_activations": _array_summary(train.activations),
                "test_activations": _array_summary(test.activations),
                "train_joint_beliefs": _array_summary(train.joint_beliefs),
                "test_joint_beliefs": _array_summary(test.joint_beliefs),
                "train_factor_beliefs": _array_summary(train.factor_beliefs),
                "test_factor_beliefs": _array_summary(test.factor_beliefs),
                "train_rewards": _array_summary(train.rewards),
                "test_rewards": _array_summary(test.rewards),
            },
        )
        # endregion
        consistency = max(
            train.product_consistency_max_abs,
            test.product_consistency_max_abs,
        )
        if consistency > 1e-10:
            raise AssertionError(
                "independent-factor posterior does not equal product of "
                f"marginals: {consistency:.3e}"
            )
        regression, _ = _regression_report(
            train,
            test,
            factor_count=factor_count,
            seed=seed_sequence_to_int(streams["regression_cv"], bits=32),
        )
        result = {
            "condition": condition,
            "factor_count": factor_count,
            "checkpoint": checkpoint_label,
            "agent_steps": agent_steps,
            "training_iteration": training_iteration,
            "is_initialization": training_iteration == 0,
            "representation": (
                "final_transformer_block_residual_before_final_layer_norm"
            ),
            "sampling_distribution": (
                "process_weighted_length_8_episode_positions_excluding_bos"
            ),
            "n_fit": len(train.activations),
            "n_test": len(test.activations),
            "product_consistency_max_abs": consistency,
            "task_accuracy_greedy": float(test.rewards.mean()),
            "factor_decodability": regression,
            "variance_geometry": _variance_report(
                test,
                factor_count=factor_count,
            ),
            "vary_one": _vary_one_report(
                module,
                train,
                test,
                factor_count=factor_count,
                frozen_contexts=frozen_contexts,
                realizations_per_context=realizations,
                seed=seed_sequence_to_int(streams["vary_one"], bits=32),
            ),
            "token_embedding": _embedding_report(
                module,
                factor_count=factor_count,
            ),
            "scope_warning": (
                "These diagnostics establish geometry and decodability, not "
                "causal use by the PPO policy."
            ),
        }

    _json_write(context.results_dir / "probe_battery.json", result)
    return result


def plot_probe_trajectory(
    reports: list[dict[str, Any]],
    *,
    condition: str,
    factor_count: int,
    path: Path,
) -> None:
    steps = np.asarray([row["agent_steps"] for row in reports])
    dimensions = np.asarray(
        [
            row["variance_geometry"]["activation"]["cev95_dimension"]
            for row in reports
        ]
    )
    factored_dimension = 2 * factor_count
    joint_dimension = 3**factor_count - 1
    regression_rmse = np.asarray(
        [row["factor_decodability"]["rmse"] for row in reports]
    )
    overlaps = np.asarray(
        [row["vary_one"]["intrinsic_mean_pairwise_overlap"] for row in reports]
    )
    accuracies = 100.0 * np.asarray(
        [row["task_accuracy_greedy"] for row in reports]
    )

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), squeeze=False)
    axes[0, 0].plot(steps, dimensions, marker="o")
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
    axes[0, 1].plot(steps, regression_rmse, marker="o")
    axes[0, 1].set_ylabel("Held-out factor RMSE")
    axes[1, 0].plot(steps, overlaps, marker="o")
    axes[1, 0].set_ylabel("Mean rank-2 subspace overlap")
    axes[1, 0].set_ylim(bottom=0.0)
    axes[1, 1].plot(steps, accuracies, marker="o")
    axes[1, 1].set_ylabel("Greedy joint-token accuracy (%)")
    for axis in axes.flat:
        axis.set_xlabel("Environment steps")
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"{condition.replace('_', ' ')}, {factor_count} independent MESS3 factors"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
