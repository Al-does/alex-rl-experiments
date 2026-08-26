"""Study-local representation controls for factored MESS3 transformers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.checkpoints import load_algorithm
from analysis.probes.factorization import (
    center_within_groups,
    dimension_additivity,
    pairwise_subspace_overlaps,
    rowwise_tensor_product,
    variance_geometry,
    vary_one_subspace,
)
from analysis.probes.linear import (
    fit_affine_probe,
    probe_predict,
    r2_score,
)
from experiments.mess3_supervised.paper_supervised_replication.mess3 import (
    bayesian_beliefs,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext

from .shared import (
    CONTEXT_LENGTH,
    FACTOR_CARDINALITY,
    StudyCondition,
    SupervisedNextTokenModel,
    _device,
    _generator,
    combine_factor_tokens,
    make_next_token_batch,
    sample_factor_paths,
)


RIDGE = 1e-6
SMOKE_TRAIN_SAMPLES = 384
SMOKE_TEST_SAMPLES = 256
FULL_TRAIN_SAMPLES = 6_000
FULL_TEST_SAMPLES = 6_000
SMOKE_FIXED_CONTEXTS = 6
SMOKE_VARIANTS = 8
FULL_FIXED_CONTEXTS = 32
FULL_VARIANTS = 32


@dataclass(frozen=True, slots=True)
class RepresentationData:
    activations: np.ndarray
    factor_beliefs: tuple[np.ndarray, ...]
    joint_beliefs: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    token_histories: np.ndarray
    critic_values: np.ndarray | None


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().double().numpy()


def factor_beliefs_from_paths(
    factor_paths: torch.Tensor,
) -> tuple[np.ndarray, ...]:
    """Return exact beliefs after the final observed subtoken for each factor."""

    if factor_paths.ndim != 3:
        raise ValueError("factor_paths must have shape (batch, time, factor)")
    return tuple(
        _to_numpy(
            bayesian_beliefs(
                factor_paths[:, :, factor_index],
            )[:, -1]
        )
        for factor_index in range(factor_paths.shape[-1])
    )


def _history_identity(
    tokens: torch.Tensor,
    vocabulary_size: int,
) -> np.ndarray:
    return _to_numpy(
        F.one_hot(tokens, num_classes=vocabulary_size)
        .reshape(len(tokens), -1)
        .to(dtype=torch.float32)
    )


@torch.inference_mode()
def extract_supervised_data(
    model: SupervisedNextTokenModel,
    tokens: torch.Tensor,
    factor_paths: torch.Tensor,
) -> RepresentationData:
    """Extract aligned final-position quantities from the CE-only model."""

    logits, activations = model(tokens, return_activations=True)
    final_logits = logits[:, -1]
    factors = factor_beliefs_from_paths(factor_paths)
    return RepresentationData(
        activations=_to_numpy(activations[:, -1]),
        factor_beliefs=factors,
        joint_beliefs=rowwise_tensor_product(factors),
        logits=_to_numpy(final_logits),
        probabilities=_to_numpy(torch.softmax(final_logits, dim=-1)),
        token_histories=_history_identity(
            tokens,
            model.condition.vocabulary_size,
        ),
        critic_values=None,
    )


def _activation_encoding(
    train: RepresentationData,
    test: RepresentationData,
) -> dict[str, Any]:
    """Reverse-regress activations from additive and full-joint beliefs."""

    train_additive = np.concatenate(train.factor_beliefs, axis=1)
    test_additive = np.concatenate(test.factor_beliefs, axis=1)
    additive_weight, additive_bias = fit_affine_probe(
        train_additive,
        train.activations,
        ridge=RIDGE,
    )
    additive_prediction = probe_predict(
        additive_weight,
        additive_bias,
        test_additive,
    )
    joint_weight, joint_bias = fit_affine_probe(
        train.joint_beliefs,
        train.activations,
        ridge=RIDGE,
    )
    joint_prediction = probe_predict(
        joint_weight,
        joint_bias,
        test.joint_beliefs,
    )
    additive_r2 = r2_score(additive_prediction, test.activations)
    joint_r2 = r2_score(joint_prediction, test.activations)
    residual = test.activations - additive_prediction
    return {
        "fit": "held_out_affine_ridge",
        "ridge": RIDGE,
        "n_train": len(train.activations),
        "n_test": len(test.activations),
        "additive_factor_belief_activation_r_squared": additive_r2,
        "full_joint_belief_activation_r_squared": joint_r2,
        "extra_variance_explained_by_joint_interactions": joint_r2 - additive_r2,
        "factor_predicted_activation_geometry": variance_geometry(
            additive_prediction,
            max_spectrum_entries=32,
        ),
        "factor_residual_activation_geometry": variance_geometry(
            residual,
            max_spectrum_entries=32,
        ),
        "interpretation": (
            "The joint-minus-additive held-out R² is the incremental activation "
            "variance attributable to belief interactions, without asserting "
            "that either encoding is causally used."
        ),
    }


def reduced_rank_curve(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_target: np.ndarray,
    test_target: np.ndarray,
    *,
    ridge: float = RIDGE,
) -> dict[str, Any]:
    """Compute held-out target R² as activation PCA rank increases."""

    train_x = np.asarray(train_features, dtype=np.float64)
    test_x = np.asarray(test_features, dtype=np.float64)
    train_y = np.asarray(train_target, dtype=np.float64)
    test_y = np.asarray(test_target, dtype=np.float64)
    if train_y.ndim == 1:
        train_y = train_y[:, None]
    if test_y.ndim == 1:
        test_y = test_y[:, None]
    if (
        train_x.ndim != 2
        or test_x.ndim != 2
        or len(train_x) != len(train_y)
        or len(test_x) != len(test_y)
        or train_x.shape[1] != test_x.shape[1]
        or train_y.shape[1] != test_y.shape[1]
    ):
        raise ValueError("features and targets must be aligned matrices")

    x_mean = train_x.mean(axis=0)
    y_mean = train_y.mean(axis=0)
    centered_x = train_x - x_mean
    centered_y = train_y - y_mean
    _, singular_values, right = np.linalg.svd(
        centered_x,
        full_matrices=False,
    )
    numerical_rank = int(
        np.count_nonzero(
            singular_values
            > np.finfo(np.float64).eps
            * max(centered_x.shape)
            * (singular_values[0] if len(singular_values) else 0.0)
        )
    )
    ranks = list(range(1, numerical_rank + 1))
    if not ranks:
        return {
            "ranks": [],
            "held_out_r_squared": [],
            "best_achievable_r_squared": 0.0,
            "rank_retaining_99pct_of_best": None,
        }

    train_scores = centered_x @ right[:numerical_rank].T
    test_scores = (test_x - x_mean) @ right[:numerical_rank].T
    denominator = np.square(train_scores).sum(axis=0) + ridge
    coefficients = train_scores.T @ centered_y
    coefficients = coefficients / denominator[:, None]
    prediction = np.repeat(y_mean[None, :], len(test_y), axis=0)
    scores: list[float] = []
    for component in range(numerical_rank):
        prediction = prediction + (
            test_scores[:, component, None]
            * coefficients[component][None, :]
        )
        scores.append(r2_score(prediction, test_y))
    best = max(0.0, max(scores))
    retained = (
        None
        if best <= 0.0
        else next(
            rank
            for rank, score in zip(ranks, scores)
            if score >= 0.99 * best
        )
    )
    return {
        "ranks": ranks,
        "held_out_r_squared": scores,
        "best_achievable_r_squared": best,
        "rank_retaining_99pct_of_best": retained,
        "ridge": ridge,
    }


def _rank_targets(
    train: RepresentationData,
    test: RepresentationData,
) -> dict[str, Any]:
    targets: dict[str, tuple[np.ndarray, np.ndarray] | None] = {
        "factor_beliefs": (
            np.concatenate(train.factor_beliefs, axis=1),
            np.concatenate(test.factor_beliefs, axis=1),
        ),
        "next_token_logits": (train.logits, test.logits),
        "next_token_probabilities": (
            train.probabilities,
            test.probabilities,
        ),
        "token_history_identity": (
            train.token_histories,
            test.token_histories,
        ),
        "critic_value": (
            None
            if train.critic_values is None or test.critic_values is None
            else (train.critic_values, test.critic_values)
        ),
    }
    report: dict[str, Any] = {}
    for name, values in targets.items():
        report[name] = (
            {
                "status": "absent",
                "reason": "the supervised CE control has no critic or value head",
            }
            if values is None
            else {
                "status": "measured",
                **reduced_rank_curve(
                    train.activations,
                    test.activations,
                    values[0],
                    values[1],
                ),
            }
        )
    return report


def construct_vary_one_factor_paths(
    fixed_contexts: torch.Tensor,
    varying_factor_paths: torch.Tensor,
    *,
    factor_index: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """Hold N-1 subtoken sequences fixed while replacing one sequence."""

    if fixed_contexts.ndim != 3:
        raise ValueError("fixed_contexts must have shape (group, time, factor)")
    if varying_factor_paths.ndim != 3:
        raise ValueError(
            "varying_factor_paths must have shape (group, variant, time)"
        )
    groups, steps, n_factors = fixed_contexts.shape
    if varying_factor_paths.shape[0] != groups:
        raise ValueError("varying paths must align with fixed groups")
    if varying_factor_paths.shape[2] != steps:
        raise ValueError("varying paths must use the same sequence length")
    if not 0 <= factor_index < n_factors:
        raise ValueError("factor_index is out of range")
    variants = varying_factor_paths.shape[1]
    controlled = (
        fixed_contexts[:, None]
        .expand(groups, variants, steps, n_factors)
        .clone()
    )
    controlled[:, :, :, factor_index] = varying_factor_paths
    return (
        controlled.reshape(groups * variants, steps, n_factors),
        np.repeat(np.arange(groups, dtype=np.int64), variants),
    )


def _vary_one_metrics(
    *,
    condition: StudyCondition,
    device: torch.device,
    seed: int,
    smoke: bool,
    activation_extractor: Callable[[torch.Tensor], np.ndarray],
) -> dict[str, Any]:
    fixed_count = SMOKE_FIXED_CONTEXTS if smoke else FULL_FIXED_CONTEXTS
    variants = SMOKE_VARIANTS if smoke else FULL_VARIANTS
    generator = _generator(device, seed)
    fixed = sample_factor_paths(
        fixed_count,
        n_factors=condition.n_factors,
        length=CONTEXT_LENGTH,
        device=device,
        generator=generator,
    )
    centered_by_factor: dict[str, np.ndarray] = {}
    bases: dict[str, np.ndarray] = {}
    factor_metrics: dict[str, Any] = {}
    for factor_index in range(condition.n_factors):
        candidates = sample_factor_paths(
            fixed_count * variants,
            n_factors=1,
            length=CONTEXT_LENGTH,
            device=device,
            generator=generator,
        )[:, :, 0].reshape(fixed_count, variants, CONTEXT_LENGTH)
        controlled, groups = construct_vary_one_factor_paths(
            fixed,
            candidates,
            factor_index=factor_index,
        )
        activations = activation_extractor(combine_factor_tokens(controlled))
        centered = center_within_groups(activations, groups)
        basis, geometry = vary_one_subspace(
            activations,
            groups,
            variance_fraction=0.95,
        )
        name = f"factor_{factor_index}"
        centered_by_factor[name] = centered
        bases[name] = basis
        cev95 = int(geometry["cev95_dimension"])
        cev99 = int(geometry["cev99_dimension"])
        factor_metrics[name] = {
            "fixed_context_groups": fixed_count,
            "variants_per_group": variants,
            "geometry": geometry,
            "requires_exactly_two_dimensions_at_cev95": cev95 == 2,
            "fits_within_two_dimensions_at_cev95": cev95 <= 2,
            "fits_within_two_dimensions_at_cev99": cev99 <= 2,
        }
    pooled = np.concatenate(tuple(centered_by_factor.values()), axis=0)
    return {
        "construction": (
            "For each factor, all other complete length-11 subtoken sequences "
            "are fixed within group; only that factor's sequence varies. "
            "Activations are mean-centered within fixed-context group."
        ),
        "factors": factor_metrics,
        "union_geometry": variance_geometry(
            pooled,
            max_spectrum_entries=32,
        ),
        "dimension_additivity_cev95": dimension_additivity(
            centered_by_factor,
            variance_fraction=0.95,
        ),
        "pairwise_principal_angle_overlap": pairwise_subspace_overlaps(bases),
    }


def embedding_additive_decomposition(
    embedding_table: np.ndarray,
    *,
    n_factors: int,
) -> dict[str, Any]:
    """Fit E(z)=mu+sum E_n(z_n) and report the interaction residual."""

    embeddings = np.asarray(embedding_table, dtype=np.float64)
    expected_rows = FACTOR_CARDINALITY**n_factors
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
        raise ValueError("embedding table does not match the joint vocabulary")
    subtokens = np.stack(
        np.unravel_index(
            np.arange(expected_rows),
            (FACTOR_CARDINALITY,) * n_factors,
        ),
        axis=1,
    )
    columns = [np.ones((expected_rows, 1), dtype=np.float64)]
    for factor_index in range(n_factors):
        columns.append(
            np.stack(
                [
                    subtokens[:, factor_index] == level
                    for level in range(1, FACTOR_CARDINALITY)
                ],
                axis=1,
            ).astype(np.float64)
        )
    design = np.concatenate(columns, axis=1)
    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        embeddings,
        rcond=None,
    )
    additive = design @ coefficients
    interaction = embeddings - additive
    centered = embeddings - embeddings.mean(axis=0)
    total = float(np.square(centered).sum())
    interaction_variance = float(np.square(interaction).sum())
    interaction_fraction = (
        0.0 if total == 0.0 else interaction_variance / total
    )
    return {
        "model": "E(z1,...,zN) = mu + sum_n E_n(zn) + E_interaction",
        "joint_vocabulary_size": expected_rows,
        "embedding_width": embeddings.shape[1],
        "additive_variance_fraction": 1.0 - interaction_fraction,
        "interaction_variance_fraction": interaction_fraction,
        "additive_geometry": variance_geometry(
            additive,
            max_spectrum_entries=32,
        ),
        "interaction_geometry": variance_geometry(
            interaction,
            max_spectrum_entries=32,
        ),
    }


def _plot_rank_curves(metrics: dict[str, Any], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for name, payload in metrics.items():
        if payload["status"] != "measured":
            continue
        axis.plot(
            payload["ranks"],
            payload["held_out_r_squared"],
            label=name.replace("_", " "),
        )
    axis.set_xlabel("Activation PCA rank k")
    axis.set_ylabel("Held-out target R²")
    axis.set_title("Reduced-rank predictive curves")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _run_from_data(
    context: RunContext,
    *,
    condition: StudyCondition,
    train: RepresentationData,
    test: RepresentationData,
    activation_extractor: Callable[[torch.Tensor], np.ndarray],
    embedding_table: np.ndarray,
    source: str,
) -> dict[str, Any]:
    reverse = _activation_encoding(train, test)
    ranks = _rank_targets(train, test)
    vary_one = _vary_one_metrics(
        condition=condition,
        device=_device(context),
        seed=int(context.seed) + 7_003,
        smoke=context.smoke,
        activation_extractor=activation_extractor,
    )
    embedding = embedding_additive_decomposition(
        embedding_table,
        n_factors=condition.n_factors,
    )
    _write_json(
        context.results_dir / "reverse_encoding_metrics.json",
        reverse,
    )
    _write_json(
        context.results_dir / "reduced_rank_curves.json",
        ranks,
    )
    _plot_rank_curves(
        ranks,
        context.results_dir / "reduced_rank_curves.png",
    )
    _write_json(context.results_dir / "vary_one_metrics.json", vary_one)
    _write_json(
        context.results_dir / "embedding_additivity_metrics.json",
        embedding,
    )
    summary = {
        "source": source,
        "condition": condition.name,
        "reverse_encoding": {
            "additive_r_squared": reverse[
                "additive_factor_belief_activation_r_squared"
            ],
            "joint_r_squared": reverse[
                "full_joint_belief_activation_r_squared"
            ],
            "joint_interaction_extra_r_squared": reverse[
                "extra_variance_explained_by_joint_interactions"
            ],
        },
        "rank_retaining_99pct": {
            name: payload.get("rank_retaining_99pct_of_best")
            for name, payload in ranks.items()
        },
        "vary_one_union_cev95_dimension": vary_one["union_geometry"][
            "cev95_dimension"
        ],
        "embedding_additive_variance_fraction": embedding[
            "additive_variance_fraction"
        ],
    }
    _write_json(context.results_dir / "analysis_summary.json", summary)
    return summary


def run_analysis_suite(
    context: RunContext,
    *,
    model: SupervisedNextTokenModel,
    condition: StudyCondition,
) -> dict[str, Any]:
    """Run all controls on the trained CE-only checkpoint."""

    if context.seed is None:
        raise ValueError("analysis requires a resolved seed")
    device = _device(context)
    model = model.to(device).eval()
    train_count = SMOKE_TRAIN_SAMPLES if context.smoke else FULL_TRAIN_SAMPLES
    test_count = SMOKE_TEST_SAMPLES if context.smoke else FULL_TEST_SAMPLES
    train_tokens, _, train_factors = make_next_token_batch(
        train_count,
        condition=condition,
        device=device,
        generator=_generator(device, context.seed + 5_001),
    )
    test_tokens, _, test_factors = make_next_token_batch(
        test_count,
        condition=condition,
        device=device,
        generator=_generator(device, context.seed + 5_002),
    )
    train = extract_supervised_data(model, train_tokens, train_factors)
    test = extract_supervised_data(model, test_tokens, test_factors)

    @torch.inference_mode()
    def activation_extractor(tokens: torch.Tensor) -> np.ndarray:
        return _to_numpy(
            model.encode_tokens(tokens.to(device), apply_final_norm=False)[:, -1]
        )

    return _run_from_data(
        context,
        condition=condition,
        train=train,
        test=test,
        activation_extractor=activation_extractor,
        embedding_table=_to_numpy(model.token_embedding_table()),
        source="supervised_next_token_ce_checkpoint",
    )


@torch.inference_mode()
def _extract_ppo_data(
    module: Any,
    tokens: torch.Tensor,
    factor_paths: torch.Tensor,
    condition: StudyCondition,
) -> RepresentationData:
    observations = F.one_hot(
        tokens,
        num_classes=condition.vocabulary_size,
    ).to(dtype=module.encoder.input_embedding.weight.dtype)
    context = torch.zeros(
        len(tokens),
        CONTEXT_LENGTH - 1,
        condition.vocabulary_size,
        dtype=observations.dtype,
        device=tokens.device,
    )
    lengths = torch.zeros(len(tokens), device=tokens.device)
    activations = module.encoder(
        context,
        lengths,
        observations,
        apply_final_norm=False,
    )
    normalized = module.encoder.final_norm(activations)
    logits = module.action_distribution_inputs(normalized)[:, -1]
    values = module.heads.values(normalized)[:, -1]
    factors = factor_beliefs_from_paths(factor_paths)
    return RepresentationData(
        activations=_to_numpy(activations[:, -1]),
        factor_beliefs=factors,
        joint_beliefs=rowwise_tensor_product(factors),
        logits=_to_numpy(logits),
        probabilities=_to_numpy(torch.softmax(logits, dim=-1)),
        token_histories=_history_identity(
            tokens,
            condition.vocabulary_size,
        ),
        critic_values=_to_numpy(values).reshape(len(tokens), -1),
    )


def analyze_ppo_checkpoint(
    context: RunContext,
    *,
    n_factors: int,
    d_model: int,
) -> dict[str, Any]:
    """Analyze one matched PPO Algorithm checkpoint.

    Contract: ``context.resume_from`` is an RLlib Algorithm checkpoint from the
    matched ``factored_mess3_beliefs_2026_08`` condition. The caller selects the
    corresponding factor count and width in ordinary Python; there are no
    scientific CLI flags. Compact outputs are written to ``context.results_dir``.
    """

    if context.resume_from is None:
        raise ValueError(
            "PPO checkpoint analysis requires RunContext.resume_from"
        )
    if context.seed is None:
        raise ValueError("analysis requires a resolved seed")
    condition = StudyCondition(n_factors=n_factors, d_model=d_model)
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    device = _device(context)
    train_count = SMOKE_TRAIN_SAMPLES if context.smoke else FULL_TRAIN_SAMPLES
    test_count = SMOKE_TEST_SAMPLES if context.smoke else FULL_TEST_SAMPLES
    train_tokens, _, train_factors = make_next_token_batch(
        train_count,
        condition=condition,
        device=device,
        generator=_generator(device, context.seed + 5_001),
    )
    test_tokens, _, test_factors = make_next_token_batch(
        test_count,
        condition=condition,
        device=device,
        generator=_generator(device, context.seed + 5_002),
    )
    with load_algorithm(context.resume_from) as algorithm:
        module = algorithm.get_module()
        if module is None:
            raise KeyError("checkpoint has no default RLModule")
        module = module.to(device).eval()
        train = _extract_ppo_data(
            module,
            train_tokens,
            train_factors,
            condition,
        )
        test = _extract_ppo_data(
            module,
            test_tokens,
            test_factors,
            condition,
        )

        def activation_extractor(tokens: torch.Tensor) -> np.ndarray:
            observations = F.one_hot(
                tokens.to(device),
                num_classes=condition.vocabulary_size,
            ).to(dtype=module.encoder.input_embedding.weight.dtype)
            context_tensor = torch.zeros(
                len(tokens),
                CONTEXT_LENGTH - 1,
                condition.vocabulary_size,
                dtype=observations.dtype,
                device=device,
            )
            lengths = torch.zeros(len(tokens), device=device)
            return _to_numpy(
                module.encoder(
                    context_tensor,
                    lengths,
                    observations,
                    apply_final_norm=False,
                )[:, -1]
            )

        return _run_from_data(
            context,
            condition=condition,
            train=train,
            test=test,
            activation_extractor=activation_extractor,
            embedding_table=_to_numpy(
                module.encoder.input_embedding.weight.transpose(0, 1)
            ),
            source="matched_ppo_algorithm_checkpoint",
        )


__all__ = [
    "RepresentationData",
    "analyze_ppo_checkpoint",
    "construct_vary_one_factor_paths",
    "embedding_additive_decomposition",
    "extract_supervised_data",
    "factor_beliefs_from_paths",
    "reduced_rank_curve",
    "run_analysis_suite",
]
