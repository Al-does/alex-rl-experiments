"""Scientific and smoke coverage for the factored representation controls."""

from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
import torch

from analysis.probes.factorization import rowwise_tensor_product
from experiments.factored_mess3_representation_controls_2026_08.analysis import (
    RepresentationData,
    _activation_encoding,
    construct_vary_one_factor_paths,
    embedding_additive_decomposition,
    reduced_rank_curve,
)
from experiments.factored_mess3_representation_controls_2026_08.shared import (
    CONTEXT_LENGTH,
    FULL_NEXT_TOKEN_EXAMPLES,
    SMOKE_NEXT_TOKEN_EXAMPLES,
    StudyCondition,
    SupervisedNextTokenModel,
    build_model_config,
    combine_factor_tokens,
    make_next_token_batch,
    run_condition,
    sample_factor_paths,
)
from harness.context import RunContext
from harness.hardware import PROFILES


CONDITIONS = (
    ("two_factor_64d", 2, 64),
    ("three_factor_64d", 3, 64),
    ("five_factor_64d", 5, 64),
    ("five_factor_120d", 5, 120),
)


def _context(tmp_path) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )


def test_all_four_leaves_import_and_build_fresh_matched_models():
    assert FULL_NEXT_TOKEN_EXAMPLES == 50_000_000
    assert SMOKE_NEXT_TOKEN_EXAMPLES == 4_096
    assert CONTEXT_LENGTH == 11

    for leaf, n_factors, width in CONDITIONS:
        module = importlib.import_module(
            "experiments.factored_mess3_representation_controls_2026_08."
            f"{leaf}.experiment"
        )
        assert callable(module.run)
        condition = StudyCondition(n_factors, width)
        first_config = build_model_config(condition)
        second_config = build_model_config(condition)
        first = SupervisedNextTokenModel(condition)
        second = SupervisedNextTokenModel(condition)

        assert first_config is not second_config
        assert first is not second
        assert first_config["d_model"] == width
        assert first_config["context_length"] == 11
        assert first.condition.vocabulary_size == 3**n_factors
        assert hasattr(first, "next_token_head")
        assert not hasattr(first, "critic")
        assert not hasattr(first, "value_head")


def test_next_token_targets_are_exactly_one_step_shifted():
    condition = StudyCondition(3, 64)
    device = torch.device("cpu")
    direct_generator = torch.Generator().manual_seed(91)
    expected_factors = sample_factor_paths(
        7,
        n_factors=3,
        length=CONTEXT_LENGTH + 1,
        device=device,
        generator=direct_generator,
    )
    expected_joint = combine_factor_tokens(expected_factors)

    inputs, targets, factors = make_next_token_batch(
        7,
        condition=condition,
        device=device,
        generator=torch.Generator().manual_seed(91),
    )

    torch.testing.assert_close(inputs, expected_joint[:, :-1])
    torch.testing.assert_close(targets, expected_joint[:, 1:])
    torch.testing.assert_close(factors, expected_factors[:, :-1])
    assert targets.numel() == 7 * CONTEXT_LENGTH


def _synthetic_data(
    rng: np.random.Generator,
    count: int,
) -> RepresentationData:
    factors = tuple(
        rng.dirichlet(np.ones(3), size=count)
        for _ in range(2)
    )
    joint = rowwise_tensor_product(factors)
    activations = np.concatenate(
        [
            factors[0][:, :2],
            factors[1][:, :2],
            (joint[:, 0] * 4.0)[:, None],
        ],
        axis=1,
    )
    return RepresentationData(
        activations=activations,
        factor_beliefs=factors,
        joint_beliefs=joint,
        logits=np.zeros((count, 9)),
        probabilities=np.full((count, 9), 1 / 9),
        token_histories=np.zeros((count, 99)),
        critic_values=None,
    )


def test_reverse_encoding_separates_additive_and_joint_interactions():
    rng = np.random.default_rng(4)
    metrics = _activation_encoding(
        _synthetic_data(rng, 2_000),
        _synthetic_data(rng, 1_000),
    )

    assert metrics["full_joint_belief_activation_r_squared"] > 0.999
    assert (
        metrics["extra_variance_explained_by_joint_interactions"]
        > 0.01
    )
    assert (
        metrics["factor_residual_activation_geometry"]["cev95_dimension"]
        == 1
    )


def test_reduced_rank_curve_finds_two_predictive_dimensions():
    rng = np.random.default_rng(8)
    train = rng.normal(size=(1_000, 6)) * np.array([10, 5, 0.2, 0.1, 0.05, 0.01])
    test = rng.normal(size=(500, 6)) * np.array([10, 5, 0.2, 0.1, 0.05, 0.01])
    train_target = train[:, :2] @ np.array([[1.0, -0.5], [0.3, 0.8]])
    test_target = test[:, :2] @ np.array([[1.0, -0.5], [0.3, 0.8]])

    curve = reduced_rank_curve(train, test, train_target, test_target)

    assert curve["best_achievable_r_squared"] > 0.999
    assert curve["rank_retaining_99pct_of_best"] == 2
    assert len(curve["ranks"]) == 6


def test_vary_one_construction_fixes_every_other_factor():
    fixed = torch.tensor(
        [
            [[0, 1, 2], [1, 2, 0]],
            [[2, 0, 1], [0, 1, 2]],
        ]
    )
    varying = torch.tensor(
        [
            [[0, 0], [1, 1], [2, 2]],
            [[2, 1], [1, 0], [0, 2]],
        ]
    )

    controlled, groups = construct_vary_one_factor_paths(
        fixed,
        varying,
        factor_index=1,
    )
    reshaped = controlled.reshape(2, 3, 2, 3)

    torch.testing.assert_close(
        reshaped[:, :, :, 0],
        fixed[:, None, :, 0].expand(2, 3, 2),
    )
    torch.testing.assert_close(
        reshaped[:, :, :, 2],
        fixed[:, None, :, 2].expand(2, 3, 2),
    )
    torch.testing.assert_close(reshaped[:, :, :, 1], varying)
    np.testing.assert_array_equal(groups, [0, 0, 0, 1, 1, 1])


def test_embedding_additivity_detects_an_interaction_residual():
    subtokens = np.stack(
        np.unravel_index(np.arange(9), (3, 3)),
        axis=1,
    )
    first = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    second = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    additive = first[subtokens[:, 0]] + second[subtokens[:, 1]]
    exact = embedding_additive_decomposition(additive, n_factors=2)
    interacted = additive.copy()
    interacted[:, 0] += (
        (subtokens[:, 0] == 1) & (subtokens[:, 1] == 2)
    ) * 3.0
    with_interaction = embedding_additive_decomposition(
        interacted,
        n_factors=2,
    )

    assert exact["interaction_variance_fraction"] < 1e-20
    assert with_interaction["interaction_variance_fraction"] > 0.01
    assert with_interaction["additive_variance_fraction"] < 1.0


@pytest.mark.slow
def test_real_smoke_trains_and_runs_every_analysis(tmp_path):
    context = _context(tmp_path)
    result = run_condition(context, n_factors=2, d_model=64)

    assert result["training"]["completed_next_token_examples"] == 4_096
    assert result["training"]["has_ppo_objective"] is False
    assert result["training"]["has_value_or_critic_head"] is False
    assert result["output_validation"]["status"] == "completed"
    validation = json.loads(
        (context.results_dir / "output_validation.json").read_text()
    )
    for name in validation["required_files"]:
        assert (context.results_dir / name).is_file()
    rank_metrics = json.loads(
        (context.results_dir / "reduced_rank_curves.json").read_text()
    )
    assert rank_metrics["critic_value"]["status"] == "absent"
    assert rank_metrics["factor_beliefs"]["status"] == "measured"
    vary_one = json.loads(
        (context.results_dir / "vary_one_metrics.json").read_text()
    )
    assert set(vary_one["factors"]) == {"factor_0", "factor_1"}
    reverse = json.loads(
        (context.results_dir / "reverse_encoding_metrics.json").read_text()
    )
    assert "extra_variance_explained_by_joint_interactions" in reverse
