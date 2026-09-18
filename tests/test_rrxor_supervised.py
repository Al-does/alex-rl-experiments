"""Scientific and wiring tests for the supervised RRXOR reproduction."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from experiments.mess3_supervised.paper_supervised_replication.training import (
    exact_validation_loss,
)
from experiments.rrxor_supervised.analysis import (
    grouped_context_split,
    run_belief_probes,
)
from experiments.rrxor_supervised.model import (
    MODEL_CONFIG,
    RRXORPaperTransformer,
    parameter_count,
)
from experiments.rrxor_supervised.paper_supervised_replication.experiment import (
    run,
)
from experiments.rrxor_supervised.process import (
    AliasTable,
    bayesian_beliefs,
    enumerate_paths,
    exact_bayesian_loss,
    labeled_operators,
    path_probabilities,
    possible_paths,
    stationary_prior,
)
from experiments.rrxor_supervised.shared import (
    FULL_TRAINING_CONFIG,
    SMOKE_TRAINING_CONFIG,
)
from experiments.rrxor_token_guess_cycle_1.process import (
    RRXOR_STATIONARY,
    RRXOR_TENSOR,
    reachable_beliefs,
)


def test_rrxor_process_adapter_preserves_exact_process_and_stationarity():
    operators = labeled_operators()
    torch.testing.assert_close(
        operators,
        torch.tensor(RRXOR_TENSOR, dtype=torch.float64),
    )
    torch.testing.assert_close(
        stationary_prior(),
        torch.tensor(RRXOR_STATIONARY, dtype=torch.float64),
    )
    transition = operators.sum(dim=0)
    torch.testing.assert_close(
        stationary_prior() @ transition,
        stationary_prior(),
    )
    assert reachable_beliefs().shape == (36, 5)


def test_positive_probability_paths_are_exact_and_normalized():
    all_paths = enumerate_paths(10)
    all_probabilities = path_probabilities(all_paths)
    paths, probabilities = possible_paths(10)

    assert all_paths.shape == (1_024, 10)
    assert paths.shape == (436, 10)
    assert torch.count_nonzero(all_probabilities) == len(paths)
    assert torch.all(probabilities > 0)
    torch.testing.assert_close(
        probabilities.sum(),
        torch.tensor(1.0, dtype=torch.float64),
    )

    alias = AliasTable.from_probabilities(probabilities, device="cpu")
    reconstructed = alias.threshold / len(probabilities)
    reconstructed = reconstructed.clone()
    reconstructed.scatter_add_(
        0,
        alias.alias,
        (1.0 - alias.threshold) / len(probabilities),
    )
    torch.testing.assert_close(
        reconstructed,
        probabilities,
        atol=1e-14,
        rtol=1e-12,
    )


def test_bayesian_targets_are_post_token_and_match_the_paper_loss_floor():
    beliefs = bayesian_beliefs(torch.tensor([[0, 1]], dtype=torch.long))
    torch.testing.assert_close(
        beliefs[0, 0],
        torch.tensor(
            [1 / 3, 1 / 3, 0.0, 1 / 6, 1 / 6],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        beliefs.sum(dim=-1),
        torch.ones_like(beliefs[..., 0]),
    )

    paths, probabilities = possible_paths(10)
    assert exact_bayesian_loss(paths, probabilities) == pytest.approx(
        0.5865014866378117,
        abs=1e-12,
    )


def test_binary_paper_model_matches_scale_and_is_causal():
    torch.manual_seed(3)
    model = RRXORPaperTransformer(MODEL_CONFIG).eval()
    assert MODEL_CONFIG.d_vocab == 2
    assert MODEL_CONFIG.context_length == 10
    assert MODEL_CONFIG.d_model == 64
    assert MODEL_CONFIG.n_layers == 4
    assert parameter_count(model) == 142_946

    first = torch.tensor([[0, 1, 0, 1, 1, 0, 1, 0, 0, 1]])
    second = first.clone()
    second[:, 6:] = torch.tensor([[0, 1, 1, 0]])
    first_logits, activations = model(first, return_activations=True)
    second_logits = model(second)
    assert first_logits.shape == (1, 10, 2)
    assert tuple(activations) == (
        "block_0",
        "block_1",
        "block_2",
        "block_3",
        "pre_ln_final",
        "final_ln",
    )
    torch.testing.assert_close(
        first_logits[:, :6],
        second_logits[:, :6],
        atol=1e-7,
        rtol=1e-6,
    )


def test_supervised_recipe_matches_paper_and_has_bounded_smoke_mode():
    assert FULL_TRAINING_CONFIG.total_steps == 1_000_000
    assert FULL_TRAINING_CONFIG.analyzed_step == 1_000_000
    assert FULL_TRAINING_CONFIG.batch_size == 64
    assert FULL_TRAINING_CONFIG.optimizer_name == "sgd"
    assert FULL_TRAINING_CONFIG.learning_rate == pytest.approx(0.01)
    assert FULL_TRAINING_CONFIG.weight_decay == 0.0
    assert FULL_TRAINING_CONFIG.momentum == 0.0

    assert SMOKE_TRAINING_CONFIG.total_steps == 100
    assert SMOKE_TRAINING_CONFIG.analyzed_step == 100
    assert SMOKE_TRAINING_CONFIG.batch_size == 64
    assert SMOKE_TRAINING_CONFIG.validation_every == 50
    assert callable(run)


def test_exact_validation_of_uniform_binary_model_is_log_two():
    model = RRXORPaperTransformer(MODEL_CONFIG)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    paths, probabilities = possible_paths(10)
    loss = exact_validation_loss(
        model,
        paths,
        probabilities,
        batch_size=512,
    )
    assert loss == pytest.approx(math.log(2.0), abs=1e-6)


def test_grouped_split_and_small_probe_keep_complete_contexts_disjoint():
    fit, test = grouped_context_split(100, seed=7)
    assert len(fit) == 20
    assert len(test) == 80
    assert not set(fit).intersection(test)

    torch.manual_seed(5)
    model = RRXORPaperTransformer(MODEL_CONFIG)
    initial_model = RRXORPaperTransformer(MODEL_CONFIG)
    contexts, _ = possible_paths(4)
    result, battery, target, pairwise = run_belief_probes(
        model,
        initial_model,
        contexts,
        seed=9,
        batch_size=32,
        smoke=True,
    )
    representations = result["geometry"]["representations"]
    assert set(representations) == {
        "layer_1",
        "layer_2",
        "layer_3",
        "layer_4",
        "concatenated_layers",
        "post_final_layer_norm",
    }
    assert result["target"]["timing"].startswith("activation at position t")
    assert target.shape == battery.predictions["concatenated_layers"].shape
    assert target.shape[1] == 5
    pairwise_metrics = result["paper_pairwise_distance_control"]
    assert pairwise_metrics["n_belief_states"] > 1
    assert np.isfinite(
        pairwise_metrics[
            "belief_distance_to_decoded_distance_r_squared"
        ]
    )
    assert pairwise["belief_distances"].shape == (
        pairwise_metrics["n_pairs"],
    )
