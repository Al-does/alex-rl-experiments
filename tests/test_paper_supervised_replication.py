"""Scientific and wiring tests for the paper-faithful MESS3 replication."""

from __future__ import annotations

import math

import pytest
import torch

from harness.artifacts import RunArtifacts
from experiments.mess3_supervised.large_batch_replication.experiment import (
    FULL_TRAINING_CONFIG as LARGE_BATCH_CONFIG,
)
from experiments.mess3_supervised.muon_large_batch_replication.experiment import (
    FULL_TRAINING_CONFIG as MUON_CONFIG,
)

from experiments.mess3_supervised.paper_supervised_replication.analysis import (
    grouped_probe_split,
    run_layer_probes,
)
from experiments.mess3_supervised.paper_supervised_replication.mess3 import (
    AliasTable,
    bayesian_beliefs,
    enumerate_paths,
    labeled_operators,
    path_probabilities,
)
from experiments.mess3_supervised.paper_supervised_replication.model import (
    PaperTransformer,
    parameter_count,
)
from experiments.mess3_supervised.paper_supervised_replication.training import (
    TrainingConfig,
    _build_optimizers,
    _optimizer_state_to_cpu,
    exact_validation_loss,
    train,
)


def test_large_batch_recipe_preserves_cumulative_learning_rate_exposure():
    paper = TrainingConfig()

    assert LARGE_BATCH_CONFIG.batch_size == 16_384
    assert LARGE_BATCH_CONFIG.learning_rate == pytest.approx(0.16)
    assert LARGE_BATCH_CONFIG.total_steps == 62_500
    assert (
        LARGE_BATCH_CONFIG.total_steps * LARGE_BATCH_CONFIG.learning_rate
        == pytest.approx(paper.total_steps * paper.learning_rate)
    )


def test_muon_recipe_uses_muon_and_adamw_parameter_groups():
    model = PaperTransformer()
    optimizers = _build_optimizers(model, MUON_CONFIG)

    assert MUON_CONFIG.batch_size == LARGE_BATCH_CONFIG.batch_size
    assert MUON_CONFIG.total_steps == LARGE_BATCH_CONFIG.total_steps
    assert MUON_CONFIG.learning_rate == pytest.approx(0.02)
    assert MUON_CONFIG.auxiliary_learning_rate == pytest.approx(3e-4)
    assert isinstance(optimizers[0], torch.optim.Muon)
    assert isinstance(optimizers[1], torch.optim.AdamW)


def test_checkpoint_serialization_does_not_mutate_live_optimizer_state():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = torch.optim.AdamW([parameter])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    live_average = optimizer.state[parameter]["exp_avg"]

    serialized = _optimizer_state_to_cpu(optimizer)

    assert optimizer.state[parameter]["exp_avg"] is live_average
    assert serialized["state"][0]["exp_avg"] is not live_average


def test_paper_matrices_and_path_distribution_are_exact():
    operators = labeled_operators()
    torch.testing.assert_close(
        operators[0],
        torch.tensor(
            [
                [0.765, 0.00375, 0.00375],
                [0.0425, 0.0675, 0.00375],
                [0.0425, 0.00375, 0.0675],
            ],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        operators.sum(dim=0),
        torch.tensor(
            [
                [0.90, 0.05, 0.05],
                [0.05, 0.90, 0.05],
                [0.05, 0.05, 0.90],
            ],
            dtype=torch.float64,
        ),
    )
    for length in (1, 4, 11):
        paths = enumerate_paths(length)
        probabilities = path_probabilities(paths)
        torch.testing.assert_close(
            probabilities.sum(),
            torch.tensor(1.0, dtype=torch.float64),
        )
        assert torch.all(probabilities > 0)


def test_bayesian_update_and_alias_table_match_path_probabilities():
    paths = enumerate_paths(4)
    probabilities = path_probabilities(paths)
    beliefs = bayesian_beliefs(paths)
    torch.testing.assert_close(
        beliefs.sum(dim=-1),
        torch.ones_like(beliefs[..., 0]),
    )

    alias = AliasTable.from_probabilities(probabilities, device="cpu")
    count = len(probabilities)
    reconstructed = torch.zeros(count, dtype=torch.float64)
    reconstructed += alias.threshold / count
    reconstructed.scatter_add_(
        0,
        alias.alias,
        (1.0 - alias.threshold) / count,
    )
    torch.testing.assert_close(reconstructed, probabilities, atol=1e-14, rtol=1e-12)


def test_model_matches_paper_scale_and_is_causal():
    torch.manual_seed(0)
    model = PaperTransformer().eval()
    assert parameter_count(model) == 143_075
    first = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 2, 0]])
    second = first.clone()
    second[:, 6:] = torch.tensor([[2, 2, 1, 1]])
    logits_first = model(first)
    logits_second = model(second)
    torch.testing.assert_close(
        logits_first[:, :6],
        logits_second[:, :6],
        atol=1e-7,
        rtol=1e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_compile_preserves_state_keys_and_eager_loss():
    torch.manual_seed(3)
    model = PaperTransformer().cuda()
    tokens = enumerate_paths(10, device="cuda")[:64]
    targets = tokens.roll(-1, dims=1)
    eager_logits = model(tokens)
    eager_loss = torch.nn.functional.cross_entropy(
        eager_logits.reshape(-1, 3),
        targets.reshape(-1),
    )
    state_keys = tuple(model.state_dict())

    model.compile(mode="reduce-overhead", fullgraph=True)
    compiled_logits = model(tokens)
    compiled_loss = torch.nn.functional.cross_entropy(
        compiled_logits.reshape(-1, 3),
        targets.reshape(-1),
    )
    torch.testing.assert_close(compiled_loss, eager_loss, atol=1e-5, rtol=1e-5)
    assert tuple(model.state_dict()) == state_keys


def test_exact_validation_weights_every_shifted_target():
    model = PaperTransformer()
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    paths = enumerate_paths(5)
    probabilities = path_probabilities(paths)
    loss = exact_validation_loss(
        model,
        paths,
        probabilities,
        batch_size=32,
    )
    assert loss == pytest.approx(math.log(3.0), abs=1e-6)


def test_probe_split_groups_sequences_and_tiny_probe_runs():
    fit, test = grouped_probe_split(100, seed=7)
    assert len(fit) == 20
    assert len(test) == 80
    assert not set(fit).intersection(test)

    torch.manual_seed(5)
    model = PaperTransformer()
    contexts = enumerate_paths(3)
    result, target, prediction = run_layer_probes(
        model,
        contexts,
        seed=9,
        batch_size=16,
    )
    assert result["n_fit_sequences"] == 5
    assert result["n_test_sequences"] == 22
    assert set(result["layers"]) == {
        "block_0",
        "block_1",
        "block_2",
        "block_3",
        "final_ln",
    }
    assert target.shape == prediction.shape
    assert target.shape[1] == 3


def test_training_checkpoint_resume_matches_uninterrupted_run(tmp_path):
    paths = enumerate_paths(4)
    probabilities = path_probabilities(paths)
    alias = AliasTable.from_probabilities(probabilities, device="cpu")

    def config(total_steps: int) -> TrainingConfig:
        return TrainingConfig(
            total_steps=total_steps,
            analyzed_step=total_steps,
            batch_size=4,
            log_every=1,
            checkpoint_every=1,
            retain_periodic_checkpoints=True,
            validation_every=total_steps,
            validation_batch_size=64,
        )

    def outputs(name: str) -> RunArtifacts:
        result = RunArtifacts(
            tmp_path / name / "results",
            tmp_path / name / "artifacts",
        )
        result.prepare()
        return result

    torch.manual_seed(11)
    partial = PaperTransformer()
    partial_outputs = outputs("partial")
    train(
        model=partial,
        paths=paths,
        probabilities=probabilities,
        alias_table=alias,
        device=torch.device("cpu"),
        seed=13,
        config=config(2),
        outputs=partial_outputs,
    )
    assert (partial_outputs.checkpoints_dir / "step_0000001.pt").exists()

    torch.manual_seed(999)
    resumed = PaperTransformer()
    resumed_outputs = outputs("resumed")
    train(
        model=resumed,
        paths=paths,
        probabilities=probabilities,
        alias_table=alias,
        device=torch.device("cpu"),
        seed=13,
        config=config(4),
        outputs=resumed_outputs,
        resume_from=partial_outputs.checkpoints_dir / "latest.pt",
    )

    torch.manual_seed(11)
    uninterrupted = PaperTransformer()
    train(
        model=uninterrupted,
        paths=paths,
        probabilities=probabilities,
        alias_table=alias,
        device=torch.device("cpu"),
        seed=13,
        config=config(4),
        outputs=outputs("uninterrupted"),
    )
    for resumed_parameter, uninterrupted_parameter in zip(
        resumed.parameters(),
        uninterrupted.parameters(),
    ):
        torch.testing.assert_close(
            resumed_parameter,
            uninterrupted_parameter,
        )
