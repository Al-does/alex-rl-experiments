"""Scientific and wiring tests for the pure next-token factor study."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from harness.artifacts import RunArtifacts

from experiments.factored_representations_next_token_2026_08.model import (
    FactoredNextTokenTransformer,
    NextTokenModelConfig,
)
from experiments.factored_representations_next_token_2026_08.process import (
    FACTOR_COUNTS,
    MESS3_ALPHA,
    MESS3_X,
    SEQUENCE_LENGTH,
    decode_joint_tokens,
    encode_joint_tokens,
    mess3_labeled_operators,
    product_beliefs,
    sample_sequences,
)
from experiments.factored_representations_next_token_2026_08.shared import (
    FULL_TRAINING_CONFIG,
    SMOKE_TRAINING_CONFIG,
)
from experiments.factored_representations_next_token_2026_08.training import (
    TrainingConfig,
    checkpoint_updates,
    language_model_io,
    train,
)


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def test_labeled_operators_match_paper_appendix_c():
    operators = mess3_labeled_operators(device="cpu", dtype=torch.float64)
    beta = (1.0 - MESS3_ALPHA) / 2.0
    y = 1.0 - 2.0 * MESS3_X
    expected_zero = torch.tensor(
        [
            [
                MESS3_ALPHA * y,
                beta * MESS3_X,
                beta * MESS3_X,
            ],
            [
                MESS3_ALPHA * MESS3_X,
                beta * y,
                beta * MESS3_X,
            ],
            [
                MESS3_ALPHA * MESS3_X,
                beta * MESS3_X,
                beta * y,
            ],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(operators[0], expected_zero)
    torch.testing.assert_close(
        operators.sum(dim=0),
        torch.tensor(
            [
                [0.7, 0.15, 0.15],
                [0.15, 0.7, 0.15],
                [0.15, 0.15, 0.7],
            ],
            dtype=torch.float64,
        ),
    )


@pytest.mark.parametrize("factor_count", FACTOR_COUNTS)
def test_device_native_sampler_returns_exact_product_beliefs(factor_count):
    batch = sample_sequences(
        batch_size=32,
        factor_count=factor_count,
        device="cpu",
        generator=_generator(7),
        dtype=torch.float64,
    )
    assert batch.tokens.shape == (32, SEQUENCE_LENGTH)
    assert batch.factor_beliefs.shape == (
        32,
        SEQUENCE_LENGTH + 1,
        factor_count,
        3,
    )
    assert batch.target_probabilities.shape == (32, SEQUENCE_LENGTH)
    assert torch.all((0 <= batch.tokens) & (batch.tokens < 3**factor_count))
    torch.testing.assert_close(
        batch.factor_beliefs.sum(dim=-1),
        torch.ones_like(batch.factor_beliefs[..., 0]),
    )

    operators = mess3_labeled_operators(device="cpu", dtype=torch.float64)
    local_probabilities = torch.einsum(
        "blfs,tsq->blft",
        batch.factor_beliefs[:, :-1],
        operators,
    )
    subtokens = decode_joint_tokens(batch.tokens, factor_count)
    selected = local_probabilities.gather(
        -1,
        subtokens.unsqueeze(-1),
    ).squeeze(-1)
    torch.testing.assert_close(
        selected.prod(dim=-1),
        batch.target_probabilities,
    )

    joint = product_beliefs(batch.factor_beliefs)
    assert joint.shape == (
        32,
        SEQUENCE_LENGTH + 1,
        3**factor_count,
    )
    torch.testing.assert_close(
        joint.sum(dim=-1),
        torch.ones_like(joint[..., 0]),
    )


@pytest.mark.parametrize("factor_count", FACTOR_COUNTS)
def test_joint_token_encoding_round_trips(factor_count):
    tokens = torch.arange(3**factor_count, dtype=torch.long)
    subtokens = decode_joint_tokens(tokens, factor_count)
    torch.testing.assert_close(encode_joint_tokens(subtokens), tokens)


@pytest.mark.parametrize("factor_count", FACTOR_COUNTS)
def test_transformer_is_64d_causal_and_uses_joint_bos_vocabulary(factor_count):
    torch.manual_seed(3)
    config = NextTokenModelConfig(factor_count=factor_count)
    model = FactoredNextTokenTransformer(config).eval()
    assert config.d_model == 64
    assert config.n_layers == 4
    assert config.n_heads == 4
    assert config.d_mlp == 256
    assert config.bos_token == 3**factor_count
    assert config.vocab_size == 3**factor_count + 1
    assert model.token_embedding_matrix().shape == (3**factor_count, 64)

    first = torch.randint(0, config.base_vocab_size, (2, SEQUENCE_LENGTH))
    first[:, 0] = config.bos_token
    second = first.clone()
    second[:, 5:] = torch.randint(
        0,
        config.base_vocab_size,
        second[:, 5:].shape,
    )
    first_logits, first_residuals = model(first, return_activations=True)
    second_logits = model(second)
    assert first_logits.shape == (2, SEQUENCE_LENGTH, config.vocab_size)
    assert first_residuals.shape == (2, SEQUENCE_LENGTH, 64)
    torch.testing.assert_close(
        first_logits[:, :5],
        second_logits[:, :5],
        atol=1e-6,
        rtol=1e-5,
    )


def test_language_model_shift_predicts_all_eight_tokens_from_bos_prefix():
    model = FactoredNextTokenTransformer(
        NextTokenModelConfig(factor_count=2)
    )
    batch = sample_sequences(
        batch_size=4,
        factor_count=2,
        device="cpu",
        generator=_generator(11),
    )
    inputs, labels = language_model_io(batch, model)
    assert inputs.shape == labels.shape == (4, SEQUENCE_LENGTH)
    assert torch.all(inputs[:, 0] == model.config.bos_token)
    torch.testing.assert_close(inputs[:, 1:], labels[:, :-1])
    torch.testing.assert_close(labels, batch.tokens)


def test_recipe_uses_only_paper_next_token_optimizer_and_budget():
    assert FULL_TRAINING_CONFIG.total_updates == 500_000
    assert FULL_TRAINING_CONFIG.batch_size == 25_000
    assert FULL_TRAINING_CONFIG.learning_rate == pytest.approx(5e-4)
    assert FULL_TRAINING_CONFIG.weight_decay == 0.0
    assert SMOKE_TRAINING_CONFIG.total_updates == 10
    assert SMOKE_TRAINING_CONFIG.batch_size == 24
    assert checkpoint_updates(10) == (0, 1, 2, 4, 8, 10)
    assert checkpoint_updates(1) == (0, 1)


def test_tiny_training_loop_optimizes_cross_entropy_and_writes_checkpoints(
    tmp_path,
):
    torch.manual_seed(17)
    model = FactoredNextTokenTransformer(
        NextTokenModelConfig(factor_count=2)
    )
    initial = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    outputs = RunArtifacts(
        tmp_path / "results",
        tmp_path / "artifacts",
    )
    outputs.prepare()
    config = TrainingConfig(
        total_updates=2,
        batch_size=4,
        validation_batch_size=8,
        log_every=1,
    )
    history, summary, checkpoints = train(
        model=model,
        factor_count=2,
        device=torch.device("cpu"),
        seed=19,
        validation_seed=23,
        config=config,
        outputs=outputs,
    )

    assert summary["completed_update"] == 2
    assert summary["target_tokens_seen"] == 2 * 4 * SEQUENCE_LENGTH
    assert math.isfinite(summary["final_validation_loss_nats"])
    assert {path.name for path in checkpoints} == {
        "update_000000.pt",
        "update_000001.pt",
        "update_000002.pt",
    }
    assert (outputs.checkpoints_dir / "latest.pt").is_file()
    assert (outputs.results_dir / "training_curve.jsonl").is_file()
    assert [record["update"] for record in history] == [0, 1, 2]
    assert any(
        not torch.equal(initial[name], parameter)
        for name, parameter in model.named_parameters()
    )


def test_new_experiment_package_has_no_rl_framework_dependency():
    package = (
        __import__(
            "experiments.factored_representations_next_token_2026_08",
            fromlist=["__path__"],
        )
    )
    root = next(iter(package.__path__))
    prohibited = ("import ray", "from ray", "PPOConfig", "HMMEnv")
    for path in sorted(__import__("pathlib").Path(root).rglob("*.py")):
        source = path.read_text()
        assert not any(term in source for term in prohibited), path
