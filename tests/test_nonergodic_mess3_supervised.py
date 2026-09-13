from __future__ import annotations

import numpy as np
import torch

from experiments.nonergodic_mess3_supervised.data import (
    BOS_TOKEN,
    CONTEXT_LENGTH,
    EPISODE_LENGTH,
    STATE_COUNT,
    STATES_PER_COMPONENT,
    TOKEN_COUNT,
    VOCAB_SIZE,
    NonergodicSequenceSampler,
    component_posteriors,
    next_token_distributions,
    weighted_beliefs,
)
from experiments.nonergodic_mess3_supervised.model import (
    ModelConfig,
    NonergodicMess3Transformer,
    parameter_count,
)
from experiments.nonergodic_mess3_supervised.training import (
    CONVERGENCE_CHECKPOINT_STEPS,
    FIGURE_CHECKPOINT_STEPS,
    TrainingConfig,
)
from experiments.nonergodic_mess3_token_guess_cycle_1.process import (
    COMPONENT_PARAMETERS,
    nonergodic_mess3_model,
)


def test_recipe_uses_the_blog_component_parameters():
    assert COMPONENT_PARAMETERS == (
        {"name": "mess3_a", "x": 0.15, "alpha": 0.60},
        {"name": "mess3_b", "x": 0.50, "alpha": 0.66},
    )
    model = nonergodic_mess3_model()
    np.testing.assert_allclose(
        model.initial_distribution.reshape(2, STATES_PER_COMPONENT).sum(axis=1),
        [0.5, 0.5],
    )


def test_sampler_emits_bos_and_never_changes_component():
    batch = NonergodicSequenceSampler(42).sample(1_024)
    assert batch.tokens.shape == (1_024, CONTEXT_LENGTH)
    assert batch.states.shape == (1_024, CONTEXT_LENGTH)
    assert np.all(batch.tokens[:, 0] == BOS_TOKEN)
    assert np.all((0 <= batch.tokens[:, 1:]) & (batch.tokens[:, 1:] < TOKEN_COUNT))
    np.testing.assert_array_equal(
        batch.states // STATES_PER_COMPONENT,
        np.broadcast_to(batch.components[:, None], batch.states.shape),
    )
    assert 450 < int(batch.components.sum()) < 575


def test_weighted_beliefs_are_aligned_to_the_observed_prefix():
    sampled = NonergodicSequenceSampler(7).sample(8)
    beliefs = weighted_beliefs(sampled.tokens)
    model = nonergodic_mess3_model()
    np.testing.assert_allclose(
        beliefs[:, 0],
        np.broadcast_to(model.initial_distribution, (len(sampled.tokens), STATE_COUNT)),
    )
    first_tokens = sampled.tokens[:, 1]
    expected = np.einsum(
        "bi,bij->bj",
        beliefs[:, 0],
        model.edge_transition_matrices[first_tokens],
    )
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(beliefs[:, 1], expected)
    np.testing.assert_allclose(beliefs.sum(axis=-1), 1.0)
    np.testing.assert_allclose(
        component_posteriors(beliefs).sum(axis=-1),
        1.0,
    )


def test_next_token_targets_use_the_source_belief_before_each_emission():
    sampled = NonergodicSequenceSampler(11).sample(16)
    beliefs = weighted_beliefs(sampled.tokens)
    distributions = next_token_distributions(beliefs)
    model = nonergodic_mess3_model()
    np.testing.assert_allclose(
        distributions[:, 0],
        beliefs[:, 0] @ model.emission_matrix,
    )
    target_probabilities = np.take_along_axis(
        distributions[:, :-1],
        sampled.tokens[:, 1:, None],
        axis=-1,
    )
    assert target_probabilities.shape == (16, EPISODE_LENGTH, 1)
    assert np.all(target_probabilities > 0.0)


def test_model_matches_the_blog_architecture_and_exposes_block_residuals():
    config = ModelConfig()
    assert config.to_dict() == {
        "d_vocab": 4,
        "context_length": 128,
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 4,
        "d_mlp": 512,
        "initialization_std": 0.02,
        "normalization_epsilon": 1e-5,
        "rope_base": 10_000.0,
        "head_dimension": 32,
        "activation": "gated_gelu",
        "normalization": "rms_norm",
        "positional_embedding": "rope",
        "attention_implementation": "scaled_dot_product_attention",
    }
    model = NonergodicMess3Transformer(config)
    assert parameter_count(model) == 1_057_408
    tokens = torch.randint(0, TOKEN_COUNT, (2, CONTEXT_LENGTH))
    tokens[:, 0] = BOS_TOKEN
    logits, residuals = model(tokens, return_residuals=True)
    assert logits.shape == (2, CONTEXT_LENGTH, VOCAB_SIZE)
    assert tuple(residuals) == (
        "embedding",
        "block_1",
        "block_2",
        "block_3",
        "block_4",
    )
    assert all(
        values.shape == (2, CONTEXT_LENGTH, config.d_model)
        for values in residuals.values()
    )


def test_model_is_causal():
    torch.manual_seed(3)
    model = NonergodicMess3Transformer().eval()
    tokens = torch.randint(0, TOKEN_COUNT, (1, 12))
    tokens[:, 0] = BOS_TOKEN
    changed = tokens.clone()
    changed[:, 8:] = (changed[:, 8:] + 1) % TOKEN_COUNT
    with torch.no_grad():
        logits = model(tokens)
        changed_logits = model(changed)
    torch.testing.assert_close(logits[:, :8], changed_logits[:, :8])


def test_training_configs_match_the_two_article_checkpoints():
    convergence = TrainingConfig.convergence()
    figure = TrainingConfig.figure_checkpoint()
    for config in (convergence, figure):
        assert config.batch_size == 512
        assert config.learning_rate == 1e-3
        assert (config.beta1, config.beta2) == (0.9, 0.999)
        assert config.weight_decay == 0.0
    assert convergence.total_steps == convergence.analyzed_step == 10_000
    assert convergence.checkpoint_steps == CONVERGENCE_CHECKPOINT_STEPS
    assert figure.total_steps == figure.analyzed_step == 45_000
    assert figure.checkpoint_steps == FIGURE_CHECKPOINT_STEPS
