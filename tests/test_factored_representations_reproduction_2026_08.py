"""Scientific and wiring tests for the factored-representation PPO reproduction."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns

from analysis.probes import center_within_groups, r2_score
from envs.hmm import HMMEnv, factored_model
from experiments.factored_representations_reproduction_2026_08.analysis import (
    _predict,
    cross_validated_svd_affine,
)
from experiments.factored_representations_reproduction_2026_08.benchmark_batch_size import (
    choose_finalists,
    recommendation,
)
from experiments.factored_representations_reproduction_2026_08.learning import (
    AUXILIARY_COEFFICIENT,
    ActorCriticWithNextJointTokenAux,
    next_joint_token_targets,
)
from experiments.factored_representations_reproduction_2026_08.model import (
    FactoredReproductionActorCritic,
    FactoredReproductionModelConfig,
)
from experiments.factored_representations_reproduction_2026_08.probe import (
    collect_vary_one_data,
)
from experiments.factored_representations_reproduction_2026_08.process import (
    FACTOR_COUNTS,
    MESS3_ALPHA,
    MESS3_X,
    PAPER_TRANSITION_MATRIX,
    decode_joint_tokens,
    encode_joint_tokens,
    environment_config,
    factor_specifications,
    joint_token_count,
    paper_mess3_model,
)
from experiments.factored_representations_reproduction_2026_08.shared import (
    MINIBATCH_SIZE,
    MODEL_CONFIG,
    SMOKE_BATCH_SIZE,
    SMOKE_MINIBATCH_SIZE,
    TRAIN_BATCH_SIZE,
    build_config,
)
from harness.context import RunContext
from harness.hardware import PROFILES


def _context(tmp_path) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )


def test_paper_mess3_parameters_match_appendix_c_labeled_operators():
    model = paper_mess3_model()
    beta = (1.0 - MESS3_ALPHA) / 2.0
    y = 1.0 - 2.0 * MESS3_X
    expected_token_zero = np.array(
        [
            [MESS3_ALPHA * y, beta * MESS3_X, beta * MESS3_X],
            [MESS3_ALPHA * MESS3_X, beta * y, beta * MESS3_X],
            [MESS3_ALPHA * MESS3_X, beta * MESS3_X, beta * y],
        ]
    )

    np.testing.assert_allclose(model.transition_matrix, PAPER_TRANSITION_MATRIX)
    np.testing.assert_allclose(
        model.transition_matrix @ np.diag(model.emission_matrix[:, 0]),
        expected_token_zero,
    )
    np.testing.assert_allclose(model.initial_distribution, np.full(3, 1.0 / 3.0))


@pytest.mark.parametrize("factor_count", FACTOR_COUNTS)
def test_factored_model_is_exact_product_with_one_joint_token(factor_count):
    factors = [paper_mess3_model() for _ in range(factor_count)]
    model = factored_model(factors=factor_specifications(factor_count))

    expected_transition = factors[0].transition_matrix
    expected_emission = factors[0].emission_matrix
    for factor in factors[1:]:
        expected_transition = np.kron(
            expected_transition,
            factor.transition_matrix,
        )
        expected_emission = np.kron(expected_emission, factor.emission_matrix)

    assert model.n_states == 3**factor_count
    assert model.n_tokens == 3**factor_count
    np.testing.assert_allclose(model.transition_matrix, expected_transition)
    np.testing.assert_allclose(model.emission_matrix, expected_emission)

    subtokens = decode_joint_tokens(
        np.arange(joint_token_count(factor_count)),
        factor_count,
    )
    np.testing.assert_array_equal(
        encode_joint_tokens(subtokens),
        np.arange(joint_token_count(factor_count)),
    )


@pytest.mark.parametrize("factor_count", FACTOR_COUNTS)
def test_delayed_environment_scores_hidden_joint_token(factor_count):
    config = environment_config(factor_count)
    config["diagnostics"] = {"tokens": True}
    environment = HMMEnv(config)
    try:
        observation, info = environment.reset(seed=7)
        assert observation.shape == (joint_token_count(factor_count),)
        assert observation.sum() == 0.0
        hidden_joint_token = info["raw_token_current"]
        next_observation, reward, _, _, next_info = environment.step(
            hidden_joint_token
        )
        assert reward == 1.0
        assert next_info["visible_source_token"] == hidden_joint_token
        assert next_observation.argmax() == hidden_joint_token
    finally:
        environment.close()


def test_model_is_64d_pre_ln_causal_and_has_learned_bos():
    torch.manual_seed(3)
    config = FactoredReproductionModelConfig()
    module = FactoredReproductionActorCritic(
        observation_space=gym.spaces.Box(0.0, 1.0, shape=(9,), dtype=np.float32),
        action_space=gym.spaces.Discrete(9),
        model_config=config.to_dict(),
    ).eval()
    assert config.d_model == 64
    assert config.n_layers == 4
    assert config.n_heads == 4
    assert config.d_mlp == 256
    assert module.encoder.token_embedding_matrix().shape == (9, 64)

    first = torch.zeros((1, 6, 9))
    first[0, 1:, 0] = 1.0
    second = first.clone()
    second[0, 4:, 0] = 0.0
    second[0, 4:, 1] = 1.0
    state = module.get_initial_state()
    context = torch.from_numpy(state["ctx"]).unsqueeze(0)
    lengths = torch.from_numpy(state["len"])
    first_residual = module.encode_chunks_pre_final_norm(
        context,
        lengths,
        first,
    )
    second_residual = module.encode_chunks_pre_final_norm(
        context,
        lengths,
        second,
    )
    torch.testing.assert_close(first_residual[:, :4], second_residual[:, :4])
    assert not torch.allclose(first_residual[:, 4:], second_residual[:, 4:])


@pytest.mark.parametrize("factor_count", FACTOR_COUNTS)
@pytest.mark.parametrize("condition", ["ppo", "ppo_aux_ce"])
def test_smoke_configs_are_fresh_and_resolve_each_design_cell(
    tmp_path,
    factor_count,
    condition,
):
    context = _context(tmp_path)
    first = build_config(
        context,
        factor_count=factor_count,
        condition=condition,
    )
    second = build_config(
        context,
        factor_count=factor_count,
        condition=condition,
    )

    assert first is not second
    assert first.seed == 42
    assert first.gamma == 0.0
    assert first.lambda_ == 0.0
    assert first.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert first.minibatch_size == SMOKE_MINIBATCH_SIZE
    assert first.num_env_runners == 0
    assert first.rl_module_spec.model_config["d_model"] == 64
    environment = first.env(first.env_config)
    try:
        assert environment.action_space.n == 3**factor_count
        assert environment.config.delay == 1
        assert environment.config.episode_length == 9
    finally:
        environment.close()

    if condition == "ppo_aux_ce":
        assert first.rl_module_spec.module_class is ActorCriticWithNextJointTokenAux
        assert (
            first.learner_config_dict["next_token_aux/lambda"]
            == AUXILIARY_COEFFICIENT
        )
        assert (
            first.rl_module_spec.model_config["next_token_aux"]["num_classes"]
            == 3**factor_count
        )
    else:
        assert first.rl_module_spec.module_class is FactoredReproductionActorCritic


def test_auxiliary_targets_predict_the_token_revealed_next():
    observations = torch.zeros((1, 4, 9))
    observations[0, 1, 3] = 1.0
    observations[0, 2, 5] = 1.0
    observations[0, 3, 7] = 1.0
    logits = torch.randn(1, 4, 9)
    aligned, targets, valid = next_joint_token_targets(
        {Columns.OBS: observations},
        logits,
    )

    assert aligned.shape == (1, 3, 9)
    torch.testing.assert_close(targets, torch.tensor([[3, 5, 7]]))
    assert valid.all()


def test_cross_validated_joint_regression_recovers_factor_targets():
    rng = np.random.default_rng(11)
    features = rng.normal(size=(500, 8))
    expected_weight = rng.normal(size=(8, 6))
    expected_bias = rng.normal(size=6)
    targets = features @ expected_weight + expected_bias
    weight, bias, report = cross_validated_svd_affine(
        features[:400],
        targets[:400],
        seed=9,
    )
    predicted = _predict(weight, bias, features[400:])

    assert report["selected_rcond"] in report["rcond_candidates"]
    assert r2_score(predicted, targets[400:]) > 0.999999


def test_tiny_vary_one_collection_centers_each_position_and_context():
    torch.manual_seed(4)
    module = FactoredReproductionActorCritic(
        observation_space=gym.spaces.Box(0.0, 1.0, shape=(9,), dtype=np.float32),
        action_space=gym.spaces.Discrete(9),
        model_config=MODEL_CONFIG,
    )
    varied = collect_vary_one_data(
        module,
        factor_count=2,
        frozen_contexts=2,
        realizations_per_context=3,
        sequence_length=3,
        seed=5,
    )

    assert set(varied.activations) == {"factor_0", "factor_1"}
    for name, activations in varied.activations.items():
        assert activations.shape == (18, 64)
        centered = center_within_groups(activations, varied.groups[name])
        for group in np.unique(varied.groups[name]):
            np.testing.assert_allclose(
                centered[varied.groups[name] == group].mean(axis=0),
                0.0,
                atol=1e-10,
            )


def test_batch_benchmark_selects_fast_safe_compiled_candidate():
    assert TRAIN_BATCH_SIZE == 32_768
    assert MINIBATCH_SIZE == 32_768

    eager = [
        {
            "batch_size": 4096,
            "status": "completed",
            "steady_state_steps_per_second": 1000.0,
            "max_peak_reserved_fraction": 0.2,
        },
        {
            "batch_size": 8192,
            "status": "completed",
            "steady_state_steps_per_second": 1500.0,
            "max_peak_reserved_fraction": 0.4,
        },
        {
            "batch_size": 16384,
            "status": "oom",
        },
    ]
    assert choose_finalists(eager) == [8192, 4096]

    compiled = [
        {
            "batch_size": 4096,
            "status": "completed",
            "steady_state_steps_per_second": 1200.0,
            "max_peak_cuda_reserved_bytes": 4_000,
            "max_peak_reserved_fraction": 0.2,
            "build_seconds": 10.0,
            "iterations": [
                {"sampled_env_steps": 4096, "duration_seconds": 20.0}
            ],
        },
        {
            "batch_size": 8192,
            "status": "completed",
            "steady_state_steps_per_second": 1800.0,
            "max_peak_cuda_reserved_bytes": 8_000,
            "max_peak_reserved_fraction": 0.4,
            "build_seconds": 12.0,
            "iterations": [
                {"sampled_env_steps": 8192, "duration_seconds": 25.0}
            ],
        },
    ]
    selected = recommendation(compiled)
    assert selected is not None
    assert selected["batch_size"] == 8192
    assert selected["steady_state_steps_per_second"] == 1800.0
    assert selected["estimated_10m_env_steps_hours"] > 0.0
