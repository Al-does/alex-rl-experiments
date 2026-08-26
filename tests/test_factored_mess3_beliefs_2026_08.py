"""Wiring tests for the independent two-factor MESS3 validation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from envs.hmm import HMMEnv, factor_marginals, product_distribution
from experiments.factored_mess3_beliefs_2026_08.analysis import (
    _advance_independent_factor_beliefs,
    _episode_clusters,
)
from experiments.factored_mess3_beliefs_2026_08.shared import (
    ENV_CONFIG,
    LARGE_JOINT_MINIBATCH_SIZE,
    LARGE_JOINT_TRAIN_BATCH_SIZE,
    MODEL_CONFIG,
    SMOKE_BATCH_SIZE,
    SMOKE_ENV_STEPS,
    TOTAL_ENV_STEPS,
    build_config,
    environment_config,
)
from experiments.mess3_token_guess_cycle_2.model import PaperActorCriticModel
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


def test_two_mess3_factory_delays_one_nine_way_token_and_scores_prediction():
    environment = HMMEnv(
        {
            **ENV_CONFIG,
            "diagnostics": {"belief": True, "tokens": True},
            "episode_length": 8,
        }
    )
    try:
        observation, info = environment.reset(seed=7)
        assert environment.model.n_states == 9
        assert environment.model.n_tokens == 9
        assert environment.observation_space.shape == (9,)
        assert environment.action_space.n == 9
        assert observation.sum() == 0.0
        assert info["visible_token_current"] is None

        marginals = factor_marginals(info["belief_current"], (3, 3))
        assert len(marginals) == 2
        np.testing.assert_allclose(marginals[0].sum(), 1.0)
        np.testing.assert_allclose(marginals[1].sum(), 1.0)
        np.testing.assert_allclose(
            product_distribution(marginals),
            info["belief_current"],
            atol=1e-12,
        )

        scalable = np.zeros((1, 2, 3), dtype=np.float64)
        filtered = _advance_independent_factor_beliefs(
            scalable,
            np.array([-1]),
            np.array([0]),
        )
        np.testing.assert_allclose(filtered[0], np.stack(marginals), atol=1e-12)

        emitted = info["raw_token_current"]
        next_observation, reward, _, _, next_info = environment.step(emitted)
        assert reward == 1.0
        assert next_info["visible_token_current"] == emitted
        assert next_observation[emitted] == 1.0
        next_marginals = factor_marginals(
            next_info["belief_current"],
            (3, 3),
        )
        filtered = _advance_independent_factor_beliefs(
            scalable,
            np.array([next_info["visible_token_current"]]),
            np.array([1]),
        )
        np.testing.assert_allclose(
            filtered[0],
            np.stack(next_marginals),
            atol=1e-12,
        )
    finally:
        environment.close()


def test_recipe_builds_fresh_paper_transformer_gamma_zero_ppo(tmp_path):
    context = _context(tmp_path)
    first = build_config(context)
    second = build_config(context)

    assert first is not second
    assert TOTAL_ENV_STEPS == 2_500_000
    assert SMOKE_ENV_STEPS == 4_096
    assert first.gamma == 0.0
    assert first.lambda_ == 0.0
    assert first.train_batch_size_per_learner == SMOKE_BATCH_SIZE
    assert first.minibatch_size == 256
    assert first.num_env_runners == 0
    assert first.num_gpus_per_learner == 0
    assert first.rl_module_spec.module_class is PaperActorCriticModel
    assert MODEL_CONFIG == {
        "d_model": 120,
        "n_layers": 4,
        "n_heads": 3,
        "d_head": 40,
        "d_mlp": 480,
        "context_length": 11,
        "max_seq_len": 11,
        "activation": "relu",
        "normalization": "layer_norm",
        "positional_embedding": "learned_absolute",
    }
    assert first.env_config["delay"] == 1
    assert first.lr == 1e-4
    assert first.num_epochs == 6


def test_three_mess3_recipe_has_27_states_tokens_and_actions(tmp_path):
    config = build_config(_context(tmp_path), n_factors=3)
    environment = HMMEnv(
        {
            **environment_config(3),
            "diagnostics": {"belief": True, "tokens": True},
            "episode_length": 8,
        }
    )
    try:
        observation, info = environment.reset(seed=11)
        assert environment.model.n_states == 27
        assert environment.model.n_tokens == 27
        assert environment.action_space.n == 27
        assert observation.shape == (27,)
        assert observation.sum() == 0.0
        marginals = factor_marginals(info["belief_current"], (3, 3, 3))
        assert len(marginals) == 3
        np.testing.assert_allclose(
            product_distribution(marginals),
            info["belief_current"],
            atol=1e-12,
        )
    finally:
        environment.close()

    assert len(config.env_config["model"]["kwargs"]["factors"]) == 3
    assert config.rl_module_spec.module_class is PaperActorCriticModel


def test_probe_bootstrap_clusters_detect_episode_step_resets_after_warmup():
    data = SimpleNamespace(
        env_indices=np.array([0, 1, 0, 1, 0, 1]),
        episode_steps=np.array([64, 64, 65, 65, 64, 64]),
    )

    clusters = _episode_clusters(data)

    assert len(np.unique(clusters)) == 4
    assert clusters[0] == clusters[2]
    assert clusters[1] == clusters[3]
    assert clusters[4] != clusters[2]
    assert clusters[5] != clusters[3]


def test_five_mess3_recipe_builds_243_way_delayed_token_task(tmp_path):
    config = build_config(_context(tmp_path), n_factors=5)
    environment = HMMEnv(
        {
            **environment_config(5),
            "diagnostics": {"tokens": True},
            "episode_length": 2,
        }
    )
    try:
        observation, info = environment.reset(seed=12)
        assert environment.model.n_states == 243
        assert environment.model.n_tokens == 243
        assert environment.action_space.n == 243
        assert observation.shape == (243,)
        assert observation.sum() == 0.0
        emitted = info["raw_token_current"]
        _, reward, _, _, step_info = environment.step(emitted)
        assert reward == 1.0
        assert step_info["visible_token_current"] == emitted
    finally:
        environment.close()

    assert len(config.env_config["model"]["kwargs"]["factors"]) == 5
    assert config.rl_module_spec.module_class is PaperActorCriticModel


def test_six_mess3_recipe_builds_729_way_environment(tmp_path):
    config = build_config(_context(tmp_path), n_factors=6)
    environment = HMMEnv(
        {
            **environment_config(6),
            "diagnostics": {"tokens": True},
            "episode_length": 2,
        }
    )
    try:
        observation, info = environment.reset(seed=13)
        assert environment.model.n_states == 729
        assert environment.model.n_tokens == 729
        assert environment.action_space.n == 729
        assert observation.shape == (729,)
        assert observation.sum() == 0.0
        assert info["visible_token_current"] is None
    finally:
        environment.close()

    assert len(config.env_config["model"]["kwargs"]["factors"]) == 6

    full_context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "full_results",
        artifacts_dir=tmp_path / "full_artifacts",
        seed=42,
        smoke=False,
        hardware=PROFILES["cpu"],
    )
    full = build_config(full_context, n_factors=6)
    assert full.train_batch_size_per_learner == LARGE_JOINT_TRAIN_BATCH_SIZE
    assert full.minibatch_size == LARGE_JOINT_MINIBATCH_SIZE
    assert full.num_env_runners == 1
    assert full.num_envs_per_env_runner == 4
