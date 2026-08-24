"""Wiring tests for the independent two-factor MESS3 validation."""

from __future__ import annotations

import numpy as np

from envs.hmm import HMMEnv, factor_marginals, product_distribution
from experiments.factored_mess3_beliefs_2026_08.shared import (
    ENV_CONFIG,
    MODEL_CONFIG,
    SMOKE_BATCH_SIZE,
    SMOKE_ENV_STEPS,
    TOTAL_ENV_STEPS,
    build_config,
)
from harness.context import RunContext
from harness.hardware import PROFILES
from learners.models import TransformerModel


def _context(tmp_path) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        seed=42,
        smoke=True,
        hardware=PROFILES["cpu"],
    )


def test_two_mess3_factory_exposes_one_nine_way_token_and_joint_state():
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
        assert observation.sum() == 1.0
        assert info["visible_token_current"] == int(observation.argmax())

        marginals = factor_marginals(info["belief_current"], (3, 3))
        assert len(marginals) == 2
        np.testing.assert_allclose(marginals[0].sum(), 1.0)
        np.testing.assert_allclose(marginals[1].sum(), 1.0)
        np.testing.assert_allclose(
            product_distribution(marginals),
            info["belief_current"],
            atol=1e-12,
        )
    finally:
        environment.close()


def test_recipe_builds_fresh_64_dimensional_gamma_zero_ppo(tmp_path):
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
    assert first.rl_module_spec.module_class is TransformerModel
    assert MODEL_CONFIG == {
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "context_len": 32,
        "max_seq_len": 32,
    }
