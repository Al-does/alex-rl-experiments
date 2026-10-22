"""Unit tests for the RockSample meta-RL (LPO drift + ES) study."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from experiments.rocksample_meta_rl_2026_09 import shared
from experiments.rocksample_meta_rl_2026_09.es import ESState, OpenES, centered_ranks
from experiments.rocksample_meta_rl_2026_09.learned_objective import (
    CANDIDATE_KEY,
    DriftSpec,
    LearnedDrift,
)
from experiments.rocksample_meta_rl_2026_09.lpo_es_drift import experiment
from harness.context import RunContext
from harness.hardware import PROFILES


def _context(tmp_path, **kwargs) -> RunContext:
    return RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        hardware=PROFILES["cpu"],
        **kwargs,
    )


def test_initial_drift_recovers_ppo_clipped_surrogate():
    spec = DriftSpec()
    drift = LearnedDrift(spec.initial_params(seed=3), spec)
    ratio = torch.linspace(0.3, 2.5, 101).repeat(3)
    advantages = torch.cat([torch.full((101,), a) for a in (-1.7, 0.4, 2.2)])

    lpo = ratio * advantages - drift(ratio, advantages)
    ppo = torch.min(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages)

    torch.testing.assert_close(lpo, ppo)
    assert float(drift.entropy_coeff) == pytest.approx(0.05)


def test_perturbed_drift_is_nonnegative_and_zero_at_unit_ratio():
    spec = DriftSpec()
    params = spec.initial_params(seed=0) + np.random.default_rng(1).normal(
        0, 0.5, spec.num_params
    )
    drift = LearnedDrift(params, spec)
    ratio = torch.linspace(0.2, 3.0, 57)
    advantages = torch.linspace(-3.0, 3.0, 57)

    assert torch.all(drift(ratio, advantages) >= 0)
    torch.testing.assert_close(drift(torch.ones(57), advantages), torch.zeros(57))


def test_drift_rejects_wrong_parameter_count():
    with pytest.raises(ValueError):
        LearnedDrift(np.zeros(3), DriftSpec())


def test_centered_ranks_handles_ties():
    np.testing.assert_allclose(
        centered_ranks(np.array([1.0, 3.0, 3.0, 0.0])), [-1 / 6, 1 / 3, 1 / 3, -0.5]
    )


def test_es_step_moves_toward_better_antithetic_member():
    es = OpenES(num_pairs=1, sigma=0.1, learning_rate=0.05)
    state = ESState(mean=np.zeros(2))
    noise = np.array([[1.0, -1.0]])
    candidates = es.candidates(state, noise)

    np.testing.assert_allclose(candidates, [[0.1, -0.1], [-0.1, 0.1]])
    stepped = es.step(state, noise, np.array([5.0, 1.0]))

    assert stepped.generation == 1
    np.testing.assert_allclose(stepped.mean, [0.05, -0.05], rtol=1e-6)
    assert ESState.from_dict(stepped.to_dict()).to_dict() == stepped.to_dict()


def test_es_noise_is_reproducible_per_generation():
    es = OpenES(num_pairs=3, sigma=0.1, learning_rate=0.01)
    state = ESState(mean=np.zeros(4))
    np.testing.assert_array_equal(es.noise(state, 42), es.noise(state, 42))
    later = ESState(mean=np.zeros(4), generation=1)
    assert not np.array_equal(es.noise(state, 42), es.noise(later, 42))


def test_antithetic_pairs_share_inner_seed():
    assert shared.inner_seed(42, 3, 1) == shared.inner_seed(42, 3, 1)
    assert shared.inner_seed(42, 3, 1) != shared.inner_seed(42, 3, 2)
    assert shared.inner_seed(42, 3, 1) != shared.inner_seed(42, 4, 1)


def test_fitness_uses_final_tail_of_history():
    assert (
        shared.fitness_from_history([0.0, float("nan"), 10.0, 20.0, 30.0, 40.0], 0.4)
        == 35.0
    )
    assert np.isnan(shared.fitness_from_history([], 0.2))


def test_build_config_is_fresh_and_uses_learned_drift(tmp_path):
    context = _context(tmp_path, smoke=True)
    first = experiment.build_config(context)
    second = experiment.build_config(context)

    assert first is not second
    assert first.learner_class.__name__ == "LearnedDriftPPOLearner"
    assert first.num_env_runners == 0
    assert first.train_batch_size_per_learner == 1_024
    assert first.entropy_coeff == 0.0
    grid = first.learner_config_dict[CANDIDATE_KEY]["grid_search"]
    assert len(grid) == 1 and len(grid[0]["params"]) == DriftSpec().num_params


def test_full_recipe_budgets(tmp_path):
    context = _context(tmp_path)
    inner = experiment.inner_recipe(context)
    meta = experiment.meta_recipe(context)

    assert inner.total_env_steps == 3_000_000
    assert inner.train_batch_size == 8_192
    assert meta.es.num_pairs == 8
    assert meta.generations == 50


def test_load_state_resumes_from_meta_state(tmp_path):
    meta = experiment.meta_recipe(_context(tmp_path, smoke=True))
    saved = ESState(mean=meta.drift.initial_params(0) + 1.0, generation=5)
    path = tmp_path / "meta_state.json"
    path.write_text(json.dumps({"es_state": saved.to_dict()}))

    resumed = shared.load_state(_context(tmp_path, smoke=True, resume_from=path), meta)

    assert resumed.generation == 5
    np.testing.assert_allclose(resumed.mean, saved.mean)
    fresh = shared.load_state(_context(tmp_path, smoke=True, seed=0), meta)
    np.testing.assert_allclose(fresh.mean, meta.drift.initial_params(0))
