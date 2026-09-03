"""Focused tests for the Cycle-5 independent token-flip diagnostic."""

from __future__ import annotations

import importlib

import numpy as np

from experiments.mess3_reward_state_action_symmetry_cycle_5.independent_flip_diagnostic.analysis import (
    independently_flip_state_0_1_tokens,
)


def _observations(tokens: list[int]) -> np.ndarray:
    observations = np.zeros((len(tokens), 6), dtype=np.float32)
    observations[np.arange(len(tokens)), tokens] = 1.0
    observations[:, 3] = 1.0
    return observations


def test_cycle_five_independent_flip_preserves_coarse_history():
    observations = _observations([0, 0, 1, 2, 1, 0, 2])

    randomized = independently_flip_state_0_1_tokens(
        observations,
        rng=np.random.default_rng(17),
        probability=0.5,
    )

    np.testing.assert_array_equal(randomized[:, 2], observations[:, 2])
    np.testing.assert_array_equal(randomized[:, 3:], observations[:, 3:])
    np.testing.assert_array_equal(
        randomized[:, :2].sum(axis=1),
        observations[:, :2].sum(axis=1),
    )
    assert np.any(randomized[:, :2] != observations[:, :2])


def test_cycle_five_independent_flip_experiment_wiring():
    module = importlib.import_module(
        "experiments.mess3_reward_state_action_symmetry_cycle_5."
        "independent_flip_diagnostic.experiment"
    )

    assert module.CYCLE == 5
    assert module.VARIANT == 2
    assert callable(module.run)


def test_cycle_five_seed_queue_uses_cycle_five_sources():
    module = importlib.import_module(
        "experiments.mess3_reward_state_action_symmetry_cycle_5."
        "independent_flip_diagnostic.seed_queue"
    )

    assert "cycle_5" in module.MODULE
    assert module.SOURCE_RESULTS == (
        module.EXPERIMENT_DIR.parent / "variant_2" / "results"
    )
    assert module.SEED_QUEUE_RESULTS == module.EXPERIMENT_DIR / "results"
