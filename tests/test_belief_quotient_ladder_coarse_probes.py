from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from analysis.probes import predictive_belief_update

from experiments.mess3_reward_state_action_symmetry_cycle_7.coarse_probe_analysis import (
    coarse_beliefs,
    coarse_filter_spec,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.coarse_probe_campaign import (
    work_items,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.control_analysis import (
    HistoryData,
    replay_beliefs,
)


def _history() -> HistoryData:
    observations = np.asarray(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    return HistoryData(
        observations=observations,
        activations=np.zeros((len(observations), 2), dtype=np.float64),
        beliefs=np.zeros((len(observations), 3), dtype=np.float64),
        episode_ids=np.asarray([0, 0, 0, 1, 1], dtype=np.int64),
        episode_steps=np.asarray([0, 1, 2, 0, 1], dtype=np.int64),
        mask=np.ones(len(observations), dtype=np.bool_),
        metadata={},
    )


def test_coarse_filter_spec_matches_reward_state_quotient() -> None:
    initial, emission, transitions = coarse_filter_spec()

    np.testing.assert_allclose(
        initial,
        [2 / 3, 1 / 3],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        emission,
        [[0.925, 0.075], [0.15, 0.85]],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        transitions[0],
        [[0.9, 0.1], [0.6, 0.4]],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        transitions[1],
        [
            [0.6675721382568806, 0.33242786174311934],
            [0.8705088185971921, 0.12949118140280794],
        ],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        transitions[2],
        [
            [0.9758075451312032, 0.02419245486879684],
            [0.8705088185971921, 0.12949118140280794],
        ],
        rtol=0.0,
        atol=1e-12,
    )
    for matrix in transitions.values():
        np.testing.assert_allclose(
            matrix.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=1e-12,
        )


def test_coarse_filter_resets_and_uses_previous_executed_action() -> None:
    data = _history()
    initial, emission, transitions = coarse_filter_spec()
    actual = coarse_beliefs(data)

    first = predictive_belief_update(initial, np.diag(emission[:, 0]))
    second = predictive_belief_update(
        first,
        transitions[1] @ np.diag(emission[:, 0]),
    )
    third = predictive_belief_update(
        second,
        transitions[0] @ np.diag(emission[:, 1]),
    )
    fourth = predictive_belief_update(initial, np.diag(emission[:, 1]))
    fifth = predictive_belief_update(
        fourth,
        transitions[2] @ np.diag(emission[:, 0]),
    )
    np.testing.assert_allclose(
        actual,
        np.stack((first, second, third, fourth, fifth)),
        rtol=0.0,
        atol=1e-12,
    )


def test_coarse_target_is_not_full_state_three_projection() -> None:
    data = _history()
    coarse = coarse_beliefs(data)[:, 1]
    full = replay_beliefs(data, variant=2)[:, 2]

    assert np.max(np.abs(coarse - full)) > 1e-3


def test_coarse_filter_rejects_non_lumpable_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = np.asarray(
        [
            [0.75, 0.15, 0.10],
            [0.15, 0.75, 0.10],
            [0.30, 0.30, 0.40],
        ],
        dtype=np.float64,
    )
    non_lumpable = baseline.copy()
    non_lumpable[1] = [0.10, 0.75, 0.15]
    environment = SimpleNamespace(
        model=SimpleNamespace(
            initial_distribution=np.full(3, 1 / 3),
            emission_matrix=np.asarray(
                [
                    [0.85, 0.075, 0.075],
                    [0.075, 0.85, 0.075],
                    [0.075, 0.075, 0.85],
                ],
                dtype=np.float64,
            ),
        ),
        action_space=SimpleNamespace(n=3),
        task=SimpleNamespace(
            transition_matrix_for_action=lambda action: (
                non_lumpable if action == 0 else baseline
            )
        ),
        close=lambda: None,
    )
    monkeypatch.setattr(
        (
            "experiments.mess3_reward_state_action_symmetry_cycle_7."
            "coarse_probe_analysis.HMMEnv"
        ),
        lambda config: environment,
    )

    with pytest.raises(ValueError, match="not strongly lumpable"):
        coarse_filter_spec()


def test_variant_two_campaign_covers_all_saved_checkpoints(
    tmp_path: Path,
) -> None:
    package = __import__(
        "experiments.mess3_reward_state_action_symmetry_cycle_7",
        fromlist=["__file__"],
    )
    root = Path(package.__file__).resolve().parents[2]
    items = work_items(root, tmp_path)

    assert len(items) == 105
    assert {item.seed for item in items} == set(range(42, 57))
    assert {item.stage for item in items} == {
        "init",
        "iteration_1",
        "iteration_2",
        "iteration_4",
        "iteration_8",
        "iteration_16",
        "final",
    }
    assert all(item.variant == 2 for item in items)
