from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from analysis.probes import predictive_belief_update

from experiments.mess3_reward_state_action_symmetry_cycle_7.coarse_probe_analysis import (
    _build_coarse_filter_spec,
    coarse_beliefs,
    coarse_filter_spec,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.coarse_probe_campaign import (
    STAGES,
    matched_seed_comparison,
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


@pytest.mark.parametrize("variant", [1, 2])
def test_two_state_coarse_filter_specs_are_exact_quotients(
    variant: int,
) -> None:
    initial, emission, transitions, diagnostics = _build_coarse_filter_spec(variant)

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
    expected_positive_state_three = (
        0.7492347090660089 if variant == 1 else 0.12949118140280794
    )
    np.testing.assert_allclose(
        transitions[1],
        [
            [0.6675721382568806, 0.33242786174311934],
            [
                1.0 - expected_positive_state_three,
                expected_positive_state_three,
            ],
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
    assert diagnostics["filter_kind"] == "exact_two_state_quotient"
    assert diagnostics["strong_lumpability_over_states_1_2"] is True
    assert diagnostics["non_lumpable_actions"] == []


def test_variant_one_constructs_its_own_action_transitions() -> None:
    _, _, variant_one = coarse_filter_spec(variant=1)
    _, _, variant_two = coarse_filter_spec(variant=2)

    assert not np.allclose(variant_one[1], variant_two[1])
    np.testing.assert_allclose(
        variant_one[1],
        [
            [0.6675721382568806, 0.33242786174311934],
            [0.2507652909339911, 0.7492347090660089],
        ],
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("variant", [1, 2, 3])
def test_coarse_filter_resets_and_uses_previous_executed_action(
    variant: int,
) -> None:
    data = _history()
    initial, emission, transitions = coarse_filter_spec(variant=variant)
    actual = coarse_beliefs(data, variant=variant)

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


@pytest.mark.parametrize("variant", [1, 2, 3])
def test_coarse_target_is_not_full_state_three_projection(variant: int) -> None:
    data = _history()
    coarse = coarse_beliefs(data, variant=variant)[:, -1]
    full = replay_beliefs(data, variant=variant)[:, 2]

    assert np.max(np.abs(coarse - full)) > 1e-3


def test_variant_three_uses_full_state_coarse_observation_filter() -> None:
    initial, emission, transitions, diagnostics = _build_coarse_filter_spec(3)

    assert initial.shape == (3,)
    assert emission.shape == (3, 2)
    assert all(matrix.shape == (3, 3) for matrix in transitions.values())
    assert diagnostics["filter_kind"] == "exact_full_three_state"
    assert diagnostics["strong_lumpability_over_states_1_2"] is False
    assert diagnostics["non_lumpable_actions"] == [1, 2]
    _, _, reward_state_transitions = coarse_filter_spec(variant=2)
    assert transitions[1].shape != reward_state_transitions[1].shape


def test_variant_three_filter_matches_manual_full_state_update() -> None:
    data = _history()
    initial, emission, transitions = coarse_filter_spec(variant=3)
    actual = coarse_beliefs(data, variant=3)

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


def test_campaign_covers_matched_checkpoints_for_all_conditions(
    tmp_path: Path,
) -> None:
    package = __import__(
        "experiments.mess3_reward_state_action_symmetry_cycle_7",
        fromlist=["__file__"],
    )
    root = Path(package.__file__).resolve().parents[2]
    items = work_items(root, tmp_path)

    assert len(items) == 315
    assert {item.seed for item in items} == set(range(42, 57))
    assert {item.stage for item in items} == set(STAGES)
    assert {item.variant for item in items} == {1, 2, 3}
    assert all(
        len([item for item in items if item.variant == variant and item.seed == seed])
        == 7
        for variant in (1, 2, 3)
        for seed in range(42, 57)
    )


def _target(error: float) -> dict[str, float]:
    return {
        "global_mse_ratio": error,
        "r_squared": 1.0 - error,
    }


def _comparison_conditions() -> dict[str, dict[str, object]]:
    errors = {
        "variant_1_ctx32_ent": (0.20, 0.10),
        "variant_2_ctx32_ent": (0.05, 0.20),
        "variant_3_ctx32_ent": (0.20, 0.05),
    }
    return {
        condition: {
            "seeds": {
                str(seed): [
                    {
                        "stage": stage,
                        "targets": {
                            "coarse_b2": _target(coarse),
                            "full_rho": _target(full),
                        },
                    }
                    for stage in STAGES
                ]
                for seed in range(42, 57)
            }
        }
        for condition, (coarse, full) in errors.items()
    }


def test_matched_comparison_direction_and_interpretation() -> None:
    comparison = matched_seed_comparison(_comparison_conditions())
    final = comparison["stages"]["final"]["conditions"]

    assert final["variant_2_ctx32_ent"]["mean_full_minus_coarse"] == pytest.approx(0.15)
    assert final["variant_3_ctx32_ent"]["mean_full_minus_coarse"] == pytest.approx(
        -0.15
    )
    readout = comparison["final_descriptive_readout"]
    assert readout["reward_state_uniquely_favors_coarse_by_mean"] is True
    assert readout["full_state_favors_full_rho_by_mean"] is True
    assert "not environmental" in readout["scope"]


def test_compact_result_records_matched_protocol_and_filter_semantics() -> None:
    result_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "mess3_reward_state_action_symmetry_cycle_7"
        / "results"
        / "coarse_probe_trajectories.json"
    )
    result = json.loads(result_path.read_text())

    assert result["schema_version"] == 2
    assert result["checkpoint_selection"]["record_count"] == 315
    assert result["analysis"]["n_fit"] == 60_000
    assert result["analysis"]["n_test"] == 80_000
    assert result["analysis"]["warmup_steps_per_episode"] == 64
    assert result["analysis"]["ridge"] == 1e-6
    models = {
        condition: report["coarse_model"]
        for condition, report in result["conditions"].items()
    }
    assert models["variant_1_ctx32_ent"]["filter_kind"] == "exact_two_state_quotient"
    assert models["variant_3_ctx32_ent"]["filter_kind"] == "exact_full_three_state"
    assert models["variant_3_ctx32_ent"]["non_lumpable_actions"] == [1, 2]
    assert (
        result["matched_seed_comparison"]["metric"]["preference"]
        == "full_minus_coarse > 0 favors the coarse-observation target; "
        "full_minus_coarse < 0 favors full rho."
    )
