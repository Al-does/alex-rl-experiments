from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from experiments.wing_two_factor_explore_cycle_1 import control_analysis as analysis
from experiments.wing_two_factor_explore_cycle_1.control_data import ControlData, replay_beliefs


def _data(seed, passive):
    rng = np.random.default_rng(seed)
    count = 256
    steps = np.tile(np.arange(32), 8)
    tokens = rng.integers(2, size=(count, 2))
    actions = rng.integers(3, size=(count, 2))
    actions[steps == 0] = -1
    if passive:
        tokens[steps == 0] = -1
    observations = np.zeros((count, 4 if passive else 10))
    active = tokens[:, 0] >= 0
    observations[active, tokens[active, 0] * 2 + tokens[active, 1]] = 1
    if not passive:
        rows = np.flatnonzero(steps > 0)
        observations[rows, 4 + actions[rows, 0]] = 1
        observations[rows, 7 + actions[rows, 1]] = 1
    data = ControlData(
        activations=np.zeros((count, 1, 6)), beliefs=np.zeros((count, 2, 3)),
        policy_probabilities=np.full((count, 4 if passive else 9), 1 / (4 if passive else 9)),
        observations=observations, episode_ids=np.repeat(np.arange(8), 32),
        episode_steps=steps, mask=steps >= 4, tokens=tokens, preceding_actions=actions,
    )
    parameters = {"alpha": 0.94, "x": 0.4, "strength": None if passive else 1.0}
    beliefs, _ = replay_beliefs(data, **parameters)
    return replace(data, beliefs=beliefs, activations=beliefs.reshape(count, 1, 6)), parameters


@pytest.mark.parametrize("passive", [True, False])
def test_complete_battery_decodes_beliefs_and_serializes_without_nonfinite_numbers(passive):
    train, parameters = _data(100, passive)
    test, _ = _data(200, passive)
    report = analysis.analyze_control_samples(
        train, test, parameters=parameters, condition="token_guess" if passive else "reward_both",
        n_resamples=5, n_null_repeats=1,
    )
    json.dumps(report, allow_nan=False)
    summary = analysis.summarize_report(report)
    json.dumps(summary, allow_nan=False)
    assert summary["factors"]["factor_1"]["belief_r_squared"] == report["factors"]["factor_1"]["layers"]["layer_1"]["belief_probe"]["r_squared"]
    assert summary["layer_selection"] == "last layer, not best test layer"
    assert report["replay_max_abs_error"] < 1e-12
    assert report["metadata"]["rl_policy_is_not_ntp"]
    for factor in report["factors"].values():
        assert factor["baselines"]["joint_ntp"]["fit"]["n_features"] == 4
        assert factor["baselines"]["log_joint_ntp"]["fit"]["n_features"] == 4
        layer = factor["layers"]["layer_1"]
        assert layer["belief_probe"]["mse"] < 1e-20
        assert layer["surplus_over_baselines"]["ntp"]["mse_improvement"] > 1e-6
        assert layer["same_suffix_pairs"]["1"]["n_pairs"] > 0
        assert layer["same_suffix_pairs"]["1"]["belief_difference"]["mse"] < 1e-20
        assert layer["nulls"]["0"]["shuffled_training_labels"]["mse"] > 1e-5
        assert layer["nulls"]["0"]["gaussian_covariance_matched"]["mse"] > 1e-5
        assert layer["belief_probe"]["fit"]["n_groups"] == 8


def test_alternative_selection_never_uses_activations_and_can_report_no_eligible_candidates(monkeypatch):
    train, parameters = _data(123, False)
    selected, first = analysis._select_alternatives(train, parameters, seed=42)
    _, second = analysis._select_alternatives(replace(train, activations=np.full_like(train.activations, np.nan)), parameters, seed=42)
    assert first == second
    assert all(candidate["eligible"] for candidate in selected)
    monkeypatch.setattr(analysis, "candidate_parameters", lambda **kwargs: [kwargs])
    selected, report = analysis._select_alternatives(train, parameters, seed=42)
    assert selected == []
    assert report["status"] == "no_eligible_alternative"
    assert report["candidates"][0]["selection_min_permutation_mse_per_factor"] == [0, 0]


def test_pairs_never_reuse_rows_or_pair_within_episode():
    keys = np.array([[1], [1], [1], [1], [2], [3]])
    groups = np.array([0, 0, 1, 2, 0, 0])
    pairs = analysis._matched_pairs(keys, groups)
    assert len(np.unique(pairs)) == pairs.size
    assert (groups[pairs[:, 0]] != groups[pairs[:, 1]]).all()
    np.testing.assert_array_equal(keys[pairs[:, 0]], keys[pairs[:, 1]])


def test_replay_alignment_fails_before_fitting():
    train, parameters = _data(1, True)
    test, _ = _data(2, True)
    with pytest.raises(ValueError, match="disagree"):
        analysis.analyze_control_samples(
            replace(train, beliefs=np.zeros_like(train.beliefs)), test,
            parameters=parameters, condition="token_guess", n_resamples=5, n_null_repeats=1,
        )


@pytest.mark.parametrize("reward_state", [0, 1, 2])
def test_reward_control_uses_the_requested_state(reward_state):
    from envs.wing.model import controlled_kernels

    train, parameters = _data(12, False)
    transitions = controlled_kernels(0.94, 0.4, 1.0).sum(axis=1)
    first = np.stack([train.beliefs[:, 0] @ transitions[a, :, reward_state] for a in range(3)], axis=1)
    second = np.stack([train.beliefs[:, 1] @ transitions[a, :, reward_state] for a in range(3)], axis=1)
    expected = (np.repeat(first, 3, axis=1) + np.tile(second, (1, 3))) / 2
    np.testing.assert_allclose(analysis._reward_features(train.beliefs, parameters, "reward_both", reward_state), expected)


def test_shuffled_history_targets_are_recomputed_not_taken_from_stale_cache():
    from experiments.wing_two_factor_explore_cycle_1.control_data import shuffled_history

    train, parameters = _data(52, False)
    test, _ = _data(53, False)
    shifted = []
    for data in (train, test):
        shuffled = shuffled_history(data, seed=10)
        beliefs, _ = replay_beliefs(shuffled, **parameters)
        shifted.append(replace(shuffled, activations=beliefs.reshape(len(beliefs), 1, 6)))
    report = analysis.analyze_control_samples(
        train, test, parameters=parameters, condition="reward_both",
        n_resamples=2, n_null_repeats=1, shifted=tuple(shifted),
    )
    for factor in report["factors"].values():
        stress = factor["layers"]["layer_1"]["shuffled_history_stress"]
        assert stress["frozen_original_probe"]["mse"] < 1e-20
        assert stress["refitted_on_shuffled_histories"]["mse"] < 1e-20


def test_cli_refuses_existing_output_before_loading_checkpoint(tmp_path):
    with pytest.raises(SystemExit):
        analysis.main(["--study", "token_guess", "--checkpoint", str(tmp_path), "--output", str(tmp_path)])


def test_simplex_picture_preserves_raw_predictions_and_paired_colors(monkeypatch):
    from experiments.wing_token_guess_cycle_1 import simplex

    rng = np.random.default_rng(15)
    target = rng.dirichlet(np.ones(3), size=(50, 2))
    decoded = target.copy()
    decoded[0, 0] = [1.2, -0.3, 0.1]
    saved = decoded.copy()
    captured = []
    original = simplex.simplex_scatter

    def capture(axis, points, **kwargs):
        captured.append((axis, points.copy(), kwargs["colors"].copy()))
        return original(axis, points, **kwargs)

    monkeypatch.setattr(simplex, "simplex_scatter", capture)
    png = simplex.render_simplexes(target, decoded, layer=3, n_fit=100, agent_steps=123, seed=42)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(captured) == 4
    figure = captured[0][0].figure
    figure.canvas.draw()
    title, subtitle = figure.texts[:2]
    assert subtitle.get_window_extent().y1 < title.get_window_extent().y0
    assert subtitle.get_window_extent().y0 > max(axis.title.get_window_extent().y1 for axis, _, _ in captured[:2])
    order = np.random.default_rng(42).permutation(len(target))
    np.testing.assert_array_equal(captured[1][1], decoded[order, 0])
    assert captured[1][1].min() == -0.3
    for factor in range(2):
        np.testing.assert_array_equal(captured[factor * 2][2], captured[factor * 2 + 1][2])
        np.testing.assert_allclose(captured[factor * 2][2], target[order, factor] @ simplex.VERTEX_COLORS)
    assert len({axis.get_xlim() for axis, _, _ in captured}) == 1
    assert len({axis.get_ylim() for axis, _, _ in captured}) == 1
    np.testing.assert_array_equal(decoded, saved)
    metrics = simplex.geometry_metrics(target, decoded)
    assert metrics["factor_1"]["outside_simplex_fraction"] == 1 / 50
    assert metrics["factor_1"]["mse"] == pytest.approx(np.mean((target[:, 0] - decoded[:, 0]) ** 2))
    assert metrics["factor_2"]["r_squared"] == 1
    assert png == simplex.render_simplexes(target, decoded, layer=3, n_fit=100, agent_steps=123, seed=42)
    invalid = decoded.copy()
    invalid[0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="sum to one"):
        simplex.geometry_metrics(target, invalid)


def test_simplex_fit_reuses_independent_seed_streams_and_last_layer(monkeypatch):
    from experiments.wing_token_guess_cycle_1 import simplex

    train, _ = _data(51, True)
    test, _ = _data(52, True)
    datasets = [replace(data, activations=np.concatenate([np.zeros_like(data.activations), data.activations], axis=1)) for data in (train, test)]
    calls = []

    def collect(module, **kwargs):
        calls.append(kwargs)
        return datasets[len(calls) - 1]

    monkeypatch.setattr(simplex, "collect_control_data", collect)
    target, decoded, metadata = simplex.collect_and_fit(object(), seed=42, n_steps=224)
    expected_seeds = [int(stream.generate_state(1)[0]) for stream in np.random.SeedSequence(42).spawn(4)[:2]]
    assert [call["seed"] for call in calls] == expected_seeds
    assert expected_seeds[0] != expected_seeds[1]
    assert metadata["layer"] == 2
    np.testing.assert_allclose(target, test.beliefs[test.mask], atol=1e-12)
    np.testing.assert_allclose(decoded, target, atol=1e-12)
