from dataclasses import replace
import json

import numpy as np
import pytest

from envs.strata.model import strata_model
from experiments.strata_token_guess_cycle_1 import geometry
from experiments.strata_token_guess_cycle_1.geometry_data import GeometryData, replay_beliefs
from experiments.strata_token_guess_cycle_1.process import environment_config


def _data(seed):
    rng = np.random.default_rng(seed)
    steps = np.tile(np.arange(40), 12)
    tokens = rng.integers(2, size=len(steps))
    observations = np.eye(2)[tokens]
    observations[steps == 0] = 0
    data = GeometryData(
        activations=np.zeros((len(steps), 2, 3)), beliefs=np.zeros((len(steps), 3)),
        observations=observations, episode_ids=np.repeat(np.arange(12), 40),
        episode_steps=steps, mask=steps >= 32,
    )
    beliefs, _ = replay_beliefs(data)
    return replace(data, beliefs=beliefs, activations=np.stack([beliefs * 0.4, beliefs], axis=1))


def test_strata_pending_map_and_null_direction_are_not_wings():
    direction, mapping = geometry.ntp_null_direction()
    assert np.linalg.norm(direction) == pytest.approx(1)
    assert abs(direction.sum()) < 1e-12
    np.testing.assert_allclose(direction @ mapping, 0, atol=1e-12)
    assert not np.allclose(direction, np.array([1, 0, -1]) / np.sqrt(2))
    data = _data(41)
    beliefs, pending = replay_beliefs(data)
    np.testing.assert_allclose(beliefs @ mapping, pending, atol=1e-12)
    assert np.max(np.abs(beliefs @ strata_model(**environment_config()["model"]["kwargs"]).emission_matrix - pending)) > 1e-3


def test_legacy_factor_report_uses_experiment_parameters_not_library_defaults(monkeypatch):
    from experiments.strata_token_guess_cycle_1 import analysis

    parameters = environment_config()["model"]["kwargs"]
    calls = []

    def explicit_model(*, alpha, t0, t1):
        calls.append({"alpha": alpha, "t0": t0, "t1": t1})
        return strata_model(alpha=alpha, t0=t0, t1=t1)

    monkeypatch.setattr(analysis, "strata_model", explicit_model)
    target = np.random.default_rng(72).dirichlet(np.ones(3), size=50)
    predicted = np.roll(target, 1, axis=1)
    report = analysis._factor_report(predicted, target)
    assert calls == [parameters]
    emission = strata_model(**parameters).emission_matrix
    direction = np.cross(np.ones(3), emission[:, 0])
    direction /= np.linalg.norm(direction)
    expected = analysis._metrics((predicted @ direction)[:, None], (target @ direction)[:, None])
    assert report["delta"] == expected


def test_held_out_controls_and_initialization_comparison_are_json_safe():
    train, test = _data(1), _data(2)
    initial = tuple(replace(data, activations=np.zeros_like(data.activations)) for data in (train, test))
    policies = [np.full((int(data.mask.sum()), 2), 0.5) for data in (train, test)]
    report, target, decoded = geometry.analyze_samples(
        train, test, policy_train=policies[0], policy_test=policies[1], initial=initial,
        repeats=1, resamples=5,
    )
    json.dumps(report, allow_nan=False)
    summary = geometry.compact_summary(report)
    json.dumps(summary, allow_nan=False)
    assert report["replay_max_abs_error"] < 1e-12
    for name, layer in report["layers"].items():
        assert layer["belief_probe"]["r_squared"] == pytest.approx(1)
        assert layer["belief_probe"]["ntp_null_contrast"]["r_squared"] == pytest.approx(1)
        np.testing.assert_allclose(target, decoded[name], atol=1e-12)
        assert layer["over_initialization"]["mse_improvement"] > 0
        assert layer["surplus"]["ntp"]["mse_improvement"] > 1e-6
        assert layer["belief_probe"]["fit"]["n_groups"] == 12
        assert summary["layers"][name]["outside_simplex_fraction"] == layer["belief_probe"]["outside_simplex_fraction"]
        assert summary["layers"][name]["max_sum_error"] == layer["belief_probe"]["max_sum_error"]
    assert report["policy"]["conditional_expected_accuracy"] == pytest.approx(0.5)
    assert len(report["alternative_search"]["selected_ids"]) <= 3


def test_alternative_selection_ignores_activations_and_rejects_identity(monkeypatch):
    data = _data(4)
    chosen, report = geometry.select_alternatives(data)
    corrupted = replace(data, activations=np.full_like(data.activations, np.nan))
    assert geometry.select_alternatives(corrupted) == (chosen, report)
    monkeypatch.setattr(geometry, "alternative_parameters", lambda: [dict(geometry.PARAMETERS)])
    chosen, report = geometry.select_alternatives(data)
    assert chosen == []
    assert report["status"] == "no_eligible_alternative"


@pytest.mark.parametrize("constant_target", [False, True])
def test_simplex_uses_raw_outputs_shared_colors_and_nonoverlapping_headings(monkeypatch, constant_target):
    data = _data(8)
    target = data.beliefs[data.mask]
    if constant_target:
        target[:] = [0.25, 0.5, 0.25]
    decoded = target.copy()
    decoded[0] = [1.2, -0.3, 0.1]
    expected_target, expected_decoded = target.copy(), decoded.copy()
    direction, _ = geometry.ntp_null_direction()
    score = geometry._score(decoded, target, direction)
    report = {"layers": {"layer_1": {"belief_probe": {}}, "layer_2": {"belief_probe": score}},
              "metadata": {"seed": 7, "agent_steps": 100, "n_fit": 100}}
    captured = []
    original = geometry.plot_belief_comparison

    def capture(targets, predicted, **kwargs):
        figure = original(targets, predicted, **kwargs)
        captured.append((targets.copy(), predicted.copy(), kwargs, figure))
        return figure

    monkeypatch.setattr(geometry, "plot_belief_comparison", capture)
    png = geometry.render_simplex(target, decoded, report)
    assert isinstance(png, bytes) and png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(captured) == 1
    targets, predicted, kwargs, figure = captured[0]
    np.testing.assert_array_equal(targets, expected_target)
    np.testing.assert_array_equal(predicted, expected_decoded)
    np.testing.assert_array_equal(target, expected_target)
    np.testing.assert_array_equal(decoded, expected_decoded)
    np.testing.assert_array_equal(kwargs["coordinates"], geometry.VERTICES)
    np.testing.assert_array_equal(kwargs["point_colors"], target @ geometry.COLORS)
    assert kwargs["state_labels"] == ("State 0", "State 1", "State 2")
    assert kwargs["seed"] == report["metadata"]["seed"]
    axes = figure.axes
    order = np.random.default_rng(7).permutation(len(target))
    for axis in axes:
        np.testing.assert_allclose(axis.collections[0].get_facecolors()[:, :3], (target @ geometry.COLORS)[order])
    assert axes[0].get_xlim() == axes[1].get_xlim()
    assert axes[0].get_ylim() == axes[1].get_ylim()
    assert not geometry.plt.fignum_exists(figure.number)
    figure.canvas.draw()
    subtitle = next(text for text in figure.texts if "Final checkpoint" in text.get_text())
    assert "Last layer (layer_2), pre-final-LayerNorm residual" in subtitle.get_text()
    assert "100 fit and 96 independent held-out samples" in subtitle.get_text()
    assert figure._suptitle.get_window_extent().y0 > subtitle.get_window_extent().y1
    assert subtitle.get_window_extent().y0 > max(axis.title.get_window_extent().y1 for axis in axes)
    contrast = score["ntp_null_contrast"]["r_squared"]
    label = "N/A" if constant_target else f"{contrast:.3f}"
    annotation = next(text for text in figure.texts if "NTP-null contrast" in text.get_text())
    assert annotation.get_text() == f"NTP-null contrast R² = {label}"
    assert annotation.get_position() == (0.5, 0.21)
    assert annotation.get_window_extent().y1 < min(axis.get_window_extent().y0 for axis in axes)
    core_metrics = next(text for text in figure.texts if text.get_text().startswith("Raw MSE"))
    assert annotation.get_window_extent().y0 > core_metrics.get_window_extent().y1
    assert any("delay-one decision-time arrival beliefs" in text.get_text() for text in figure.texts)
    assert any("Strata's NTP-null direction is computed from its own emission map" in text.get_text() for text in figure.texts)
    if constant_target:
        assert score["r_squared"] is None and contrast is None


def test_training_graph_uses_bayes_ceiling_without_clipping(monkeypatch):
    import matplotlib.axes

    axes = []
    original = matplotlib.axes.Axes.set_ylim

    def set_ylim(axis, bottom=None, top=None, **kwargs):
        if bottom == 0 and top is not None:
            axes.append(axis)
        return original(axis, bottom, top, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylim", set_ylim)
    rows = [{"agent_steps": 0, "token_accuracy": 0.60}, {"agent_steps": 2500000, "token_accuracy": 0.695}]
    png = geometry.render_training(rows, bayes_lower=0.704, bayes_upper=0.705, constant_accuracy=0.7025)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert axes[-1].get_ylim() == (0, 70.5)
    figure = axes[-1].figure
    figure.canvas.draw()
    title, subtitle = figure.texts[:2]
    assert title.get_window_extent().y0 > subtitle.get_window_extent().y1
    expanded = geometry.render_training(rows, bayes_lower=0.65, bayes_upper=0.66, constant_accuracy=0.65)
    assert expanded.startswith(b"\x89PNG\r\n\x1a\n")
    assert axes[-1].get_ylim()[1] > 69.5
    assert any("empirical" in text.get_text().lower() for text in axes[-1].figure.texts)
