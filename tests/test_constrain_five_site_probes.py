from copy import deepcopy

import numpy as np
import pytest

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import constrain_five_site_probes as correction


def test_established_simplex_projection_and_coarse_simplex():
    project, digest = correction.projector()
    assert len(digest) == 64
    values = np.array([[2., -1., 0.], [.2, .3, .5], [-4., -4., -4.]])
    projected = project(values, "belief")
    np.testing.assert_allclose(projected, [[1, 0, 0], [.2, .3, .5], [1/3, 1/3, 1/3]])
    np.testing.assert_allclose(projected.sum(axis=1), 1)
    coarse = np.array([[-1.], [.4], [2.]])
    actual = project(np.concatenate([1-coarse, coarse], axis=1), "belief")[:, 1:2]
    np.testing.assert_allclose(actual, [[0], [.4], [1]])
    np.testing.assert_array_equal(values[0], [2, -1, 0])


@pytest.mark.parametrize("values", [np.empty((0, 3)), np.array([[np.nan, 0.]]),
                                    np.array([[np.inf, 0.]]), np.zeros((2, 1))])
def test_projection_rejects_empty_nonfinite_or_invalid_beliefs(values):
    project, _ = correction.projector()
    with pytest.raises(ValueError):
        project(values, "belief")


def fixture_reports():
    return [{"variant": 1, "seed": 51, "checkpoint_step": 0,
             "scores": {"block_1": {"b": {"error": 2., "r_squared": -1.},
                                      "rho": {"error": .5, "r_squared": .5}}}}]


def fixture_replacement():
    return {"variant": 1, "seed": 51, "checkpoint_step": 0, "site": "block_1", "target": "b",
            "old": {"error": 2., "r_squared": -1.}, "new": {"error": .8, "r_squared": .2}}


def test_replacements_only_touch_selected_scores_without_mutating_inputs():
    reports = fixture_reports()
    saved = deepcopy(reports)
    corrected = correction.apply_replacements(reports, [fixture_replacement()])
    assert reports == saved
    assert corrected[0]["scores"]["block_1"]["b"]["error"] == .8
    assert corrected[0]["scores"]["block_1"]["rho"] == reports[0]["scores"]["block_1"]["rho"]
    assert correction.apply_replacements([], []) == []


@pytest.mark.parametrize("mode", ["missing", "duplicate", "extra", "changed_old"])
def test_replacements_reject_inexact_scope_or_original(mode):
    row = fixture_replacement()
    replacements = [row]
    if mode == "missing":
        replacements = []
    elif mode == "duplicate":
        replacements.append(deepcopy(row))
    elif mode == "extra":
        replacements.append({**row, "target": "rho"})
    else:
        row["old"] = {"error": 3.}
    with pytest.raises(ValueError):
        correction.apply_replacements(fixture_reports(), replacements)
