import copy
import json

import numpy as np
import pytest

from experiments.belief_quotient_history_jax_2026_11.env import belief_reset, belief_update
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
from experiments.within_class_belief_probes_2026_11.eps02 import (
    STUDIES, directory, filter_histories, selected_runs, validate_recipe,
)
from experiments.within_class_belief_probes_2026_11.eps02_metrics import class_variance, summarize_seeds


def test_inventory_selection_full_target_and_no_seed_cache_collision():
    runs = selected_runs()
    assert len(runs) == 48
    assert len({directory(r) for r in runs}) == 48
    primary = set()
    for run in runs:
        recipe = json.loads((run/"resolved_recipe.json").read_text())
        validate_recipe(recipe)
        if recipe["study"] == STUDIES[0]:
            primary.add((recipe["variant"], recipe["seed"]))
    assert primary == {(v, s) for v in (1, 2, 3) for s in range(42, 57)}


@pytest.mark.parametrize("field,value", [("temperature", 1), ("exploration_epsilon", .02),
                                         ("entropy_schedule", [[0, .005]])])
def test_reject_wrong_training(field, value):
    recipe = json.loads((selected_runs()[0]/"resolved_recipe.json").read_text())
    recipe["training"][field] = value
    with pytest.raises(ValueError):
        validate_recipe(recipe)


def test_reject_old_emissions():
    recipe = json.loads((selected_runs()[0]/"resolved_recipe.json").read_text())
    recipe["process"]["emission"][2] = [0, 0, 1]
    with pytest.raises(ValueError):
        validate_recipe(recipe)


def test_filter_is_full_current_state_posterior_using_previous_executed_action():
    process = environment.ladder_process(3)
    obs = np.zeros((1, 3, 6))
    obs[0, :, :3] = np.eye(3)[[2, 0, 1]]
    obs[0, 1:, 3:] = np.eye(3)[[1, 2]]
    actual = filter_histories(obs, process)
    belief = belief_reset(process, 2)
    expected = [np.asarray(belief)]
    for action, token in ((1, 0), (2, 1)):
        belief = belief_update(process, belief, action, token)
        expected.append(np.asarray(belief))
    np.testing.assert_allclose(actual[0], expected, atol=1e-7)
    np.testing.assert_allclose(actual.sum(-1), 1)
    assert actual[0, 0, 2] < 1
    with pytest.raises(ValueError):
        filter_histories(np.empty((1, 0, 6)), process)


def test_between_class_decomposition_and_fit_mean_predictor():
    y = np.array([[.8, .1, .1], [.6, .2, .2], [.1, .8, .1], [.2, .6, .2]])
    c = np.array([0, 0, 1, 1])
    result = class_variance(y, c, y, c)
    assert result["between_class_fraction"] == pytest.approx(result["class_only_held_out"]["r2"])
    assert result["global_sst"] == pytest.approx(result["within_class_sst"]/(1-result["between_class_fraction"]))
    assert class_variance(y, np.zeros(4), y, np.zeros(4))["between_class_fraction"] == 0
    assert class_variance(y, c, y[:2], c[:2])["class_only_excluded_rows"] == 2
    assert class_variance(np.ones((2, 3))/3, c[:2], y, c)["between_class_fraction"] is None
    empty = class_variance(np.empty((0, 3)), np.empty(0), y, c)
    assert empty["between_class_fraction"] is None and empty["class_only_held_out"]["r2"] is None
    with pytest.raises(ValueError):
        class_variance(y, c[:1], y, c)


def record(seed, trained, initialized, eligible=True):
    metric = {"r2": trained}
    stage = {"global": metric, "one_class": not eligible,
             "variance": {"between_class_fraction": .2, "class_only_held_out": metric},
             "metrics": {k: {t: metric for t in ("belief", "delta", "rho")}
                         for k in ("full_h", "head_blind", "logits")},
             "controls": {k: {"belief": metric} for k in
                          ("history_shuffle", "random_input", "label_permutation")}}
    matched = copy.deepcopy(stage)
    matched["metrics"]["full_h"]["belief"]["r2"] = initialized
    return {"seed": seed, "within_class_eligible": eligible,
            "final": stage, "init": stage, "matched_init": matched}


def test_equal_seed_paired_bootstrap_and_one_class_exclusion():
    records = [record(42, .8, .7), record(43, .6, .5), record(44, 1., .3, False)]
    result = summarize_seeds(records, eligible_only=True, n=10000)
    assert result["n_seeds"] == 2 and result["one_class_seeds"] == [44]
    assert result["positive_final_minus_matched_init"] == 2
    assert result["metrics"]["within_trained"]["mean"] == pytest.approx(.7)
    np.testing.assert_allclose(result["metrics"]["final_minus_matched_init"]["ci95"], [.1, .1])
    all_seeds = summarize_seeds(records, eligible_only=False)
    assert all_seeds["metrics"]["within_trained"]["mean"] == pytest.approx(.8)
    empty = summarize_seeds([records[-1]], eligible_only=True)
    assert empty["n_seeds"] == 0 and empty["metrics"]["within_trained"]["mean"] is None
    assert summarize_seeds(records[:1], eligible_only=True)["metrics"]["within_trained"]["ci95"] is None
    with pytest.raises(ValueError):
        summarize_seeds([records[0], records[0]], eligible_only=False)
    with pytest.raises(ValueError):
        summarize_seeds([], eligible_only=False, n=0)


def test_undefined_metrics_have_explicit_seed_coverage():
    records = [record(42, None, .7), record(43, .6, .5)]
    result = summarize_seeds(records, eligible_only=True)
    metric = result["metrics"]["within_trained"]
    assert metric["n_seeds"] == 1 and metric["excluded_seeds"] == [42]
    assert metric["mean"] == .6 and metric["ci95"] is None


def test_report_append_preserves_other_sections_and_replaces_only_own_block(tmp_path):
    from experiments.within_class_belief_probes_2026_11.eps02_report import append_section, START

    path = tmp_path/"README.md"
    path.write_text("Previous analysis\n")
    append_section(path, "New results")
    path.write_text(path.read_text()+"Following analysis\n")
    append_section(path, "Updated results")
    text = path.read_text()
    assert text.count(START) == 1 and "New results" not in text
    assert "Previous analysis" in text and "Following analysis" in text and "Updated results" in text
    path.write_text(START)
    with pytest.raises(ValueError):
        append_section(path, "Broken")
