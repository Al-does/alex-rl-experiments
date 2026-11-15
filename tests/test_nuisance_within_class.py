import copy
import json

import numpy as np
import pytest

from experiments.within_class_belief_probes_2026_11.nuisance import selected_runs, validate_recipe
from experiments.within_class_belief_probes_2026_11.temp15_analyze import TARGETS


def test_selected_nuisance_recipes_and_full_three_state_targets():
    runs = selected_runs()
    assert len(runs) == 3
    for run in runs:
        recipe = json.loads((run/"resolved_recipe.json").read_text())
        validate_recipe(recipe)
        assert np.asarray(recipe["process"]["transitions"]).shape == (3, 3, 3)
        assert np.asarray(recipe["process"]["emission"]).shape == (3, 3)
        reports = json.loads((run/"condition_summary.json").read_text())["checkpoint_reports"]
        assert all(set(r["targets"]) == {"belief"} for r in reports)
    belief = np.array([[.2, .5, .3], [.4, .1, .5]])
    targets = np.concatenate([belief, belief[:, :1]-belief[:, 1:2], belief[:, 2:3]], axis=1)
    np.testing.assert_array_equal(targets[:, TARGETS["belief"]], belief)
    np.testing.assert_allclose(targets[:, TARGETS["delta"]].ravel(), [-.3, .3])
    np.testing.assert_array_equal(targets[:, TARGETS["rho"]].ravel(), [.3, .5])


@pytest.mark.parametrize("field,value", [("epsilon", .1), ("temperature", 1.5), ("entropy_schedule", [[0, .01]])])
def test_nuisance_recipe_rejects_other_exploration_arms(field, value):
    recipe = json.loads((selected_runs()[0]/"resolved_recipe.json").read_text())
    changed = copy.deepcopy(recipe)
    changed["training"][field] = value
    with pytest.raises(ValueError):
        validate_recipe(changed)


def test_nuisance_recipe_rejects_added_latent():
    recipe = json.loads((selected_runs()[0]/"resolved_recipe.json").read_text())
    recipe["process"]["initial"] = [1/6]*6
    with pytest.raises(ValueError):
        validate_recipe(recipe)
