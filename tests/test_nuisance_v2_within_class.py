import json

import numpy as np
import pytest

from experiments.within_class_belief_probes_2026_11.nuisance_v2 import selected_runs, validate_recipe, STUDIES
from experiments.belief_quotient_history_jax_2026_11.shared import belief_targets
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import environment


def test_selected_v2_recipes_match_environment_and_three_state_target():
    runs=selected_runs()
    assert len(runs)==6
    assert len({(r.parents[2].name,r.parent.parent.name) for r in runs})==6
    for run in runs:
        recipe=json.loads((run/"resolved_recipe.json").read_text())
        validate_recipe(recipe)
        np.testing.assert_allclose(recipe["process"]["transitions"],environment.action_transitions(recipe["variant"]),rtol=1e-6)
        np.testing.assert_allclose(recipe["process"]["emission"],environment.emission_matrix())
        points=json.loads((run/"condition_summary.json").read_text())["checkpoint_reports"]
        assert all(set(p["targets"])=={"belief"} for p in points)
    belief=np.array([[.2,.5,.3]])
    np.testing.assert_array_equal(belief_targets(belief)["belief"],belief)


@pytest.mark.parametrize("field,value",[("exploration_epsilon",.1),("temperature",1.),("entropy_schedule",[[0,.005]]),("total_env_steps",4_000_000_000)])
def test_v2_rejects_wrong_training_recipe(field,value):
    recipe=json.loads((selected_runs()[0]/"resolved_recipe.json").read_text())
    recipe["training"][field]=value
    with pytest.raises(ValueError):validate_recipe(recipe)


def test_v2_rejects_epsilon_study_mismatch_and_wrong_environment():
    recipe=json.loads((selected_runs()[0]/"resolved_recipe.json").read_text())
    recipe["study"]=STUDIES[1]
    with pytest.raises(ValueError):validate_recipe(recipe)
    recipe["study"]=STUDIES[0]
    recipe["process"]["emission"][0]=[.1875,.0625,.75]
    with pytest.raises(ValueError):validate_recipe(recipe)
