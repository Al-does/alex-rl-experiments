import copy
import json

import pytest

from experiments.within_class_exploration_sweep_2026_11 import analyze

PR219_RUN = next((analyze.EXPERIMENTS/"belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11").glob("variant_3_transformer/results/*"))


def recipe():
    return json.loads((PR219_RUN/"resolved_recipe.json").read_text())


def sweep_recipe(arm, variant=3):
    r = copy.deepcopy(recipe())
    settings = analyze.ARMS[arm]
    r["study"], r["condition"] = analyze.SWEEP, f"variant_{variant}"
    r["training"]["exploration_epsilon"] = settings["exploration_epsilon"]
    r["training"]["entropy_schedule"] = [[0, settings["entropy_coeff"]]]
    return r, f"variant_{variant}_transformer_{arm}"


def test_pr219_baselines_selected_for_both_conditions():
    runs = [r for r in analyze.selected_runs() if r.parents[2].name in analyze.PR219]
    assert sorted((r.parents[2].name, r.parent.parent.name) for r in runs) == sorted(
        (s, f"variant_{v}_transformer") for s in analyze.PR219 for v in analyze.VARIANTS)
    for run in runs:
        analyze.validate_recipe(json.loads((run/"resolved_recipe.json").read_text()), run.parent.parent.name)


@pytest.mark.parametrize("arm", sorted(analyze.ARMS))
def test_sweep_recipes_validate_against_their_leaf(arm):
    r, leaf = sweep_recipe(arm)
    analyze.validate_recipe(r, leaf)
    other = next(a for a in analyze.ARMS if a != arm)
    with pytest.raises(ValueError):
        analyze.validate_recipe(r, f"variant_3_transformer_{other}")


@pytest.mark.parametrize("field,value", [("temperature", 1.0), ("total_env_steps", 10), ("entropy_schedule", [[0, .002], [10, 0]])])
def test_validator_rejects_other_training_settings(field, value):
    r, leaf = sweep_recipe("ent002")
    r["training"][field] = value
    with pytest.raises(ValueError):
        analyze.validate_recipe(r, leaf)


def test_validator_rejects_trivial_quotient():
    r = recipe()
    r["condition"] = "variant_1"
    with pytest.raises(ValueError):
        analyze.validate_recipe(r, "variant_1_transformer")
