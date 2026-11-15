import importlib
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pytest

from experiments.belief_quotient_history_jax_2026_11.continuation import locate_parent_run
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import shared as parent
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_cont4b_2026_11 import shared as cont


@pytest.mark.parametrize("variant", (1, 2, 3))
def test_continuation_matches_parent_recipe(variant):
    leaf = f"variant_{variant}_transformer"
    module = importlib.import_module(f"experiments.{cont.STUDY}.{leaf}.experiment")
    parent_module = importlib.import_module(f"experiments.{parent.STUDY}.{leaf}.experiment")
    restored = {"params": object(), "step_offset": 2_012_217_344, "parent": {"run_id": "p"}}
    context = type("Context", (), {"smoke": False, "seed": 42})()
    with patch.object(cont.base, "run_study", side_effect=lambda *a, **kw: kw), \
            patch.object(cont, "load_parent_checkpoint", return_value=restored) as load:
        result = module.run(context)
        reference = parent_module.run(context)
    assert load.call_args.kwargs["parent_leaf_dir"] == cont.PARENT_ROOT / leaf
    assert load.call_args.kwargs["study"] == parent.STUDY
    assert result["continue_from"] is restored and result["study"] == cont.STUDY
    assert asdict(result["config"]) == asdict(reference["config"])
    assert result["config"].total_env_steps == parent.TOTAL_ENV_STEPS
    assert result["spec"] == reference["spec"]
    for key in ("initial", "transitions", "emission", "reward", "reward_components"):
        np.testing.assert_array_equal(getattr(result["process"], key),
                                      getattr(reference["process"], key))


@pytest.mark.parametrize("variant", (1, 2, 3))
def test_parent_runs_are_checked_in(variant):
    found = locate_parent_run(cont.PARENT_ROOT / f"variant_{variant}_transformer",
                              study=parent.STUDY, seed=42)
    assert found["final_step"] == 2_012_217_344 and found["b2_prefix"]
