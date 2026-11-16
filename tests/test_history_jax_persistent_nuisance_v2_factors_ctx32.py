import importlib
from dataclasses import asdict, replace
from unittest.mock import patch

import numpy as np
import pytest

from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import environment as env
from experiments.reward_factors_persistent_nuisance_jax_2026_11 import shared as base
from experiments.reward_factors_persistent_nuisance_v2_jax_temp15_2026_11 import shared as ctx10
from experiments.reward_factors_persistent_nuisance_v2_jax_temp15_ctx32_2026_11 import shared as ctx32
from experiments.reward_factors_persistent_nuisance_v2_jax_temp15_ctx32_eps05_2026_11 import shared as eps05

CASES = [(study, eps, condition) for study, eps in ((ctx32, 0.0), (eps05, 0.05))
         for condition in ("reward_both", "reward_factor_1")]


@pytest.mark.parametrize("study,eps,condition", CASES)
def test_leaves(study, eps, condition):
    module = importlib.import_module(f"experiments.{study.STUDY}.{condition}_transformer.experiment")
    for smoke in (False, True):
        with patch.object(base, "run_study", side_effect=lambda *a, **kw: kw):
            result = module.run(type("Context", (), {"smoke": smoke})())
            reference = ctx10.run_condition(type("Context", (), {"smoke": smoke})(),
                                            condition=condition, architecture="transformer")
        assert result["study"] == study.STUDY and result["condition"] == condition
        assert asdict(result["config"]) == {**asdict(reference["config"]), "exploration_epsilon": eps}
        assert reference["config"].exploration_epsilon == 0.0
        assert result["spec"] == replace(reference["spec"], context_len=32)
        assert reference["spec"].context_len == 10 and result["spec"].kind == "transformer"
        assert result["recipe_extra"] == reference["recipe_extra"]
        np.testing.assert_allclose(result["process"].emission, np.kron(env.EMISSION_MATRIX, env.EMISSION_MATRIX))
        for key in ("initial", "transitions", "reward"):
            np.testing.assert_array_equal(getattr(result["process"], key), getattr(reference["process"], key))
    with pytest.raises(ValueError):
        study.run_condition(type("Context", (), {"smoke": True})(), condition=condition, architecture="gru128")


def test_recipe_constants():
    assert ctx32.TOTAL_ENV_STEPS == eps05.TOTAL_ENV_STEPS == 2_000_000_000
    assert ctx32.ENTROPY_SCHEDULE == ((0, 0.0),) and ctx32.TEMPERATURE == 1.5
    assert ctx32.CONTEXT_LEN == eps05.CONTEXT_LEN == 32
    assert (ctx32.EXPLORATION_EPSILON, eps05.EXPLORATION_EPSILON) == (0.0, 0.05)
