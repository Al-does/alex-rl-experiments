import importlib
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pytest

from experiments.belief_quotient_history_jax_2026_11.env import belief_reset, belief_update
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment as env
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import shared as ladder
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import shared as original_ladder
from experiments.reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11 import shared as factors
from experiments.reward_factors_persistent_nuisance_v2_jax_temp15_ctx32_2026_11 import shared as original_factors


def test_design_and_filter():
    np.testing.assert_array_equal(env.BASELINE_TRANSITION_MATRIX,
        [[.995505, .002495, .002], [.002495, .995505, .002], [.005, .005, .99]])
    np.testing.assert_array_equal(env.EMISSION_MATRIX,
        [[.1125, .0375, .85], [.0375, .1125, .85], [.01, .01, .98]])
    assert env.EMISSION_EPSILON == .02
    assert env.EFFECT_SIZE == 2.5 and env.REWARD_STATE == 2
    for variant in (1, 2, 3):
        process = env.ladder_process(variant)
        np.testing.assert_allclose(process.initial, [1/3]*3)
        np.testing.assert_array_equal(process.reward, [0, 0, 1])
        for action in range(3):
            expected = env.BASELINE_TRANSITION_MATRIX.copy()
            expected[:, 2] *= np.exp(2.5 * env.DIRECTIONS[variant][:, action])
            expected /= expected.sum(1, keepdims=True)
            np.testing.assert_allclose(process.transitions[action], expected, atol=1e-7)
            for token in range(3):
                prior = np.full(3, 1/3) * env.EMISSION_MATRIX[:, token]
                prior /= prior.sum()
                np.testing.assert_allclose(belief_reset(process, token), prior, atol=1e-7)
                posterior = (prior @ expected) * env.EMISSION_MATRIX[:, token]
                posterior /= posterior.sum()
                np.testing.assert_allclose(belief_update(process, prior, action, token),
                                           posterior, atol=1e-7)
    with pytest.raises(ValueError):
        env.ladder_process(0)


CASES = [(ladder, original_ladder, f"variant_{v}_{a}") for v in (1, 2, 3)
         for a in ("transformer", "memoryless", "gru128", "gru92")] + [
    (factors, original_factors, f"{c}_transformer")
    for c in ("reward_both", "reward_factor_1")]


@pytest.mark.parametrize("study,original,leaf", CASES)
def test_leaves(study, original, leaf):
    module = importlib.import_module(f"experiments.{study.STUDY}.{leaf}.experiment")
    old_module = importlib.import_module(f"experiments.{original.STUDY}.{leaf}.experiment")
    for smoke in (False, True):
        context = type("Context", (), {"smoke": smoke})()
        with patch.object(study.base, "run_study", side_effect=lambda *a, **kw: kw):
            result = module.run(context)
            reference = old_module.run(context)
        assert asdict(result["config"]) == asdict(reference["config"])
        assert result["config"].exploration_epsilon == 0.
        assert result["spec"] == reference["spec"]
        assert result["study"] == study.STUDY
        assert result["recipe_extra"]["environment_revision"] == "persistent_nuisance_eps02"
        for key in ("initial", "transitions", "reward", "reward_components"):
            np.testing.assert_array_equal(getattr(result["process"], key),
                                          getattr(reference["process"], key))
        expected_emission = env.EMISSION_MATRIX if study is ladder else np.kron(
            env.EMISSION_MATRIX, env.EMISSION_MATRIX)
        np.testing.assert_allclose(result["process"].emission, expected_emission)
        with pytest.raises(ValueError):
            study.run_condition(context, architecture="invalid",
                **({"variant": 1} if study is ladder else {"condition": "reward_both"}))
