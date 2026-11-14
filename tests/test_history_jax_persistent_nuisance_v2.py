from dataclasses import asdict
import importlib
from unittest.mock import patch

import numpy as np
import pytest

from experiments.belief_quotient_persistent_nuisance_jax_2026_11 import shared as old
from experiments.belief_quotient_persistent_nuisance_jax_temp15_2026_11 import shared as temp
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import environment as env
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import shared as ladder
from experiments.reward_factors_persistent_nuisance_v2_jax_temp15_2026_11 import shared as factors


def test_design():
    np.testing.assert_array_equal(env.BASELINE_TRANSITION_MATRIX,
        [[.995505, .002495, .002], [.002495, .995505, .002], [.005, .005, .99]])
    np.testing.assert_array_equal(env.EMISSION_MATRIX,
        [[.1125, .0375, .85], [.0375, .1125, .85], [0, 0, 1]])
    assert env.EFFECT_SIZE == 2.5
    assert env.REWARD_STATE == 2
    for variant in (1, 2, 3):
        process = env.ladder_process(variant)
        np.testing.assert_allclose(process.initial, [1/3]*3)
        np.testing.assert_array_equal(process.reward, [0, 0, 1])
        np.testing.assert_allclose(process.transitions[0], env.BASELINE_TRANSITION_MATRIX)
        for action in (1, 2):
            expected = env.BASELINE_TRANSITION_MATRIX.copy()
            expected[:, 2] *= np.exp(2.5 * old.DIRECTIONS[variant][:, action])
            expected /= expected.sum(1, keepdims=True)
            np.testing.assert_allclose(process.transitions[action], expected, atol=1e-7)
    with pytest.raises(ValueError):
        env.ladder_process(0)
    for token in range(3):
        from experiments.belief_quotient_history_jax_2026_11.env import belief_reset
        assert np.isfinite(belief_reset(env.ladder_process(2), token)).all()


ARCHITECTURES = ("transformer", "memoryless", "gru128", "gru92")
CASES = [(ladder, f"variant_{v}_{a}") for v in (1, 2, 3) for a in ARCHITECTURES] + [
    (factors, f"{c}_{a}") for c in ("reward_both", "reward_factor_1")
    for a in ARCHITECTURES
]


@pytest.mark.parametrize("study,leaf", CASES)
def test_leaves(study, leaf):
    module = importlib.import_module(f"experiments.{study.STUDY}.{leaf}.experiment")
    for smoke in (False, True):
        with patch.object(study.base, "run_study", side_effect=lambda *a, **kw: kw):
            result = module.run(type("Context", (), {"smoke": smoke})())
        expected = study.base.train_config(smoke, temp.TOTAL_ENV_STEPS)
        config = asdict(expected)
        config.update(entropy_schedule=((0, 0.0),), temperature=1.5)
        assert asdict(result["config"]) == config
        assert study.TOTAL_ENV_STEPS == 2_000_000_000
        assert result["study"] == study.STUDY
        assert result["recipe_extra"]["environment_revision"] == "persistent_nuisance_v2"
        process = result["process"]
        assert result["spec"] == study.base.model_spec(result["architecture"], process)
        if study is ladder:
            np.testing.assert_allclose(process.emission, env.EMISSION_MATRIX)
        else:
            np.testing.assert_allclose(process.initial, np.full(9, 1/9))
            np.testing.assert_allclose(process.emission, np.kron(env.EMISSION_MATRIX, env.EMISSION_MATRIX))
            kernels = env.action_transitions(3)
            for i, (a, b) in enumerate(study.base.ACTION_PAIRS):
                np.testing.assert_allclose(process.transitions[i], np.kron(kernels[a], kernels[b]), atol=1e-7)
    with pytest.raises(ValueError):
        study.run_condition(type("Context", (), {"smoke": True})(),
                            architecture="invalid",
                            **({"variant": 1} if study is ladder else {"condition": "reward_both"}))


def test_original_defaults():
    assert old.EFFECT_SIZE == 1.5
    np.testing.assert_array_equal(old.EMISSION_MATRIX,
        [[.1875, .0625, .75], [.0625, .1875, .75], [0, 0, 1]])
    np.testing.assert_allclose(old.BASELINE_TRANSITION_MATRIX,
        [[.9375, .0375, .025], [.0375, .9375, .025], [.075, .075, .85]])
    for base in (old, factors.base):
        assert base.train_config(False).total_env_steps == 200_000_000
        assert base.train_config(False).temperature == 1.
    with patch.object(old, "run_study", side_effect=lambda *a, **kw: kw):
        result = temp.run_condition(type("Context", (), {"smoke": False})(),
                                    variant=2, architecture="transformer")
    np.testing.assert_allclose(result["process"].emission, old.EMISSION_MATRIX)
    assert result["config"].temperature == 1.5
    assert result["config"].total_env_steps == 2_000_000_000
