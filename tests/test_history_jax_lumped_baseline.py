import numpy as np
import pytest

from experiments.belief_quotient_history_jax_2026_11 import env
from experiments.belief_quotient_history_jax_lumped_2026_11 import shared
from experiments.belief_quotient_history_jax_lumped_2026_11.design import controller_references
from experiments.reward_factors_history_jax_lumped_2026_11.process import ACTION_PAIRS, factor_process
from experiments.reward_factors_history_jax_lumped_2026_11 import shared as factors


@pytest.mark.parametrize("variant", [1, 2, 3])
def test_distributions_and_preserved_reward_class_probabilities(variant):
    historical = np.array([[.9375, .0375, .025], [.0375, .9375, .025], [.075, .075, .85]])
    expected = np.stack([historical] + [shared.tilt_reward_state(
        historical, shared.DIRECTIONS[variant][:, a]) for a in (1, 2)])
    actual = shared.action_transitions(variant)
    for distribution in (actual, shared.emission_matrix()):
        assert np.all(distribution >= 0)
        np.testing.assert_allclose(distribution.sum(-1), 1, atol=1e-15)
    np.testing.assert_allclose(actual[:, :, 2], expected[:, :, 2], atol=1e-15)
    np.testing.assert_allclose(actual[:, :, 0], actual[:, :, 1], atol=1e-15)
    if variant == 2:
        np.testing.assert_array_equal(actual[:, 0], actual[:, 1])
    if variant == 3:
        assert actual[1, 0, 2] > actual[2, 0, 2]
        assert actual[2, 1, 2] > actual[1, 1, 2]


@pytest.mark.parametrize("condition", ["reward_both", "reward_factor_1"])
def test_factor_independence(condition):
    process = factor_process(condition)
    kernels = shared.action_transitions(3)
    for action, (first, second) in enumerate(ACTION_PAIRS):
        np.testing.assert_allclose(process.transitions[action], np.kron(kernels[first], kernels[second]), atol=3e-8)
    np.testing.assert_allclose(process.emission, np.kron(shared.emission_matrix(), shared.emission_matrix()))
    rng = np.random.default_rng(0)
    joint = np.full((3, 3), 1 / 9)
    marginals = [np.full(3, 1 / 3), np.full(3, 1 / 3)]
    for _ in range(100):
        action, token = rng.integers(9, size=2)
        joint = (joint.ravel() @ np.asarray(process.transitions[action])).reshape(3, 3)
        joint *= np.asarray(process.emission[:, token]).reshape(3, 3)
        joint /= joint.sum()
        for i, (a, x) in enumerate(zip(ACTION_PAIRS[action], divmod(token, 3))):
            marginals[i] = (marginals[i] @ kernels[a]) * shared.emission_matrix()[:, x]
            marginals[i] /= marginals[i].sum()
        np.testing.assert_allclose(joint, np.outer(*marginals), atol=1e-7)


@pytest.mark.parametrize("length", [0, 1, 300])
def test_scalar_filter_and_fixed_action_ab_invariance(length):
    process = shared.ladder_process(2)
    rng = np.random.default_rng(42)
    tokens = rng.integers(3, size=length + 1)
    swapped = np.where(tokens == 2, 2, 1 - tokens)
    actions = rng.integers(3, size=length)
    full = env.belief_reset(process, tokens[0])
    other = env.belief_reset(process, swapped[0])
    scalar = 1 / 3
    for step, token in enumerate(tokens):
        if step:
            action = actions[step - 1]
            kernel = shared.action_transitions(2)[action]
            scalar = (1 - scalar) * kernel[0, 2] + scalar * kernel[2, 2]
            full = env.belief_update(process, full, action, token)
            other = env.belief_update(process, other, action, swapped[step])
        likelihood_c, likelihood_not = (.75, .125) if token == 2 else (.125, .4375)
        scalar = scalar * likelihood_c / (scalar * likelihood_c + (1 - scalar) * likelihood_not)
        assert float(full[2]) == pytest.approx(scalar, abs=2e-7)
        assert float(other[2]) == pytest.approx(scalar, abs=2e-7)
    a, b = (np.asarray(env.belief_reset(process, token)) for token in (0, 1))
    assert a[0] > a[1] and b[1] > b[0]


def test_reference_controllers():
    values = controller_references()
    for variant, reactive, history in [(2, .2950888739, .3171048338), (3, .2668669030, .2896779494)]:
        assert values[variant]["current_token"] == pytest.approx(reactive, abs=5e-11)
        assert values[variant]["feasible_three_token"] == pytest.approx(history, abs=5e-11)


def test_invalid_variant():
    with pytest.raises(ValueError):
        shared.action_transitions(0)


@pytest.mark.parametrize("recipe", [shared, factors])
def test_lumped_recipe_budget_and_owned_kernels(recipe):
    config = recipe.train_config(False)
    assert config.total_env_steps == 2_000_000_000
    assert config.iterations == 101 and config.num_envs == 19_456
    assert config.num_minibatches == 64 and config.num_epochs == 1
    assert config.entropy_schedule == ((0, .015),)
    assert recipe.train_config(True).total_env_steps == 4096
    assert recipe.action_transitions.__module__ == recipe.__name__
    np.testing.assert_array_equal(recipe.BASELINE_TRANSITION_MATRIX[:2], [[.4875, .4875, .025]] * 2)
    assert recipe.ENVIRONMENT_REVISION == "uniform_nonreward_destination_v2"
