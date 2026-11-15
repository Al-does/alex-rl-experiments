import numpy as np
import pytest

from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import environment as v2
from experiments.within_class_drivers_2026_11 import theory


def test_family_endpoints_reproduce_the_v2_quotients():
    np.testing.assert_array_equal(theory.kernels_for(1.), v2.action_transitions(2))
    np.testing.assert_array_equal(theory.kernels_for(-1.), v2.action_transitions(3))
    np.testing.assert_allclose(theory.emission_for(0.), v2.emission_matrix())


def test_residual_share_separates_explained_from_unexplained_variance():
    rng = np.random.default_rng(0)
    classes = rng.integers(2, size=4000)
    code = rng.integers(5, size=4000)
    noise = rng.normal(size=4000)
    assert theory.residual_share(code.astype(float), classes, code) == pytest.approx(0, abs=1e-12)
    assert theory.residual_share(noise, classes, code) > .99
    assert theory.residual_share(np.ones(4000), classes, code) is None


def test_lumped_filter_is_exact_only_under_the_reward_state_symmetry():
    emission = theory.emission_for()
    for lam, exact in ((1., True), (-1., False)):
        kernels = theory.kernels_for(lam)
        q = theory.solve(kernels, emission, n=40, gamma=.98)
        optimal = theory.simulate(kernels, emission, lambda b: q(b).argmax(1), episodes=64, keep=False)
        lumped = theory.lumped_filter_reward(kernels, emission, q, episodes=64)
        if exact:
            assert lumped["reward"] == pytest.approx(optimal["reward"], abs=1e-12)
        else:
            assert lumped["reward"] < optimal["reward"] - .05


def test_probe_helpers():
    pytest.importorskip("jax")
    from experiments.within_class_drivers_2026_11 import probe
    values = np.array([[0., 2., 5.], [1., 1., 0.]])
    np.testing.assert_allclose(probe.margin(values), [3., 0.])
    x = np.arange(40.)
    classes = np.repeat([0, 1], 20)
    y = np.where(classes == 0, x, -x)
    assert probe.rank_correlation(x, y, classes) == pytest.approx(0)
    assert probe.rank_correlation(x, x ** 3, classes) == pytest.approx(1)
    assert probe.rank_correlation(x, np.ones(40), classes) is None
    target = x[:, None]
    bins = probe.boundary_bins(target, target + 1, target + 2, x, classes)
    assert [b["mse_trained"] for b in bins] == [1.] * 3 and [b["mse_matched_init"] for b in bins] == [4.] * 3
    assert bins[0]["mean_q_margin"] < bins[1]["mean_q_margin"] < bins[2]["mean_q_margin"]
