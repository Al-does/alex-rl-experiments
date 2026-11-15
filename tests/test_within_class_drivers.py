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


def test_trivial_quotient_is_reported_outside_the_interpolation_family():
    from experiments.within_class_drivers_2026_11 import probe, report

    np.testing.assert_array_equal(probe.family(None), v2.action_transitions(1))
    np.testing.assert_array_equal(probe.family(-1.), theory.kernels_for(-1.))
    assert report.arm_key(None, 0.) == "trivial"
    arms = [(-1., 0.), (1., .05), (None, 0.), (1., 0.)]
    assert sorted(arms, key=lambda a: report.position(*a)) == [(None, 0.), (1., 0.), (-1., 0.), (1., .05)]
    assert report.spearman([1, 2, 3, 4], [10, 30, 20, 40]) == pytest.approx(.8)


def test_early_rise_helpers_align_lags_and_score_within_class():
    from experiments.within_class_drivers_2026_11 import early_rise

    obs = np.arange(2 * 70 * 1, dtype=float).reshape(2, 70, 1)
    now, past = early_rise.lagged(obs, 0, 12), early_rise.lagged(obs, 3, 12)
    assert now.shape == (12, 1) and np.allclose(now - past, 3)
    assert np.allclose(now, early_rise.flat(obs, 12))
    rng = np.random.default_rng(0)
    x = [rng.normal(size=(400, 3)) for _ in range(2)]
    classes = [np.arange(400) % 2 for _ in range(2)]
    y = [np.stack([v[:, 0] * np.where(c, 2., -1.) + c, rng.normal(size=400)], axis=1) for v, c in zip(x, classes)]
    exact, noise = early_rise.r2(x[0], y[0], x[1], y[1], classes)
    assert exact == pytest.approx(1., abs=1e-6) and noise < .05


def test_truncated_filter_matches_exact_filter_and_forgets_older_tokens():
    from experiments.within_class_drivers_2026_11 import window_filter
    from experiments.within_class_drivers_2026_11.theory import emission_for, kernels_for

    transitions, emission = np.asarray(kernels_for(-1.)), np.asarray(emission_for(0.))
    initial, prior = np.full(3, 1 / 3), np.array([.2, .3, .5])
    rng = np.random.default_rng(0)
    token, action = rng.integers(0, 2, size=(2, 40)) * 2, rng.integers(0, 3, size=(2, 40))
    token[:, ::3] = rng.integers(0, 2, size=token[:, ::3].shape)
    obs = np.concatenate([np.eye(3)[token], np.eye(3)[action]], axis=-1)

    def exact(tokens, actions, start, reset):
        belief = start * emission[:, tokens[0]] if reset else start @ transitions[actions[0]] * emission[:, tokens[0]]
        for t, a in zip(tokens[1:], actions[1:]):
            belief = (belief / belief.sum()) @ transitions[a] * emission[:, t]
        return belief / belief.sum()

    full = window_filter.truncated_filter(obs, transitions, emission, 40, prior, initial)
    short = window_filter.truncated_filter(obs, transitions, emission, 5, prior, initial)
    for e in range(2):
        for t in (0, 3, 39):
            assert np.allclose(full[e, t], exact(token[e, :t + 1], action[e, :t + 1], initial, True))
        assert np.allclose(short[e, 4], full[e, 4])
        assert np.allclose(short[e, 20], exact(token[e, 16:21], action[e, 16:21], prior, False))
    changed = obs.copy()
    changed[:, 10] = obs[:, 10][:, [1, 0, 2, 3, 4, 5]]
    assert np.allclose(window_filter.truncated_filter(changed, transitions, emission, 5, prior, initial)[:, 15:],
                       short[:, 15:])
