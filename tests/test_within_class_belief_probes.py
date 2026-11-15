import numpy as np
import pytest

from experiments.within_class_belief_probes_2026_11.metrics import (
    affine, bootstrap, null_basis, refit, scalar_targets, score, seed_mean_intervals,
)


def test_affine_intercept_and_held_out():
    x = np.arange(10.)[:, None]
    y = 3*x+7
    np.testing.assert_allclose(affine(x, y, x+20), 3*(x+20)+7, atol=1e-5)
    with pytest.raises(ValueError):
        affine(x[:0], y[:0], x)
    with pytest.raises(ValueError):
        affine(x, y[:-1], x)
    with pytest.raises(ValueError):
        affine(x, y, x, ridge=-1)


def test_class_centred_denominator_and_pooling():
    y = np.array([[0.], [2.], [100.], [104.]])
    c = np.array([0, 0, 1, 1])
    pred = y+1
    result = score(y, pred, c)
    assert result["sst"] == 10
    assert result["sse"] == 4
    assert result["r2"] == .6
    assert result["rmse"] == 1
    assert score(y, pred)["r2"] > .99
    assert score(y, y+100, c)["r2"] < 0
    assert score(np.ones((3, 1)), np.ones((3, 1)))["r2"] is None
    assert score(np.empty((0, 3)), np.empty((0, 3)))["rmse"] is None
    with pytest.raises(ValueError):
        score(y, pred[:2])
    with pytest.raises(ValueError):
        score(y, pred, c[:2])


def test_refit_separate_classes_and_missing_fit_class():
    x = np.arange(12.)[:, None]
    c = np.repeat([0, 1], 6)
    y = x + 100*c[:, None]
    np.testing.assert_allclose(refit(x, y, x, c, c), y, atol=1e-5)
    assert np.isnan(refit(x, y, x[:1], c, np.array([2]))).all()
    assert refit(x, y, x[:0], c, c[:0]).shape == (0, 1)
    p1 = refit(x, y, x, c, c, permute_seed=1)
    np.testing.assert_array_equal(p1, refit(x, y, x, c, c, permute_seed=1))
    assert not np.allclose(p1, y)


def test_head_blind_ignores_common_logit_shift():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(3, 64))
    basis = null_basis(w)
    assert basis.shape == (64, 62)
    np.testing.assert_allclose((w[:2]-w[2])@basis, 0, atol=1e-12)
    np.testing.assert_allclose(basis.T@basis, np.eye(62), atol=1e-12)
    shifted = null_basis(w+rng.normal(size=(1, 64)))
    np.testing.assert_allclose(basis@basis.T, shifted@shifted.T, atol=1e-12)
    with pytest.raises(ValueError):
        null_basis(np.zeros((3, 64)))
    with pytest.raises(ValueError):
        null_basis(np.zeros((0, 64)))


def test_scalars_coordinate_order():
    b = np.array([[.8, .15, .05], [.1, .7, .2], [.15, .05, .8]])
    margin, split = scalar_targets(b)
    np.testing.assert_allclose(margin[:, 0], [.65, .5, .65])
    np.testing.assert_allclose(split[:, 0], [.75, 1/3, .75])
    assert scalar_targets(b[:0])[0].shape == (0, 1)
    with pytest.raises(ValueError):
        scalar_targets(np.ones((4, 2)))


def test_cluster_bootstrap_matches_direct_episode_resampling():
    y = np.arange(24.).reshape(12, 2)
    pred = y+np.arange(12)[:, None]/10
    c = np.tile([0, 1], 6)
    ep = np.repeat(np.arange(3), 4)
    result = bootstrap(y, pred, c, ep, n=100, seed=7)
    rng = np.random.default_rng(7)
    weights = rng.multinomial(3, np.full(3, 1/3), size=100)
    scores = []
    for weight in weights:
        ix = np.concatenate([np.tile(np.flatnonzero(ep == k), count) for k, count in enumerate(weight)])
        scores.append(score(y[ix], pred[ix], c[ix])["r2"])
    np.testing.assert_allclose(result["r2_ci95"], np.quantile(scores, [.025, .975]))
    np.testing.assert_allclose(result["r2_se"], np.std(scores, ddof=1))
    assert result["n_episodes"] == 3
    with pytest.raises(ValueError):
        bootstrap(y[:0], pred[:0], c[:0], ep[:0])


def test_training_seed_bootstrap_preserves_paired_columns():
    values = np.array([[1., 2.], [2., 4.], [4., 8.]])
    result = seed_mean_intervals(values, n=100, seed=7)
    draws = np.random.default_rng(7).integers(3, size=(100, 3))
    expected = values[draws].mean(axis=1)
    np.testing.assert_allclose(result["mean"], values.mean(0))
    np.testing.assert_allclose(result["ci95"], np.quantile(expected, [.025, .975], axis=0).T)
    assert result["n_seeds"] == 3 and result["n_resamples"] == 100
    difference = seed_mean_intervals(values[:, 1:]-2*values[:, :1])
    assert difference["ci95"] == [[0., 0.]]
    for invalid in (values[:0], values[:1], np.array([[1., np.nan], [2., 3.]]), values[:, 0]):
        with pytest.raises(ValueError):
            seed_mean_intervals(invalid)


def test_episode_bootstrap_undefined_and_misaligned_se():
    y = np.ones((6, 1))
    classes = np.zeros(6, dtype=int)
    episodes = np.repeat(np.arange(3), 2)
    result = bootstrap(y, y, classes, episodes, n=10)
    assert result["r2_se"] is None
    assert result["valid_r2_resamples"] == 0
    with pytest.raises(ValueError):
        bootstrap(y, y[:-1], classes, episodes)
    with pytest.raises(ValueError):
        bootstrap(y, y, classes[:-1], episodes)


@pytest.mark.parametrize("width", [3, 92, 128])
def test_head_blind_supports_recurrent_widths(width):
    w = np.random.default_rng(width).normal(size=(3, width))
    basis = null_basis(w)
    assert basis.shape == (width, width-2)
    np.testing.assert_allclose((w[:2]-w[2]) @ basis, 0, atol=1e-12)
    np.testing.assert_allclose(basis.T @ basis, np.eye(width-2), atol=1e-12)


def test_paired_episode_bootstrap_matches_direct_resampling():
    y = np.arange(24.).reshape(12, 2)
    pred, other = y+.1, y+np.arange(12)[:, None]/10
    classes = np.tile([0, 1], 6)
    episodes = np.repeat(np.arange(3), 4)
    result = bootstrap(y, pred, classes, episodes, n=100, seed=7, comparison=other)
    weights = np.random.default_rng(7).multinomial(3, np.full(3, 1/3), size=100)
    differences = []
    for weight in weights:
        ix = np.concatenate([np.tile(np.flatnonzero(episodes == k), count) for k, count in enumerate(weight)])
        differences.append(score(y[ix], pred[ix], classes[ix])["r2"]-score(y[ix], other[ix], classes[ix])["r2"])
    np.testing.assert_allclose(result["paired_difference_ci95"], np.quantile(differences, [.025, .975]))
    np.testing.assert_allclose(result["paired_difference_se"], np.std(differences, ddof=1))
    same = bootstrap(y, pred, classes, episodes, n=100, comparison=pred)
    assert same["paired_difference_ci95"] == [0., 0.]
    constant = bootstrap(np.ones_like(y), pred, classes, episodes, n=10, comparison=other)
    assert constant["paired_difference_ci95"] is None
    with pytest.raises(ValueError):
        bootstrap(y[:0], pred[:0], classes[:0], episodes[:0], comparison=other[:0])
    with pytest.raises(ValueError):
        bootstrap(y, pred, classes, episodes, comparison=other[:-1])
