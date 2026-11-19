import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.rocksample_jax_2026_10 import belief_steering as bs
from experiments.rocksample_jax_2026_10 import model as tm
from experiments.rocksample_jax_2026_10 import env as rs


def test_affine_and_shift():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(100, 8))
    y = x[:, :2] * 2 + 3
    d, b = bs.affine(x, y)
    np.testing.assert_allclose(x @ d + b, y, atol=1e-10)
    e = np.linalg.pinv(d)
    null = np.zeros((2, 8)); null[:, -1] = 1
    h = jnp.asarray(x[0])
    changed, before, after = bs.shift(h, jnp.asarray(d), jnp.asarray(b),
        jnp.asarray(e), jnp.asarray(null), 0, 0., 1., False, True)
    assert abs(float(after)) < 1e-5
    control, _, _ = bs.shift(h, jnp.asarray(d), jnp.asarray(b),
        jnp.asarray(e), jnp.asarray(null), 0, 0., 1., True, True)
    np.testing.assert_allclose(np.linalg.norm(control-h), np.linalg.norm(changed-h), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(control-h) @ d, 0., atol=1e-5)
    unchanged, _, _ = bs.shift(h, jnp.asarray(d), jnp.asarray(b),
        jnp.asarray(e), jnp.asarray(null), 0, 1., 1., False, False)
    np.testing.assert_array_equal(unchanged, h)


def test_cached_sites_and_downstream_propagation():
    spec = tm.TransformerSpec(5, 3, d_model=8, n_heads=2, context_len=2)
    params = tm.init_params(spec, jax.random.key(1))
    cache = tm.empty_cache(spec)
    obs = jnp.ones(5)
    intact, original = tm.encode_step(spec, params, cache, obs)
    def identity(site, h):
        return h
    same, copied = tm.encode_step(spec, params, cache, obs, residual_transform=identity)
    np.testing.assert_array_equal(same, intact)
    np.testing.assert_array_equal(copied.k, original.k)
    def early(site, h):
        return h + jnp.arange(8) * .1 if site == 0 else h
    _, altered = tm.encode_step(spec, params, cache, obs, residual_transform=early)
    np.testing.assert_array_equal(altered.k[0], original.k[0])
    assert not np.allclose(altered.k[1], original.k[1])
    features = bs.replay(spec, params, jnp.ones((2, 4, 5)))
    assert features.shape == (2, 4, 4, 8)
    np.testing.assert_allclose(features[0, 0, -1], intact, atol=1e-6)


def test_refuses_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["belief_steering", "--output", str(tmp_path)])
    with pytest.raises(FileExistsError):
        bs.main()


def test_rollout_smoke():
    env = rs.RockSampleParams.from_instance(episode_length=4)
    spec = tm.TransformerSpec(env.obs_dim, env.num_actions, d_model=8, n_heads=2,
                              context_len=2)
    params = tm.init_params(spec, jax.random.key(11))
    fitted = (jnp.zeros((4, 8, 7)), jnp.full((4, 7), .5),
              jnp.zeros((4, 7, 8)), jnp.ones((4, 7, 8))/jnp.sqrt(8))
    values = bs.rollout(spec, env, params, fitted, episodes=2, seed=12, rock=0,
                        good=1, site=2, strength=0., control=False)
    result = bs.summarize(values, 0)
    assert result["episodes"] == 2
    assert len(result["sample_ci95"]) == 7
    np.testing.assert_allclose(result["decoded_before_after"], [.5, .5])
    assert np.isfinite(result["mean_return"])


def test_calibrated_endpoints_and_minimum_norm():
    rng = np.random.default_rng(23)
    w = rng.normal(size=(4, 16, 7)).astype(np.float32)
    encoder = rng.normal(size=(4, 7, 16)).astype(np.float32)
    bias = rng.normal(size=(4, 7)).astype(np.float32)
    null = rng.normal(size=(4, 7, 16)).astype(np.float32)
    variants, diagnostics = bs.calibrated_maps((w, bias, encoder, null))
    assert len(diagnostics) == 4
    for site in range(4):
        h = rng.normal(size=16).astype(np.float32)
        decoded = h @ w[site] + bias[site]
        for rock in range(7):
            for target in (0., 1.):
                shifts = {}
                for name in ("calibrated_encoder", "min_norm_decoder"):
                    d, b, e, n = variants[name]
                    changed, _, after = bs.shift(jnp.asarray(h), d[site], b[site], e[site],
                        n[site], rock, target, 1., False, True)
                    desired = decoded.copy(); desired[rock] = target
                    np.testing.assert_allclose(np.asarray(changed @ d[site] + b[site]),
                                               desired, atol=1e-4)
                    assert abs(float(after)-target) < 1e-4
                    shifts[name] = np.linalg.norm(np.asarray(changed)-h)
                assert shifts["min_norm_decoder"] <= shifts["calibrated_encoder"] + 1e-4


def test_calibration_rejects_rank_deficiency():
    with pytest.raises(ValueError, match="full-rank"):
        bs.calibrated_maps((np.zeros((4, 8, 7)), np.zeros((4, 7)),
                            np.zeros((4, 7, 8)), np.ones((4, 7, 8))))


def test_conditional_selectivity():
    picked = np.array([[True, True, False], [False, False, True]])
    good = np.array([[True, True, True], [False, False, False]])
    values = (picked, np.array([2, 0]), np.array([10., -10.]),
              np.array([1., 2., 0., 0., 0.]), np.array(2), good,
              np.array([True, True]), np.empty((0,)), np.array(0.), np.array(0.))
    result = bs.summarize(values, 0)
    assert result["nontarget"]["good"] == dict(picked=1, count=2, probability=.5)
    assert result["nontarget"]["bad"] == dict(picked=1, count=2, probability=.5)
    assert result["visit_without_pickup_probability"] == .5


def test_nuisance_encoder_recovers_conditional_belief_coefficients():
    rng = np.random.default_rng(31)
    beliefs = rng.normal(size=(500, 7))
    nuisance = beliefs @ rng.normal(size=(7, 5)) + rng.normal(size=(500, 5))
    belief_directions = rng.normal(size=(7, 16))
    nuisance_directions = rng.normal(size=(5, 16))
    activations = beliefs @ belief_directions + nuisance @ nuisance_directions + 2
    controlled = bs.nuisance_encoder(beliefs, nuisance, activations)
    ordinary, _ = bs.affine(beliefs, activations)
    np.testing.assert_allclose(controlled, belief_directions, atol=1e-10)
    assert np.linalg.norm(ordinary-belief_directions) > 1
    no_nuisance = bs.nuisance_encoder(beliefs, np.empty((500, 0)), activations)
    np.testing.assert_allclose(no_nuisance, ordinary, atol=1e-10)


def test_nuisance_features_use_past_samples_not_current_or_truth():
    env = rs.RockSampleParams.from_instance(episode_length=3)
    obs = np.zeros((1, 3, env.obs_dim))
    obs[:, :, 2+2*env.k] = 1
    rows = dict(valid=np.ones((1, 3), bool), obs=obs,
        position=np.broadcast_to(env.rocks[0], (1, 3, 2)),
        action=np.array([[4, 5, 4]]))
    features = bs.nuisance_features(rows, env)
    assert features.shape == (3, 24+2+12+2+7)
    np.testing.assert_array_equal(features[:, -7], [0, 1, 1])
    np.testing.assert_array_equal(features[:, -6:], 0)
    rows["valid"] = np.zeros((1, 3), bool)
    assert bs.nuisance_features(rows, env).shape == (0, features.shape[1])
