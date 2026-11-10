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
