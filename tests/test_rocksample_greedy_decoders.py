from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.rocksample_jax_2026_10 import belief_steering_greedy_decoders as gd
from experiments.rocksample_jax_2026_10 import belief_steering_greedy_report as report


def test_online_nuisance_matches_fit_history_and_empty_batch():
    env = gd.bs.rs.RockSampleParams.from_instance()
    spec = gd.bs.tm.TransformerSpec(env.obs_dim, env.num_actions, d_model=8,
                                  n_heads=2, context_len=2)
    params = gd.bs.tm.init_params(spec, jax.random.key(1))
    rows = gd.bs.bp.collect(spec, env, params, params, episodes=3, key_seed=4, scripted=True)
    sampled = np.zeros((3, env.k), bool)
    previous = np.full(3, env.num_actions)
    features = []
    for t in range(env.episode_length):
        features.append(np.asarray(jax.vmap(lambda p, a, o, s: gd.online_nuisance(
            env, p, t, a, o, s))(rows['position'][:, t], previous, rows['obs'][:, t], sampled)))
        sampled |= ((rows['action'][:, t] == 4)[:, None] & rows['valid'][:, t, None] &
            np.all(rows['position'][:, t, None] == np.asarray(env.rocks), axis=-1))
        previous = rows['action'][:, t]
    actual = np.stack(features, axis=1)[rows['valid']]
    np.testing.assert_array_equal(actual, gd.bs.nuisance_features(rows, env))
    empty = gd.online_nuisance(env, jnp.zeros((0, 2), int), jnp.zeros(0, int),
        jnp.zeros(0, int), jnp.zeros((0, env.obs_dim)), jnp.zeros((0, env.k)))
    assert empty.shape == (0, 144)


def test_nuisance_aware_endpoint_plain_diagnostic_and_release():
    rng = np.random.default_rng(8)
    d = rng.normal(size=(16, 7))
    e = np.linalg.pinv(d)
    g = rng.normal(size=(5, 7))
    b = rng.normal(size=7)
    plain = rng.normal(size=(16, 7))
    null = gd.null_directions(d, rng, 7)
    fitted = tuple(map(jnp.asarray, (d, b, e, null, d, g, plain, b, g, b)))
    h, z = jnp.asarray(rng.normal(size=16)), jnp.asarray(rng.normal(size=5))
    for rock in range(7):
        for target in (0, 1):
            after, diag = gd.decoder_shift(h, z, fitted, rock, target, False, True, 1.)
            before = h@d+z@g+b
            expected = np.asarray(before).copy(); expected[rock] = target
            np.testing.assert_allclose(after@d+z@g+b, expected, atol=1e-5)
            assert float(diag[5]) < 1e-5
            np.testing.assert_allclose(diag[6], abs((after@plain+b)[rock]-target), atol=1e-5)
            released, _ = gd.decoder_shift(h, z, fitted, rock, target, False, False, 1.)
            np.testing.assert_array_equal(released, h)
            control, _ = gd.decoder_shift(h, z, fitted, rock, target, True, True, 1.)
            np.testing.assert_allclose((control-h)@d, 0, atol=1e-5)
            np.testing.assert_allclose(np.linalg.norm(control-h), np.linalg.norm(after-h), rtol=1e-5)


def test_greedy_rollout_deterministic_env_keys_and_sampled_zero_strength():
    env = gd.bs.rs.RockSampleParams.from_instance(episode_length=5)
    spec = gd.bs.tm.TransformerSpec(env.obs_dim, env.num_actions, d_model=8,
                                  n_heads=2, context_len=2)
    params = gd.bs.tm.init_params(spec, jax.random.key(6))
    d = jnp.zeros((4, 8, 7)); b = jnp.full((4, 7), .5)
    e = jnp.zeros((4, 7, 8)); null = jnp.ones_like(e)/jnp.sqrt(8)
    g = jnp.zeros((4, 24+4+12+2+7, 7))
    fitted = (d, b, e, null, d, g, d, b, g, b)
    kw = dict(episodes=3, seed=4, rock=0, good=1, site=0, strength=0.)
    one = gd.rollout(spec, env, params, fitted, greedy=True, **kw)
    two = gd.rollout(spec, env, params, fitted, greedy=True, **kw)
    for a, c in zip(jax.tree.leaves(one), jax.tree.leaves(two)):
        np.testing.assert_array_equal(a, c)
    sampled = gd.rollout(spec, env, params, fitted, **kw)[0]
    original = gd.bs.rollout(spec, env, params, fitted[:4], control=False, **kw)
    for i in (0, 1, 2, 5, 6, 7):
        np.testing.assert_array_equal(sampled[i], original[i])


def test_zero_variance_r2_and_no_null_direction():
    assert gd.score(np.ones((3, 2)), np.zeros((3, 2))) == [None, None]
    with pytest.raises(ValueError, match='no usable'):
        gd.null_directions(np.eye(7), np.random.default_rng(1), 7)


def test_paper_zero_variance_scores_are_null(monkeypatch):
    features = np.random.default_rng(2).normal(size=(1, 10, 4, 8))
    marginal = np.zeros((10, 7))
    marginal[:, 0] = np.linspace(.2, .8, 10)
    joint = np.zeros((10, 128))
    joint[:, 0], joint[:, -1] = marginal[:, 0], 1-marginal[:, 0]
    rows = dict(valid=np.ones((1, 10), bool), obs=features)
    monkeypatch.setattr(gd.bs.bp, 'collect', lambda *a, **kw: rows)
    monkeypatch.setattr(gd.bs.bp, 'targets', lambda *a: SimpleNamespace(
        marginals=marginal, joint=joint))
    monkeypatch.setattr(gd.bs, 'replay', lambda *a: features)
    _, scores, _ = gd.paper.fit_maps(SimpleNamespace(n_layers=3, d_model=8),
        SimpleNamespace(k=7), None, 1, greedy=True)
    for site in range(4):
        assert scores['marginal'][site][1:] == [None]*6
        assert scores['marginal'][site][0] == pytest.approx(1.)
        assert scores['joint'][site] == pytest.approx(1.)


def test_interventions_require_completed_step_zero(tmp_path):
    gd.cont.write(tmp_path/'sanity.json', dict(checkpoints=[]))
    with pytest.raises(ValueError, match='complete step-0'):
        gd.run(tmp_path)


def test_selective_success_uses_unrounded_boundaries():
    metrics = {d: dict(steered=dict(target=t, nontarget_good=.9))
               for d, t in zip(gd.paper.DIRECTIONS, (.1, .5))}
    assert report.selective(metrics)
    metrics['Bad→Good']['steered']['target'] = .4998605
    assert not report.selective(metrics)
    metrics['Bad→Good']['steered']['target'] = .5
    metrics['Good→Bad']['steered']['nontarget_good'] = .899999
    assert not report.selective(metrics)
