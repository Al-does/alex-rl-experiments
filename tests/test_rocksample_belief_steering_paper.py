import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.rocksample_jax_2026_10 import belief_steering_paper as paper


@pytest.mark.parametrize("signed", [False, True])
def test_joint_target_preserves_other_joint_and_normalization(signed):
    rng = np.random.default_rng(33)
    full = rng.normal(size=(3, 128)) if signed else rng.dirichlet(np.ones(128), size=3)
    full[:, -1] = 1-full[:, :-1].sum(-1)
    decoded = np.asarray(paper.full_joint(jnp.asarray(full[:, :-1])))
    bits = paper.bs.bp.configuration_bits(7)
    for rock in range(7):
        others = np.delete(bits, rock, axis=1) @ (1 << np.arange(6))
        for target in (0, 1):
            result = np.asarray(paper.joint_target(jnp.asarray(decoded), rock, target))
            np.testing.assert_allclose(result.sum(-1), 1, atol=1e-5)
            np.testing.assert_array_equal(result[:, bits[:, rock] != target], 0)
            for row in range(3):
                expected = np.bincount(others, weights=decoded[row], minlength=64)
                actual = np.bincount(others, weights=result[row], minlength=64)
                np.testing.assert_allclose(actual, expected, atol=1e-6)
                np.testing.assert_allclose(result[row] @ bits, decoded[row] @ bits +
                    np.eye(7)[rock]*(target-decoded[row] @ bits[:, rock]), atol=1e-5)


def test_joint_target_point_mass_and_omitted_coordinate():
    decoded = paper.full_joint(jnp.zeros(127))
    np.testing.assert_array_equal(decoded, np.eye(128)[127])
    for rock in range(7):
        changed = paper.joint_target(decoded, rock, 0)
        np.testing.assert_array_equal(changed, np.eye(128)[127 ^ (1 << rock)])
        np.testing.assert_array_equal(paper.joint_target(decoded, rock, 1), decoded)


def test_plain_joint_shift_and_release():
    rng = np.random.default_rng(9)
    h = jnp.asarray(rng.normal(size=16), dtype=jnp.float32)
    d = jnp.asarray(rng.normal(size=(16, 127))*.01, dtype=jnp.float32)
    b = jnp.full(127, 1/128)
    e = jnp.asarray(rng.normal(size=(127, 16)), dtype=jnp.float32)
    null = jnp.ones((7, 16))/4
    desired = paper.joint_target(paper.full_joint(h @ d+b), 2, 1)[:-1]
    expected = h + .8*((desired-(h @ d+b)) @ e)
    changed, _ = paper.shift(h, d, b, e, null, 2, 1, .8, False, True, "joint")
    np.testing.assert_allclose(changed, expected, atol=1e-6)
    unchanged, _ = paper.shift(h, d, b, e, null, 2, 1, .8, False, False, "joint")
    np.testing.assert_array_equal(unchanged, h)


def test_warmup_and_empty_fit(monkeypatch):
    valid = np.ones((2, 10), bool)
    valid[1, 9] = False
    mask = paper.retained_mask(valid)
    assert not mask[:, :8].any()
    np.testing.assert_array_equal(mask[:, 8:], [[True, True], [True, False]])
    monkeypatch.setattr(paper.bs.bp, "collect", lambda *a, **kw: dict(valid=np.ones((2, 8), bool)))
    monkeypatch.setattr(paper.bs.bp, "targets", lambda *a: SimpleNamespace())
    with pytest.raises(ValueError, match="no valid decisions"):
        paper.fit_maps(None, None, None, 2)


def test_strength_rules_include_zero_and_fallback():
    records = [dict(strength=s, metrics=dict(target=p)) for s, p in ((0., .9), (.5, .4), (1., .2))]
    assert paper.select_strength(records, "Good→Bad", .4)["strength"] == .5
    assert paper.select_strength(records, "Bad→Good", .4)["strength"] == 0
    assert not paper.select_strength(records, "Good→Bad", .1)["selection_succeeded"]
    assert paper.select_strength(records, "Good→Bad", .1)["strength"] == 1


@pytest.mark.parametrize("representation,width", [("marginal", 7), ("joint", 127)])
def test_rollout_zero_strength_matches_existing_policy(representation, width):
    env = paper.bs.rs.RockSampleParams.from_instance(episode_length=4)
    spec = paper.bs.tm.TransformerSpec(env.obs_dim, env.num_actions, d_model=8, n_heads=2, context_len=2)
    params = paper.bs.tm.init_params(spec, jax.random.key(11))
    fitted = (jnp.zeros((4, 8, width)), jnp.full((4, width), 1/128 if width == 127 else .5),
              jnp.zeros((4, width, 8)), jnp.ones((4, 7, 8))/jnp.sqrt(8))
    kwargs = dict(episodes=2, seed=12, rock=0, good=1, site=2, strength=0., control=False)
    values = paper.rollout(spec, env, params, fitted, representation=representation, **kwargs)
    marginal = (fitted[0][:, :, :7], jnp.full((4, 7), .5), fitted[2][:, :7], fitted[3])
    original = paper.bs.rollout(spec, env, params, marginal, **kwargs)
    for index in (0, 1, 2, 5, 6, 7):
        np.testing.assert_array_equal(values[index], original[index])
    random1 = paper.rollout(spec, env, params, fitted, representation=representation, random_actions=True, **kwargs)
    random2 = paper.rollout(spec, env, params, marginal, representation="marginal", random_actions=True, **kwargs)
    for index in (0, 1, 2, 5, 6, 7):
        np.testing.assert_array_equal(random1[index], random2[index])


def test_refuses_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["paper", "--output", str(tmp_path)])
    with pytest.raises(FileExistsError):
        paper.main()
