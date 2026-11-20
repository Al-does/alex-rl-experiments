import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.belief_quotient_causal_decoding_2026_11 import decoders as d


def test_affine_band_and_errors():
    x = np.arange(30.).reshape(10, 3)
    np.testing.assert_allclose(d.affine(x, x[:, :2])(x), x[:, :2], atol=1e-5)
    history = jnp.arange(12).reshape(1, 4, 3)
    band = d.band_features(history, (0, 1))
    np.testing.assert_array_equal(band[0, 2], jnp.r_[history[0, 2], history[0, 1]])
    with pytest.raises(ValueError):
        d.affine(np.empty((0, 3)), np.empty((0, 2)))
    with pytest.raises(ValueError):
        d.affine(np.full((2, 3), np.nan), np.ones((2, 2)))


def test_null_and_donor_abstention():
    h = jnp.ones((2, 3))
    direction = jnp.array([[1., 2., 3.], [4., 5., 6.]])
    delta = d.null_write(h, direction, lambda x: x[:1])
    np.testing.assert_allclose(delta[:, 0], 0., atol=1e-5)
    np.testing.assert_allclose(delta[:, 1:], direction[:, 1:])
    target = jnp.zeros((2, 2))
    shift, support = d.donor_write(h, target, target, jnp.empty((0, 3)), jnp.empty((0, 2)), jnp.eye(3), jnp.ones(2))
    assert not support.any()
    assert not shift.any()
    shift, support = d.donor_write(h, target, target, jnp.zeros((1, 3)), jnp.ones((1, 2)), jnp.eye(3), jnp.ones(2))
    assert not support.any()
    shift, support = d.donor_write(h, target, target, jnp.zeros((1, 3)), jnp.zeros((1, 2)), jnp.eye(3), jnp.ones(2))
    assert support.all()
    np.testing.assert_allclose(shift, -h)


def test_gradient_and_causal_training():
    h = jnp.zeros((4, 3))
    policy = lambda x: jax.nn.softmax(x[:, :2], -1)
    delta = d.gradient_write(h, jnp.ones((4, 1)), lambda x: x[:, 2:], policy, steps=100)
    assert float(jnp.mean((delta[:, 2:] - 1) ** 2)) < .05
    donor = jnp.broadcast_to(jnp.array([2., -2., 0.]), h.shape)
    u, loss = d.das(h, donor, policy, policy(donor), beta=0., rank=1, steps=100)
    np.testing.assert_allclose(u.T @ u, jnp.eye(1), atol=1e-5)
    assert float(loss) < .02
    with pytest.raises(ValueError):
        d.das(h, donor, policy, policy(donor), rank=4)


def test_mlp_and_policy_inverse():
    b = jnp.array([[[.1, .9]], [[.9, .1]], [[.4, .6]], [[.6, .4]]])
    z = 4 * b[:, 0]
    inverse = d.policy_implied(b, z, steps=200)
    np.testing.assert_allclose(inverse(z), b, atol=.06)
    probe = d.mlp(z, b[:, 0], width=8, steps=200)
    np.testing.assert_allclose(probe(z), b[:, 0], atol=.06)
