import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import steering_gru as gru


def fixture(layers=2, width=16):
    spec = models.ModelSpec('gru', 6, 3, width, layers, chunk_len=32)
    params = models.init_params(spec, jax.random.PRNGKey(7))
    maps = gru.empty_maps(spec)
    maps['decoder'] = jax.random.normal(jax.random.PRNGKey(8), maps['decoder'].shape)*.1
    maps['encoder'] = jax.random.normal(jax.random.PRNGKey(9), maps['encoder'].shape)
    maps['orthogonal'] = jax.random.normal(jax.random.PRNGKey(11), maps['orthogonal'].shape)
    observations = jax.random.normal(jax.random.PRNGKey(10), (12, 3, 6))
    return spec, params, maps, observations


@pytest.mark.parametrize('layers', (2, 4))
def test_unsteered_step_matches_trainer(layers):
    spec, params, maps, observations = fixture(layers)
    carry = expected_carry = models.init_carry(spec, 3)
    for t, obs in enumerate(observations):
        logits, carry, sites, _ = gru.step(spec, params, carry, obs, maps)
        expected, _, embedding, expected_carry = models.step(spec, params, expected_carry, obs, jnp.asarray(t))
        np.testing.assert_allclose(logits, expected, atol=1e-6)
        np.testing.assert_allclose(carry, expected_carry, atol=1e-6)
        np.testing.assert_allclose(sites[:, -1], embedding, atol=1e-6)
        np.testing.assert_allclose(sites[:, :-1], jnp.moveaxis(carry, 0, 1), atol=1e-6)


def test_write_installs_the_shadow_target_and_is_carried():
    spec, params, maps, observations = fixture()
    carry = models.init_carry(spec, 3)
    _, _, reference, _ = gru.step(spec, params, carry, observations[0], maps)
    _, written, sites, stats = gru.step(spec, params, carry, observations[0], maps, 0, 1, 1., False, reference)
    wanted = gru.decode(reference[:, 0], maps, 0)[:, jnp.array([1, 0, 2])]
    moved = gru.decode(sites[:, 0], maps, 0)-gru.decode(reference[:, 0], maps, 0)
    np.testing.assert_allclose(moved[:, :2], ((wanted-gru.decode(reference[:, 0], maps, 0))[:, :2]
                               @ maps['encoder'][0]) @ maps['decoder'][0], atol=1e-4)
    np.testing.assert_allclose(written[0], sites[:, 0], atol=1e-6)
    assert float(stats[:, 0].min()) > 0
    _, final_only, _, _ = gru.step(spec, params, carry, observations[0], maps, spec.n_layers, 1, 1., False, reference)
    np.testing.assert_allclose(final_only, jnp.moveaxis(reference[:, :-1], 0, 1), atol=1e-6)


def test_pulse_writes_once_and_zero_shift_is_identity():
    spec, params, maps, observations = fixture()
    saved = gru.carries(spec, params, maps, observations, jnp.asarray([0, 4]))
    np.testing.assert_allclose(saved[0], 0)
    carry = models.init_carry(spec, 3)
    for obs in observations[:4]:
        carry = gru.step(spec, params, carry, obs, maps)[1]
    np.testing.assert_allclose(saved[1], carry, atol=1e-6)
    none = jnp.zeros((3, spec.n_layers+1))
    diff, tv, _, relative, norms = gru.pulse_branch(spec, params, maps, saved[1], observations[4:], gru.ALL_HIDDEN, 1, False, none)
    control = gru.pulse_branch(spec, params, maps, saved[1], observations[4:], gru.ALL_HIDDEN, 1, True, norms[0])
    np.testing.assert_allclose(control[4][0], norms[0], rtol=1e-5)
    np.testing.assert_allclose(control[4][1:], 0)
    assert float(norms[0, :, :spec.n_layers].min()) > 0
    assert float(jnp.abs(diff[0]).max()) > 1e-4 and float(relative[0].min()) > 0
    np.testing.assert_allclose(relative[1:], 0)
    zero = dict(maps, encoder=jnp.zeros_like(maps['encoder']))
    diff, tv, changed, _, _ = gru.pulse_branch(spec, params, zero, saved[1], observations[4:], gru.ALL_HIDDEN, 1, False, none)
    np.testing.assert_allclose(diff, 0, atol=1e-6)
    assert not bool(changed.any()) and float(tv.max()) < 1e-6


def test_tracking_reports_slope_and_correlation():
    exact = np.array([[1., -1., 2., .5], [.5, -.5, 1., .25]])
    result = gru.tracking(.5*exact, exact)
    np.testing.assert_allclose(result['slope'], .5)
    np.testing.assert_allclose(result['correlation'], 1.)
