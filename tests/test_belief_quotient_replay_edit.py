import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from experiments.belief_quotient_history_jax_2026_11 import env, models
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.environment import ladder_process
from experiments.belief_quotient_replay_edit_2026_11 import closed_loop, replay


def history(length=24):
    rng = np.random.default_rng(22)
    return np.eye(3, dtype=np.float32)[rng.integers(3, size=(3, length, 2))].reshape(3, length, 6)


@pytest.mark.parametrize("variant", [2, 3])
def test_replay_swap_and_exact_filter(variant):
    process = ladder_process(variant)
    orig = history()
    swap, _ = replay.edit(orig, "swap", np.random.default_rng(12), variant == 3)
    if variant == 2:
        np.testing.assert_array_equal(swap[..., 3:], orig[..., 3:])
    else:
        np.testing.assert_array_equal(swap[..., 3:], orig[..., 3:][..., replay.SWAP_ACTION])
    b = replay.filtered(process, orig, variant)
    twin = replay.filtered(process, swap, variant)
    np.testing.assert_allclose(twin[..., :3], b[..., replay.SWAP_TOKEN], atol=1e-10)
    tokens = orig[..., :3].argmax(-1)
    actions = orig[..., 3:].argmax(-1)
    y = env.belief_reset(process, tokens[0, 0])
    for i in range(1, orig.shape[1]):
        y = env.belief_update(process, y, actions[0, i], tokens[0, i])
    np.testing.assert_allclose(y, b[0, -1, :3], atol=1e-6)


def test_fair_coin_preserves_quotient_not_full_belief():
    process = ladder_process(2)
    obs = history()
    alt, _ = replay.edit(obs, "relabel", np.random.default_rng(13))
    np.testing.assert_array_equal(obs[..., 2:], alt[..., 2:])
    b, bt = [replay.filtered(process, x, 2) for x in (obs, alt)]
    np.testing.assert_array_equal(b[..., 3], bt[..., 3])
    assert np.abs(b[..., 2] - bt[..., 2]).max() > 1e-5
    assert np.abs((b[..., 0] - b[..., 1]) - (bt[..., 0] - bt[..., 1])).max() > 1e-5


def test_site_patch_identity_and_temporal_scope():
    spec = models.ModelSpec("transformer", 6, 3, 8, 4, context_len=4)
    params = models.init_params(spec, jax.random.PRNGKey(31))
    obs = history()
    alt, mask = replay.edit(obs, "flip", np.random.default_rng(12), positions=[17])
    with jax.default_matmul_precision("highest"):
        z, h = replay.forward(spec, params, obs)
        z_alt, h_alt = replay.forward(spec, params, alt)
        np.testing.assert_allclose(models.sequence(spec, params, obs)[0], z, atol=1e-6)
        for i in range(5):
            empty = replay.patch(spec, params, h[..., i, :], h_alt[..., i, :], np.zeros_like(mask), i)
            all_positions = replay.patch(spec, params, h[..., i, :], h_alt[..., i, :], np.ones_like(mask), i)
            np.testing.assert_allclose(empty, z, atol=1e-6)
            np.testing.assert_allclose(all_positions, z_alt, atol=1e-6)
        last = replay.patch(spec, params, h[..., 4, :], h_alt[..., 4, :], mask, 4)
        np.testing.assert_allclose(last[:, 18:], z[:, 18:], atol=1e-6)


def test_construct_preserves_actions_and_refilters():
    process = ladder_process(2)
    obs = history(closed_loop.SUFFIX)
    start = jnp.broadcast_to(process.initial, (len(obs), 3))
    goal = jnp.broadcast_to(jnp.array([0., 0., 1.]), start.shape)
    alt, b_ref = closed_loop.construct(process, start, jnp.asarray(obs), jnp.arange(closed_loop.SUFFIX), goal, 3)
    np.testing.assert_array_equal(alt[..., 3:], obs[..., 3:])
    np.testing.assert_allclose(b_ref, replay.filtered(process, np.asarray(alt), 2)[:, -1, :3], atol=1e-6)
    assert np.all(np.asarray(b_ref[:, 2]) >= replay.filtered(process, obs, 2)[:, -1, 2] - 1e-6)


def test_fidelity_scores_direction_and_magnitude():
    logits = np.array([[0., 1., 2.], [2., 1., 0.]])
    twin = logits[:, ::-1]
    drop = np.array([0., 1.])
    for subset in ("all", "high_drop", "low_drop"):
        assert replay.fidelity(logits, twin, twin, drop)[subset]["fidelity"] == 1
        assert replay.fidelity(logits, twin, logits, drop)[subset]["fidelity"] == 0
        assert replay.fidelity(logits, logits, logits, drop)[subset]["fidelity"] is None
