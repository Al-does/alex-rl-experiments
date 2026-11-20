"""Small predictive-only identification and timing/alignment contracts."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs import core, compare, data


def synthetic(count, length, seed):
    rng = np.random.default_rng(seed)
    T = np.array([[.85, .1, .05], [.1, .8, .1], [.05, .1, .85]])
    E = np.array([[.8, .15, .05], [.05, .8, .15], [.15, .05, .8]])
    b = np.full((count, 3), 1/3)
    xs, bs, targets = [], [], []
    for t in range(length):
        if t:
            b = b @ T
        probabilities = b @ E
        token = np.array([rng.choice(3, p=p) for p in probabilities])
        b = b * E[:, token].T
        b /= b.sum(-1, keepdims=True)
        xs.append(token)
        bs.append(b.copy())
        targets.append(b @ T @ E)
    return np.stack(xs, 1), np.stack(targets, 1), np.stack(bs, 1)


def test_predictive_fit_recovers_three_state_geometry():
    x, y, b = synthetic(64, 24, 1)
    vx, vy, _ = synthetic(16, 24, 2)
    tx, ty, tb = synthetic(32, 24, 3)
    spec = core.Spec("hmm", 2, 3, 1, 3)
    p, metric = core.fit(spec, x, y, vx, vy, updates=1800, batch=32, lr=.003)
    g, tg = compare.flat(core.states(spec, p, x)), compare.flat(core.states(spec, p, tx))
    score = compare.r2(compare.flat(tb), compare.predict(tg, compare.affine(g, compare.flat(b))))
    print(f"synthetic 3-state predictive fit: test_kl={float(core.loss(spec,p,tx,ty)):.8g} belief_affine_r2={score:.8f}")
    assert score > .95
    assert metric["val_kl"] < .003


@pytest.mark.parametrize("kind", ["affine", "gru", "tanh", "hmm"])
@pytest.mark.parametrize("k", [0, 2])
def test_states_reset_constant_and_finite_gradients(kind, k):
    spec = core.Spec(kind, k, 3, 1, 3)
    p = core.init(spec, 2)
    x, y, _ = synthetic(2, 4, 4)
    g = np.asarray(core.states(spec, p, x))
    assert g.shape == (2, 4, k)
    grad = jax.grad(core.loss, 1)(spec, p, x, y)
    assert all(np.isfinite(a).all() for a in jax.tree.leaves(grad))
    if k == 0:
        logp = np.asarray(core.log_predictions(spec, p, x))
        np.testing.assert_allclose(logp, np.broadcast_to(logp[0, 0], logp.shape))


def test_hmm_reset_omits_transition():
    spec = core.Spec("hmm", 2, 3, 1, 3)
    p = core.init(spec, 3)
    p["prior"] = jnp.array([3., -2., 0.])
    p["T"] = jnp.array([[[0., 0., 10.]]*3])
    g = core.initial(spec, p)
    result = core.update(spec, p, g, jnp.array(0), True)
    expected = jax.nn.softmax(p["prior"])*jax.nn.softmax(p["E"], -1)[:, 0]
    np.testing.assert_allclose(result, (expected/expected.sum())[:-1], atol=1e-6)


def test_empty_and_invalid_targets_rejected():
    spec = core.Spec("tanh", 1, 3, 1, 3)
    x, y, _ = synthetic(2, 4, 4)
    with pytest.raises(ValueError, match="nonempty"):
        core.fit(spec, x[:0], y[:0], x, y, updates=1)
    with pytest.raises(ValueError, match="distributions"):
        core.fit(spec, x, -y, x, y, updates=1)


def test_conditional_control_loss():
    spec = core.Spec("tanh", 2, 3, 2, 6, output_groups=2)
    p = core.init(spec, 0)
    x = jnp.zeros((2, 4), dtype=jnp.int32)
    logp = core.log_predictions(spec, p, x)
    assert logp.shape == (2, 4, 2, 3)
    np.testing.assert_allclose(float(core.loss(spec, p, x, jnp.exp(logp))), 0, atol=1e-6)


def test_quotient_and_rr_targets():
    from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.environment import ladder_process
    process = ladder_process(2)
    prior, T, E = data.quotient_model(process)
    np.testing.assert_allclose(T.sum(-1), 1, atol=1e-6)
    np.testing.assert_allclose(E.sum(-1), 1, atol=1e-6)
    result = data.truth_targets({"belief": np.ones((1, 2, 3))/3, "token": np.array([[0, 2]]),
                                 "previous_action": np.array([[0, 1]])}, process, True)
    expected = prior*E[:, 0]
    expected /= expected.sum()
    np.testing.assert_allclose(result["quotient"][0, 0], expected)
    rr = data.truth_targets({"belief": np.ones((1, 2, 9))/9}, None, False)
    np.testing.assert_allclose(rr["factor_1"], 1/3)
    np.testing.assert_allclose(rr["factor_2"], 1/3)


def test_matrix_comparison_recovers_permutation():
    from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.environment import ladder_process
    process = ladder_process(2)
    spec = core.Spec("hmm", 2, 3, 3, 3)
    p = core.init(spec, 0)
    order = jnp.array([2, 0, 1])
    p["T"] = jnp.log(process.transitions[:, order][:, :, order])
    p["E"] = jnp.log(process.emission[order])
    result = compare.matrix_error(spec, p, process, True)
    assert result["combined_mse"] < 1e-12
    assert result["permutation_method"] == "exhaustive"


def test_commutation_exact_filter_and_constant_k0():
    from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.environment import ladder_process
    process = ladder_process(2)
    spec = core.Spec("hmm", 2, 3, 3, 3)
    p = core.init(spec, 0)
    p.update(T=jnp.log(process.transitions), E=jnp.log(process.emission), prior=jnp.log(process.initial))
    x = np.random.default_rng(1).integers(0, 9, (4, 20))
    g = np.asarray(core.states(spec, p, x))
    b = np.concatenate((g, 1-g.sum(-1, keepdims=True)), -1)
    score = compare.commutation(spec, p, g[:2], g[2:], b[:2], b[2:], x[2:], process)
    assert score["mse"] < 1e-8
    assert score["constant_decoder_mse"] > score["mse"]
    zero = core.Spec("tanh", 0, 3, 3, 3)
    zp = core.init(zero, 0)
    train_g, test_g = core.states(zero, zp, x[:2]), core.states(zero, zp, x[2:])
    result = compare.geometry(train_g, test_g, {"belief": b[:2]}, {"belief": b[2:]}, mlp_updates=1)
    assert result["belief"]["reverse_affine_r2"] is None


def test_rollout_symbols_reset_policy_and_predictive_targets():
    from experiments.belief_quotient_history_jax_2026_11 import models
    from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.environment import ladder_process
    process = ladder_process(2)
    spec = models.architecture_spec("memoryless", obs_dim=6, n_actions=3, n_tokens=3, context_len=32)
    p = models.init_params(spec, jax.random.PRNGKey(10))
    result = jax.tree.map(np.asarray, data.collect_batch(spec, process, p, jax.random.PRNGKey(20), 2))
    assert result["x"].shape == (2, 1024)
    assert (result["previous_action"][:, 0] == -1).all()
    assert result["reset"][:, 0].all() and not result["reset"][:, 1:].any()
    np.testing.assert_array_equal(result["previous_action"][:, 1:], result["action"][:, :-1])
    np.testing.assert_array_equal(result["x"][:, 1:], result["previous_action"][:, 1:]*3+result["token"][:, 1:])
    np.testing.assert_allclose(result["policy"].sum(-1), 1, atol=1e-6)
    chosen = np.take_along_axis(result["all_predictive"], result["action"][..., None, None], axis=2)[..., 0, :]
    np.testing.assert_allclose(result["predictive"], chosen, atol=1e-6)
    prior = np.asarray(process.initial)*np.asarray(process.emission)[:, result["token"][:, 0]].T
    np.testing.assert_allclose(result["belief"][:, 0], prior/prior.sum(-1, keepdims=True), atol=1e-6)
