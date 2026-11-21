"""Small predictive-only identification and timing/alignment contracts."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import json
from types import SimpleNamespace

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


@pytest.mark.parametrize("kind", ["affine", "gru", "tanh", "hmm"])
def test_vmapped_step_matches_independent_sequential_adam(kind):
    spec = core.Spec(kind, 2, 3, 1, 3)
    params = jax.vmap(lambda seed: core.init(spec, seed))(jnp.array([0, 1, 0, 1]))
    optimizer, single, parallel = core.train_steps(spec)
    opt = jax.vmap(optimizer.init)(params)
    x, y, _ = synthetic(2, 4, 4)
    xs, ys = np.stack([x]*4), np.stack([y]*4)
    rates = jnp.array([.001, .001, .003, .003])
    actual_p, actual_opt, values = parallel(params, opt, xs, ys, rates)
    for i in range(4):
        expected_p, expected_opt, value = single(jax.tree.map(lambda a: a[i], params),
                                                jax.tree.map(lambda a: a[i], opt), x, y, rates[i])
        for a, b in zip(jax.tree.leaves((actual_p, actual_opt)), jax.tree.leaves((expected_p, expected_opt))):
            np.testing.assert_allclose(a[i], b, atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(values[i], value, atol=1e-6)


def test_group_fit_keeps_independent_validation_best():
    x, y, _ = synthetic(4, 5, 4)
    spec = core.Spec("tanh", 2, 3, 1, 3)
    results = core.fit_many(spec, x, y, x, y, seeds=(0, 1), lrs=(.001, .003),
                            updates=11, val_interval=5, batch=2)
    assert len(results) == 4
    for p, metric in results:
        assert metric["selected_update"] in (5, 10, 11)
        assert metric["parallel_candidates"] == 4
        np.testing.assert_allclose(float(core.loss(spec, p, x, y)), metric["val_kl"], atol=1e-6)


def test_harness_pending_manifest_durability_contract(tmp_path, monkeypatch):
    from devops.vast import self_destruct
    relative = "experiments/study/analysis_gbs/results/run/run_manifest.json"
    manifest = tmp_path/relative
    manifest.parent.mkdir(parents=True)
    monkeypatch.setattr(self_destruct, "_run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=f"?? {relative}\0"))
    manifest.write_text(json.dumps({"status": "running"}))
    assert not self_destruct.required_durability_completed(tmp_path, log=lambda _: None)
    manifest.write_text(json.dumps({"status": "completed", "remote_artifacts": {"status": "completed"}}))
    assert not self_destruct.required_durability_completed(tmp_path, log=lambda _: None)
    manifest.with_name("remote_artifacts.json").write_text("{}")
    assert self_destruct.required_durability_completed(tmp_path, log=lambda _: None)


def test_rr_control_repairs_observed_gpu_distribution_drift():
    from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
    T, E = environment.action_transitions(3), environment.emission_matrix()
    process = {"initial": np.full(9, 1/9),
               "transitions": np.stack([np.kron(a, b) for a in T for b in T]),
               "emission": np.kron(E, E)}
    rng = np.random.default_rng(32)
    records = {"token": rng.integers(9, size=(2, 128)), "action": rng.integers(9, size=(2, 128))}
    records["previous_action"] = np.concatenate((np.full((2, 1), -1), records["action"][:, :-1]), 1)
    x = np.maximum(records["previous_action"], 0)*9+records["token"]
    # Actual offending row from box 6's failed seed-42 training dataset.
    row = np.array([.0015116623835638165, .0030495277605950832, .03505082055926323,
                    .00408769678324461, .008245193399488926, .09483004361391068,
                    .032553721219301224, .06572653353214264, .7557574510574341], dtype=np.float32)
    bad = np.broadcast_to(row, (2, 128, 9, 9))
    spec = core.Spec("gru", 2, 9, 9, 81, 32, 9)
    with pytest.raises(ValueError, match="distributions"):
        core.fit_many(spec, x, bad, x, bad, updates=1)
    exact = data.exact_targets(records, process)
    assert exact["all_predictive"].dtype == np.float64
    np.testing.assert_allclose(exact["all_predictive"].sum(-1), 1, atol=5e-16, rtol=0)
    np.testing.assert_allclose(exact["belief"].sum(-1), 1, atol=5e-16, rtol=0)
    selected = np.take_along_axis(exact["all_predictive"], records["action"][..., None, None], 2)[..., 0, :]
    np.testing.assert_array_equal(exact["predictive"], selected)
    results = core.fit_many(spec, x, exact["all_predictive"], x, exact["all_predictive"],
                            seeds=(0,), lrs=(.003,), updates=2, val_interval=1, batch=2, chunk_length=64)
    assert np.isfinite(results[0][1]["val_kl"])


@pytest.mark.parametrize("kind", ["affine", "gru", "tanh", "hmm"])
def test_carried_chunks_equal_complete_reset_history(kind):
    spec = core.Spec(kind, 2, 3, 1, 3)
    p = core.init(spec, 13)
    x, _, _ = synthetic(2, 128, 14)
    full = core.states(spec, p, x)
    first = core.scan_states(spec, p, x[:, :64], jnp.zeros((2, 2)), 0)
    second = core.scan_states(spec, p, x[:, 64:], first[:, -1], 64)
    np.testing.assert_allclose(np.concatenate((first, second), 1), full, atol=1e-6)


def test_tbptt_validation_boundaries_keep_optimizer_and_carry():
    spec = core.Spec("gru", 2, 3, 1, 3)
    params = jax.tree.map(lambda a: a[None], core.init(spec, 2))
    optimizer, _, _ = core.train_steps(spec)
    opt = jax.vmap(optimizer.init)(params)
    block = core.chunk_block(spec, optimizer)
    x, y, _ = synthetic(2, 16, 3)
    xs = np.moveaxis(x.reshape(1, 2, 4, 4), 2, 0)
    ys = np.moveaxis(y.reshape(1, 2, 4, 4, 3), 2, 0)
    carry = jnp.zeros((1, 2, 2))
    whole = block(params, opt, carry, xs, ys, jnp.arange(4)*4, jnp.array([.003]))
    p, o, c, _ = block(params, opt, carry, xs[:2], ys[:2], jnp.arange(2)*4, jnp.array([.003]))
    split = block(p, o, c, xs[2:], ys[2:], jnp.arange(2)*4+8, jnp.array([.003]))
    for expected, actual in zip(jax.tree.leaves(whole[:3]), jax.tree.leaves(split[:3])):
        np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("chunk", [-1, 3])
def test_invalid_tbptt_window_rejected(chunk):
    x, y, _ = synthetic(2, 8, 3)
    with pytest.raises(ValueError, match="chunk length"):
        core.fit(core.Spec("gru", 2, 3, 1, 3), x, y, x, y, updates=1, chunk_length=chunk)


@pytest.mark.parametrize("k", [2, 6, 10])
def test_affine_symbol_selection_values_and_gradients(k):
    spec = core.Spec("affine", k, 3, 3, 3)
    p = core.init(spec, 12)
    x = jnp.arange(32) % 9
    g = jax.random.normal(jax.random.PRNGKey(2), (32, k))
    def reference(p, g):
        return jnp.square(jnp.einsum("bi,bij->bj", g, p["A"][x])+p["b"][x]).sum()
    def current(p, g):
        return jnp.square(core.update(spec, p, g, x)).sum()
    expected = jax.value_and_grad(reference, (0, 1))(p, g)
    actual = jax.value_and_grad(current, (0, 1))(p, g)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-6)


def test_constant_geometry_and_nonfinite_metrics_write_as_null(tmp_path):
    from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs import run
    state = np.full((2, 8, 2), .1, dtype=np.float32)
    truth = {"belief": np.full((2, 8, 3), 1/3, dtype=np.float32)}
    geometry = compare.geometry(state, state, truth, truth, mlp_updates=1)
    for name in ("affine_r2", "mlp_r2", "reverse_affine_r2", "reverse_mlp_r2"):
        assert geometry["belief"][name] is None
        assert geometry["belief"][name+"_reason"]
    path = tmp_path/"fit.json"
    run.write(path, {"geometry": geometry, "bad": float("nan"), "nested": [float("inf")]})
    saved = json.loads(path.read_text())
    assert saved["bad"] is None and saved["nested"] == [None]
    assert saved["nonfinite_metrics"]["bad"]
    assert compare.r2(np.empty((0, 2)), np.empty((0, 2))) is None
    assert compare.r2(np.arange(3.), np.full(3, np.nan)) is None


def test_probe_failure_does_not_skip_other_probes(monkeypatch):
    from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs import run
    spec = core.Spec("tanh", 2, 3, 1, 3)
    x, _, b = synthetic(2, 8, 3)
    splits = {name: {"x": x, "belief": b, "truth_belief": b} for name in ("train", "test")}
    def broken(*a, **kw):
        raise ValueError("probe failed")
    monkeypatch.setattr(compare, "geometry", broken)
    monkeypatch.setattr(compare, "commutation", lambda *a: {"mse": np.nan})
    result = run.probe_fit(spec, core.init(spec, 1), splits, None, True, 1)
    assert result["geometry"] is None
    assert result["geometry_error"] == "ValueError: probe failed"
    assert result["commutation"]["mse"] is None
    assert result["commutation"]["nonfinite_metrics"]["mse"]
    assert "hmm_matrices" in result


def test_summarizer_accepts_failed_probes(tmp_path, monkeypatch):
    from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_gbs import summarize
    root = tmp_path/"results"
    root.mkdir()
    (root/"run_manifest.json").write_text(json.dumps({"status": "completed"}))
    (root/"summary.json").write_text(json.dumps({"elbows": [{"seed": 42, "class": "affine", "elbow_k": 1}]}))
    (root/"seed42_affine_k1.json").write_text(json.dumps({"condition": "variant_1", "seed": 42,
        "control": False, "class": "affine", "k": 1, "test_kl": .01, "selected": {"val_kl": .01},
        "geometry": None, "geometry_error": "probe failed", "commutation": None}))
    output = tmp_path/"summary.csv"
    monkeypatch.setattr("sys.argv", ["summarize", "--inputs", str(root), "--output", str(output)])
    summarize.main()
    assert "probe failed" in output.read_text()
