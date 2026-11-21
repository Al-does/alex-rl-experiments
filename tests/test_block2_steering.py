import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.algorithm_generality_probes_eps02_2026_11 import steer_moving_base as base
from experiments.belief_quotient_block2_steering_2026_11 import continuation as c
from experiments.belief_quotient_causal_decoding_2026_11 import decoders as d
from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.reward_factors_block2_steering_2026_11.encoder_run import choose_strength, result_path
from experiments.reward_factors_block2_steering_2026_11 import report


@pytest.fixture
def model():
    spec = models.ModelSpec("transformer", 6, 3, 8, 4, 1, 3)
    params = models.init_params(spec, jax.random.PRNGKey(0))
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 11, 6))
    carry = models.init_carry(spec, 2)
    states, outputs, sites = [], [], []
    for t in range(obs.shape[1]):
        states.append(carry)
        z, h, carry = base.forward(spec, params, carry, obs[:, t], jnp.int32(t))
        outputs.append(z)
        sites.append(h)
    return spec, params, obs, states, jnp.stack(outputs, 1), jnp.stack(sites, 1)


@pytest.mark.parametrize("site", (1, 3))
def test_continuation_matches_real_forward_and_gradients(model, site):
    spec, params, obs, caches, zs, hs = model
    for t in (0, 2, 7, 10):
        h, cache, time = hs[:, t, site], caches[t], jnp.int32(t)
        continuation = lambda x: c.logits(spec, params, x, cache, time, site)
        np.testing.assert_allclose(continuation(h), zs[:, t], atol=2e-6)
        delta = jnp.ones_like(h) * .2
        target, _, _ = base.forward(spec, params, cache, obs[:, t], time, site=site, shift=lambda x: delta)
        np.testing.assert_allclose(continuation(h+delta), target, atol=2e-6)
        direction = jax.random.normal(jax.random.PRNGKey(3), h.shape)
        analytic = jnp.sum(jax.grad(lambda x: continuation(x).sum())(h) * direction)
        epsilon = .001
        numeric = (continuation(h+epsilon*direction).sum()-continuation(h-epsilon*direction).sum())/(2*epsilon)
        np.testing.assert_allclose(analytic, numeric, atol=.002, rtol=.02)


@pytest.mark.parametrize("site", (1, 3))
def test_reconstructed_real_context_matches_batched_queries(model, site):
    spec, params, _, caches, zs, hs = model
    ep, times = np.array([0, 1, 0, 1]), np.array([0, 2, 7, 10])
    context = c.contexts(spec, params, hs, ep, times, site)
    for layer in range(site+1, spec.n_layers):
        for row, (e, t) in enumerate(zip(ep, times)):
            for key in ("k", "v"):
                np.testing.assert_allclose(context[key][layer, row], caches[t][key][layer, e], atol=2e-6)
    pred = c.batch_logits(spec, params, hs[ep, times, site], context, jnp.asarray(times), site)
    np.testing.assert_allclose(pred, zs[ep, times], atol=2e-6)
    assert c.jacobians(spec, params, hs[ep, times, site], context, jnp.asarray(times), site).shape == (4, 3, spec.width)


def test_full_donor_is_not_lookup_before_final_block(model):
    spec, params, _, caches, zs, hs = model
    t = jnp.int32(10)
    donor = hs[::-1, 10]
    tail = lambda site: c.logits(spec, params, donor[:, site], caches[10], t, site)
    np.testing.assert_allclose(tail(3), zs[::-1, 10], atol=2e-6)
    assert float(jnp.max(jnp.abs(tail(1)-zs[::-1, 10]))) > 1e-4
    empty_cache = jax.tree_util.tree_map(jnp.zeros_like, caches[10])
    changed = c.logits(spec, params, hs[:, 10, 1], empty_cache, t, 1)
    assert float(jnp.max(jnp.abs(changed-zs[:, 10]))) > 1e-4


def test_gradient_and_null_writes_use_downstream_cache(model):
    spec, params, _, caches, _, hs = model
    h, cache, t = hs[:, 10, 1], caches[10], jnp.int32(10)
    policy = lambda x: jax.nn.softmax(c.logits(spec, params, x, cache, t, 1), -1)
    target = jnp.roll(policy(h), 1, axis=-1)
    delta = d.behavioural_write(h, target, policy, jnp.eye(spec.width), steps=20)
    assert float(d.kl(target, policy(h+delta)).mean()) < float(d.kl(target, policy(h)).mean())
    transform = lambda z: z[:1]
    projected = c.null_write(spec, params, h, jnp.ones_like(h), cache, t, 1, transform)
    jac = c.jacobians(spec, params, h, cache, jnp.full((len(h),), t), 1, transform)
    np.testing.assert_allclose(jnp.einsum("bij,bj->bi", jac, projected), 0., atol=1e-5)


def test_write_changes_only_later_cache_and_survives_to_next_step(model):
    spec, params, obs, caches, _, hs = model
    t = jnp.int32(8)
    _, _, original = base.forward(spec, params, caches[8], obs[:, 8], t)
    _, _, written = base.forward(spec, params, caches[8], obs[:, 8], t, site=1,
                                shift=lambda x: jnp.arange(spec.width)[None]*.1)
    for key in ("k", "v"):
        np.testing.assert_array_equal(original[key][:2], written[key][:2])
        assert not np.allclose(original[key][2:], written[key][2:])
    a = base.forward(spec, params, original, obs[:, 9], jnp.int32(9))[0]
    b = base.forward(spec, params, written, obs[:, 9], jnp.int32(9))[0]
    assert not np.allclose(a, b)


def test_donor_matching_selection_paths_and_input_errors(model):
    h = jnp.zeros((2, 3))
    bank = jnp.array([[1., 0., 0.], [0., 1., 0.]])
    labels = jnp.array([[1., 0.], [0., 1.]])
    goal = jnp.array([[1., 0.], [.5, .5]])
    index, supported = c.donor_index(h, goal, goal, bank, labels, jnp.eye(3)[:, :1], jnp.zeros(2), jnp.ones(3))
    np.testing.assert_array_equal(index[:1], [0])
    np.testing.assert_array_equal(supported, [True, False])
    _, supported = c.donor_index(h, goal, goal, bank[:0], labels[:0], jnp.eye(3), jnp.zeros(2), jnp.ones(3))
    assert not supported.any()
    spec, params, _, caches, _, hs = model
    with pytest.raises(ValueError):
        c.contexts(spec, params, hs, [], [], 1)
    with pytest.raises(ValueError):
        c.logits(spec, params, h, caches[0], jnp.int32(0), 0)
    assert choose_strength((0., .5, 1.), {"0.0": .4, "0.5": .09, "1.0": .08}, True, .1) == .5
    assert choose_strength((0., .5, 1.), {"0.0": .4, "0.5": .5, "1.0": .5}, False) == .5
    assert "site1" in result_path("ladder", 42, "marginal", "mlp", "exact", 1).name


def test_grid_reference_matches_historical_rounding():
    from experiments.algorithm_generality_probes_eps02_2026_11.steer_prediction import Policy
    from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
    for variant in (2, 3):
        policy = Policy(variant, n=10, max_iterations=2, environment=environment)
        belief = np.random.default_rng(42).dirichlet(np.ones(3), 30)
        np.testing.assert_array_equal(c.grid_actions(policy, jnp.asarray(belief)), policy(belief))


def test_report_rejects_empty_wrong_site_and_smoke(tmp_path):
    with pytest.raises(FileNotFoundError):
        report.aggregate(tmp_path)
    first = None
    for site in (1, 3):
        for test in report.TESTS:
            for method in (*report.METHODS, *report.ENCODERS):
                actual = method.replace("joint_", "marginal_", 1) if test.startswith("ladder") and method.startswith("joint_") else method
                file = report.path(tmp_path, test, 42, actual, site)
                file.parent.mkdir(parents=True, exist_ok=True)
                value = 2. if method == "map" else 1. if method == "intact" else 1.5
                metric = dict(reward=value, actions_changed=.1)
                row = dict(test=test, seed=42, site=site, smoke=False, tuning_episodes=16,
                           confirmation_episodes=32, horizon=1024, positions="latest_every_step",
                           action_change_reference="intact_same_executed_history",
                           metrics={"0": metric, "1": metric, "2": metric})
                file.write_text(json.dumps(row))
                if first is None:
                    first = file
    result = report.aggregate(tmp_path)
    assert len(result["per_seed"]) == 52
    assert result["aggregate"]["1:eq4"]["moving_gap"] == {"mean": .5, "sd": None}
    original = json.loads(first.read_text())
    for key, value in (("site", 3), ("smoke", True), ("confirmation_episodes", 2), ("action_change_reference", "local")):
        first.write_text(json.dumps(dict(original, **{key: value})))
        with pytest.raises(ValueError):
            report.aggregate(tmp_path)
