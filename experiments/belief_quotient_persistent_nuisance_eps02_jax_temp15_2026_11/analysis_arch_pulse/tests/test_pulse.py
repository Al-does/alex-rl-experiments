import importlib
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
REFERENCE = Path(os.environ.get("PULSE_REFERENCE", HERE.parent / "artifacts" / "arch_pulse_reference"))
if not REFERENCE.exists():
    pytest.skip("Set up the pinned PR239 worktree as documented in analysis_arch_pulse/README.md", allow_module_level=True)

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_arch_pulse import recurrent as r
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_arch_pulse.reference import core, pulse
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.analysis_arch_pulse import diagnose_rebound as audit

import jax
import jax.numpy as jnp
import numpy as np

from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment


@pytest.fixture
def setup():
    spec = models.ModelSpec("gru", 6, 3, 8, 2)
    params = models.init_params(spec, jax.random.PRNGKey(9))
    maps = r.empty_maps(spec)
    maps["decoder"] = maps["decoder"].at[:, :2].set(jnp.eye(2))
    maps["encoder"] = maps["encoder"].at[:, :, :2].set(jnp.eye(2))
    maps["bias"] = maps["bias"] + jnp.array([.2, .3])
    obs = jnp.array([[1., 0., 0., 0., 1., 0.]])
    return spec, params, maps, obs


@pytest.mark.parametrize("layers,width", [(2, 128), (4, 92)])
def test_sites_and_unwritten_step_match_trainer(layers, width):
    spec = models.ModelSpec("gru", 6, 3, width, layers)
    params = models.init_params(spec, jax.random.PRNGKey(1))
    obs = jax.random.normal(jax.random.PRNGKey(2), (2, 6))
    carry = models.init_carry(spec, 2)
    maps = r.empty_maps(spec)
    logits, carry, h = r.step(spec, params, carry, obs, 0, maps)
    expected = models.step(spec, params, models.init_carry(spec, 2), obs, 0)
    np.testing.assert_allclose(logits, expected[0], atol=1e-6)
    np.testing.assert_allclose(carry, expected[-1], atol=1e-6)
    np.testing.assert_array_equal(h, np.swapaxes(carry, 0, 1))
    assert h.shape == (2, layers, width)
    assert maps["decoder"].shape == (layers, width, 2)
    assert maps["encoder"].shape == (layers, 2, width)


@pytest.mark.parametrize("site", [0, 1])
@pytest.mark.parametrize("mode", [1, 2])
def test_write_feeds_next_layer_and_is_carried(setup, site, mode):
    spec, params, maps, obs = setup
    initial = models.init_carry(spec, 1)
    _, intact, _ = r.step(spec, params, initial, obs, 0, maps)
    _, written, sites = r.step(spec, params, initial, obs, 0, maps, site, mode, 1.)
    base = intact[site]
    delta = core.ladder_displacement(base, maps, site, mode) @ maps["encoder"][site]
    assert np.linalg.norm(delta) > .01
    np.testing.assert_allclose(written[site], base + delta, atol=1e-6)
    np.testing.assert_array_equal(sites[:, site], written[site])
    if site == 0:
        downstream = models._gru_cell(params["blocks"][1], written[0], initial[1])
        np.testing.assert_allclose(written[1], downstream, atol=1e-6)
        assert not np.allclose(written[1], intact[1])
    else:
        np.testing.assert_array_equal(written[0], intact[0])
    _, next_written, _ = r.step(spec, params, written, obs, 1, maps)
    _, next_intact, _ = r.step(spec, params, intact, obs, 1, maps)
    assert not np.allclose(next_written, next_intact)


def test_branch_writes_only_once_and_replays_fixed_tokens_and_actions(setup):
    spec, params, maps, obs = setup
    observations = jnp.concatenate((obs, obs[:, ::-1], obs), axis=0)[:, None, :]
    carry = models.init_carry(spec, 1)
    final, decoded = r.branch(spec, params, maps, carry, observations, 0, 1, 2, 1.)
    manual, expected = carry, []
    for t, token_and_action in enumerate(observations):
        _, manual, h = r.step(spec, params, manual, token_and_action, t, maps, 1, 2, float(t == 0))
        expected.append(core.decode(h[:, -1], maps, 1))
    np.testing.assert_allclose(final, manual, atol=1e-6)
    np.testing.assert_allclose(decoded, expected, atol=1e-6)
    _, intact = r.branch(spec, params, maps, carry, observations, 0, 1, 2, 0.)
    assert not np.allclose(decoded, intact)
    np.testing.assert_array_equal(observations[1, 0], obs[0, ::-1])


def test_snapshots_are_before_pulse_step(setup):
    spec, params, maps, obs = setup
    observations = jnp.broadcast_to(obs, (5, 1, 6))
    saved = r.snapshots(spec, params, maps, observations, jnp.array([0, 2]), 2)
    carry = models.init_carry(spec, 1)
    np.testing.assert_array_equal(saved[0], carry)
    for t in range(2):
        _, carry, _ = r.step(spec, params, carry, obs, t, maps)
    np.testing.assert_allclose(saved[1], carry, atol=1e-6)


def test_rollout_matches_pinned_original_recurrent_driver(setup):
    spec, params, maps, _ = setup
    original = importlib.import_module(core.__package__ + ".steering_gru")
    process = environment.ladder_process(2)
    key = jax.random.PRNGKey(42)
    records = r.rollout(spec, process, params, maps, key, 2, 16)
    reference = original.rollout(spec, process, params, original.empty_maps(spec), key, 2, 16, capture=True)
    for name in ("obs", "belief", "action", "reward"):
        np.testing.assert_allclose(records[name], reference[name], atol=1e-6)
    np.testing.assert_allclose(records["h"], reference["h"][:, :, :spec.n_layers], atol=1e-6)


def synthetic_records():
    belief = np.random.default_rng(42).dirichlet([2, 2, 2], size=(20, 2)).astype(np.float32)
    hidden = np.stack((belief[..., :2], 2 * belief[..., :2] + 1), axis=2)
    return {"h": hidden, "belief": belief}


def test_fit_and_scores_all_sites_and_episode_major_retention():
    records = synthetic_records()
    h, b = r.retained(records, 20)
    np.testing.assert_array_equal(b[:12], records["belief"][8:, 0])
    maps = r.fit_maps(records, 20)
    assert h.shape == (20, 2, 2)
    for score in r.probe_scores(records, maps, 20):
        assert score["rho"] == pytest.approx(1., abs=1e-10)
        assert score["delta"] == pytest.approx(1., abs=1e-10)


def test_zero_variance_scores_are_null():
    records = synthetic_records()
    records["belief"][:] = [.25, .25, .5]
    maps = r.fit_maps(records, 20)
    assert r.probe_scores(records, maps, 20) == [{"rho": None, "delta": None}] * 2


@pytest.mark.parametrize("rows,warmup", [(0, 8), (-1, 8), (100, 8), (1, 20)])
def test_invalid_or_empty_retention(rows, warmup):
    with pytest.raises(ValueError):
        r.retained(synthetic_records(), rows, warmup)


@pytest.mark.parametrize("times,horizon", [((), 1), ((-1,), 1), ((2,), 3)])
def test_invalid_pulse_windows(setup, times, horizon):
    spec, params, maps, obs = setup
    with pytest.raises(ValueError, match="pulse times"):
        r.run_pulse(spec, params, maps, {"obs": jnp.broadcast_to(obs, (4, 1, 6))},
                    2, environment, times=times, horizon=horizon)


def test_exact_trace_updates_with_following_observation_and_preceding_action():
    beliefs = np.array([[.2, .3, .5]])
    observations = np.array([[[1, 0, 0, 0, 0, 0]], [[0, 1, 0, 0, 1, 0]]], float)
    changed = beliefs[:, [1, 0, 2]]
    process = environment.ladder_process(3)
    emission = np.asarray(process.emission)[:, 1]
    transition = np.asarray(process.transitions)[1]
    def update(b):
        b = (b @ transition) * emission
        return b / b.sum(-1, keepdims=True)
    expected = np.array([pulse.coordinates(changed) - pulse.coordinates(beliefs),
                         pulse.coordinates(update(changed)) - pulse.coordinates(update(beliefs))])
    actual = pulse.exact_trace(beliefs, observations, 3, "mirror", environment=environment)
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_short_pulse_reports_every_hidden_layer_and_lag(setup):
    spec, params, maps, _ = setup
    records = r.rollout(spec, environment.ladder_process(2), params, maps,
                        jax.random.PRNGKey(3042), 2, 8)
    result = r.run_pulse(spec, params, maps, records, 2, environment, times=(2, 4), horizon=2)
    assert set(result["conditions"]) == {f"{mode}_layer{i}" for mode in ("mirror", "rho_flip") for i in (1, 2)}
    for condition in result["conditions"].values():
        assert np.asarray(condition["measured"]["absolute"]).shape == (3, 2)
        assert np.isfinite(condition["measured"]["absolute"]).all()
    np.testing.assert_array_equal(result["conditions"]["mirror_layer1"]["full_filter"],
                                  result["conditions"]["mirror_layer2"]["full_filter"])


def test_independent_cell_terms_match_trainer(setup):
    spec, params, _, obs = setup
    hidden = jax.random.normal(jax.random.PRNGKey(7), (1, spec.width))
    x = models._dense(params["input"], obs)
    actual, _, _ = audit.cell_terms(params["blocks"][0], x, hidden)
    np.testing.assert_allclose(actual, models._gru_cell(params["blocks"][0], x, hidden), atol=1e-6)


def test_independent_trace_and_lower_drive_control(setup):
    spec, params, maps, obs = setup
    observations = jnp.broadcast_to(obs, (9, 1, 6))
    initial = models.init_carry(spec, 1)
    normal = audit.controlled_trace(spec, params, maps, initial, observations, "normal")
    carry, expected = initial, []
    for lag, observation in enumerate(observations):
        _, carry, _ = r.step(spec, params, carry, observation, lag, maps, 1, 2, float(lag == 0))
        expected.append(carry)
    np.testing.assert_allclose(normal["written"], expected, atol=1e-6)
    changed_lower = audit.controlled_trace(spec, params, maps, initial, observations, "same_top_with_lower_write")
    np.testing.assert_allclose(changed_lower["written"][0, -1], normal["written"][0, -1], atol=1e-6)
    assert not np.allclose(changed_lower["written"][0, 0], normal["written"][0, 0])
    reset = audit.controlled_trace(spec, params, maps, initial, observations, "drop_top_carry")
    np.testing.assert_allclose(reset["rho_diff"][1:], 0., atol=1e-6)
    np.testing.assert_allclose(normal["components"][1:].sum(-1), normal["rho_diff"][1:], atol=1e-6)


def test_independent_rho_target_matches_driver_including_out_of_simplex():
    beliefs = jnp.array([[.2, .3, .5], [.7, .5, -.2], [-.1, -.1, 1.2]])
    actual = audit.rho_target(beliefs)
    np.testing.assert_allclose(actual, core.target(beliefs, 2), atol=1e-6)
    np.testing.assert_allclose(actual[:, 2], 1-np.asarray(beliefs[:, 2]))
    np.testing.assert_allclose(audit.rho_target(beliefs, True)[:, 2], [.5, 1., 0.])


def test_episode_distribution_and_leave_one_episode_out():
    values = np.ones((3, 129, 32), np.float32)
    values[:, 4] = 3
    result = audit.describe(values)
    assert result["leave_one_of32_episodes_out_lag4_minus_lag1_range"] == [2., 2.]
    assert np.asarray(result["per_episode_absolute_mean_over_times_0_8"]).shape == (32, 9)
    assert result["lag_distribution"][4]["top5_of96_share"] == pytest.approx(5/96)
    assert result["lag_distribution"][4]["top5_of32_episodes_share"] == pytest.approx(5/32)
    assert audit.describe(np.zeros_like(values))["lag_distribution"][4]["top5_of96_share"] == 0.
