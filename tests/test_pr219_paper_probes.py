import numpy as np
import pytest
import jax

from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_eps05_2026_11 import paper_probes as p


def observations(tokens, actions):
    return np.concatenate([np.eye(3)[tokens], np.eye(3)[actions]], axis=-1)


def test_exact_quotient_equals_full_reward_marginal():
    rng = np.random.default_rng(31)
    obs = observations(rng.integers(3, size=(4, 1024)), rng.integers(3, size=(4, 1024)))
    process = p.environment.ladder_process(2)
    full = p.replay_filter(obs, process)
    coarse = p.replay_filter(obs, process, coarse=True)
    np.testing.assert_allclose(full[..., 2], coarse[..., 1], atol=1e-14, rtol=1e-13)
    np.testing.assert_allclose(full.sum(-1), 1)


def test_full_state_condition_rejects_exact_two_state_quotient():
    obs = observations(np.zeros((2, 10), dtype=int), np.zeros((2, 10), dtype=int))
    with pytest.raises(ValueError, match="no exact"):
        p.replay_filter(obs, p.environment.ladder_process(3), coarse=True)


def test_filter_conditions_on_current_token_and_previous_action():
    process = p.environment.ladder_process(2)
    obs = observations(np.array([[0, 2]]), np.array([[0, 1]]))
    out = p.replay_filter(obs, process)[0]
    initial = np.asarray(process.initial, dtype=float) * np.asarray(process.emission, dtype=float)[:, 0]
    initial /= initial.sum()
    transition = np.asarray(process.transitions[1], dtype=float)
    transition /= transition.sum(-1, keepdims=True)
    expected = (initial @ transition) * np.asarray(process.emission, dtype=float)[:, 2]
    expected /= expected.sum()
    np.testing.assert_allclose(out[0], initial)
    np.testing.assert_allclose(out[1], expected)


@pytest.mark.parametrize("kind", ["shuffle", "random_input"])
def test_controls_preserve_valid_pairs_and_reset_semantics(kind):
    rng = np.random.default_rng(5)
    obs = observations(rng.integers(3, size=(3, 30)), rng.integers(3, size=(3, 30)))
    obs[:, 0, 3:] = 0
    changed = p.control_inputs(obs, kind, rng)
    np.testing.assert_equal(changed[:, 0, 3:], 0)
    np.testing.assert_equal(changed[:, 1:, 3:].sum(-1), 1)
    if kind == "shuffle":
        np.testing.assert_equal(changed[:, 0], obs[:, 0])
        for before, after in zip(obs[:, 1:], changed[:, 1:]):
            assert sorted(map(tuple, before)) == sorted(map(tuple, after))


def test_ab_randomization_preserves_c_and_actions():
    rng = np.random.default_rng(9)
    obs = observations(rng.integers(3, size=(4, 1000)), rng.integers(3, size=(4, 1000)))
    changed, flips, eligible = p.ab_inputs(obs, rng)
    np.testing.assert_equal(changed[..., 3:], obs[..., 3:])
    np.testing.assert_equal(changed[..., 2], obs[..., 2])
    assert .45 < flips.sum() / eligible.sum() < .55


def test_ridge_scores_known_affine_target():
    rng = np.random.default_rng(42)
    xx = rng.normal(size=(1000, 5))
    yy = xx @ rng.normal(size=(5, 3)) + 2
    metrics, _ = p.score(xx[:500], xx[500:], yy[:500], yy[500:])
    assert metrics["global_mse_ratio"] < 1e-12


def test_episode_interval_is_reproducible():
    assert p.episode_interval([1, 2, 3]) == p.episode_interval([1, 2, 3])


@pytest.mark.parametrize("architecture", p.ARCHITECTURES)
def test_full_history_replay_matches_real_rollout(architecture):
    process = p.environment.ladder_process(2)
    spec = p.models.architecture_spec(architecture, obs_dim=6, n_actions=3,
                                     n_tokens=3, context_len=32)
    params = p.models.init_params(spec, jax.random.PRNGKey(3))
    records = p.collect(spec, process, params, jax.random.PRNGKey(4), 2, 256)
    logits, embedding = p.replay(spec, params, records["obs"])
    np.testing.assert_allclose(embedding, records["embedding"], atol=1e-6)
    np.testing.assert_equal(np.asarray(logits).argmax(-1), records["action"])


def test_missing_or_corrupt_geometry_invalidates_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "HERE", tmp_path)
    geometry = tmp_path / "simplex.npz"
    geometry.write_bytes(b"geometry")
    cache = tmp_path / "checkpoint.json"
    report = {"geometry": {"file": geometry.name, "sha256": p.sha256(geometry)}}
    p.write_json(cache, report)
    assert p.cached_report(cache) == report
    geometry.write_bytes(b"corrupt")
    assert p.cached_report(cache) is None
    geometry.unlink()
    assert p.cached_report(cache) is None


def test_intermediate_report_cache_needs_no_geometry(tmp_path):
    cache = tmp_path / "checkpoint.json"
    assert p.cached_report(cache) is None
    p.write_json(cache, {"checkpoint_step": 0})
    assert p.cached_report(cache) == {"checkpoint_step": 0}
