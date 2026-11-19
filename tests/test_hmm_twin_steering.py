import jax
import jax.numpy as jnp
import numpy as np

from experiments.algorithm_generality_probes_eps02_2026_11 import steer_core as old_ladder
from experiments.algorithm_generality_probes_eps02_2026_11 import steer_moving as old_xor
from experiments.algorithm_generality_probes_eps02_2026_11 import steer_moving_base as xor_base
from experiments.belief_quotient_history_jax_2026_11 import env, models
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
from experiments.belief_quotient_twin_steering_2026_11 import ladder, twins
from experiments.reward_factors_twin_steering_2026_11 import xor


def test_edit_keeps_actions_and_untargeted_factor():
    rng = np.random.default_rng(2)
    tokens = rng.integers(9, size=(2, 72))
    actions = np.eye(9)[rng.integers(9, size=tokens.shape)]
    actions[:, 0] = 0
    obs = np.concatenate((np.eye(9)[tokens], actions), axis=-1)
    alternatives = list(twins.edit_histories(obs, environment.emission_matrix(), 2, seed=0))
    assert len(alternatives) == 6
    for index, twin in enumerate(alternatives):
        np.testing.assert_array_equal(twin[..., 9:], actions)
        token = twin[..., :9].argmax(-1)
        assert np.any(token != tokens)
        if index < 2:
            np.testing.assert_array_equal(token % 3, tokens % 3)
        if 2 <= index < 4:
            np.testing.assert_array_equal(token // 3, tokens // 3)


def test_replay_filters_before_decision_and_reconstructs_xor_rollout():
    process = old_xor.source_process(next(iter(old_xor.SOURCE_STUDIES)))
    spec = models.ModelSpec("transformer", 18, 9, 8, 4)
    params = models.init_params(spec, jax.random.PRNGKey(3))
    key = xor_base._keys(4)[0]
    record = xor_base.collect(spec, params, process, 2, 12, key)
    obs = xor.observations(process, record["action"], key)
    h, b = twins.replay(spec, params, process, obs)
    np.testing.assert_allclose(b, record["belief"], atol=1e-7)
    np.testing.assert_allclose(h, record["sites"][:, :, 3], atol=1e-6)
    alternative, *_ = twins.edit_histories(obs, environment.emission_matrix(), 2, 10)
    _, b = twins.replay(spec, params, process, alternative)
    tokens = alternative[0, :, :9].argmax(-1)
    prior = env.belief_reset(process, tokens[0])
    np.testing.assert_allclose(b[0, 0], prior)
    for t in range(1, 12):
        prior = env.belief_update(process, prior, record["action"][0, t - 1], tokens[t])
        np.testing.assert_allclose(b[0, t], prior, atol=1e-7)


def test_both_write_directions_and_norm_matched_control():
    rng = np.random.default_rng(11)
    belief, target = rng.uniform(size=(2, 20, 2))
    basis = rng.normal(size=(2, 5))
    basis[:, 2:] = 0
    origin = rng.normal(size=(20, 5))
    origin[:, :2] = 0
    h, twin_h = origin + belief @ basis, origin + target @ basis
    write = twins.fit_write(h, belief, twin_h, target)
    for x, goal, expected in [(h, target, twin_h - h), (twin_h, belief, h - twin_h)]:
        np.testing.assert_allclose(twins.displacement(x, goal, write), expected, atol=1e-10)
        direction = np.eye(5)[0]
        null = twins.scaled_shift(x, goal, write, 1., True, direction)
        np.testing.assert_allclose(np.linalg.norm(null, axis=1), np.linalg.norm(expected, axis=1), rtol=1e-6)


def test_ladder_lambda_zero_matches_archived_actions_and_rewards():
    spec = models.ModelSpec("transformer", 6, 3, 8, 4)
    process = environment.ladder_process(2)
    params = models.init_params(spec, jax.random.PRNGKey(42))
    maps = old_ladder.empty_maps(spec.width)
    key = jax.random.PRNGKey(2042)
    old = old_ladder.rollout(spec, process, params, maps, key, 2, 64)
    maps["write"] = dict(residual=jnp.eye(8), target=jnp.zeros((2, 8)), offset=jnp.ones(8))
    new = ladder.rollout(spec, process, params, maps, key, 2, 64, 2, 0.)
    for name in ("action", "reward", "changed"):
        np.testing.assert_array_equal(new[name], old[name])


def test_xor_lambda_zero_and_archived_xor_schedule(monkeypatch):
    monkeypatch.setattr(xor_base, "HORIZON", 32)
    monkeypatch.setattr(old_xor, "HORIZON", 32)
    spec = models.ModelSpec("transformer", 18, 9, 8, 4)
    process = old_xor.source_process(next(iter(old_xor.SOURCE_STUDIES)))
    params = models.init_params(spec, jax.random.PRNGKey(42))
    q = jnp.asarray(old_xor.q_tables("xor", "discounted"), jnp.float32)
    schedule = jnp.asarray(old_xor.reward_schedule("xor"), jnp.float32)
    reward = (np.arange(9) // 3 == 2) != (np.arange(9) % 3 == 2)
    np.testing.assert_array_equal(schedule[:, 0], np.tile(reward, (32, 1)))
    key = xor_base._keys(2042)[0]
    old = old_xor._evaluate(spec, params, process, 2, "agent", 3, "joint", (True, True), False,
                           {}, q, schedule, key)
    writer = dict(decoder=jnp.zeros((8, 8)), bias=jnp.zeros(8),
                  anchors=jnp.tile(jnp.arange(3), (2, 1)), null=jnp.eye(8)[0],
                  write=dict(residual=jnp.eye(8), target=jnp.zeros((8, 8)), offset=jnp.ones(8)))
    new = xor.rollout(spec, params, process, 2, "twin", "joint", writer, q, schedule, key, 0.)
    for name in old:
        np.testing.assert_array_equal(new[name], old[name])
