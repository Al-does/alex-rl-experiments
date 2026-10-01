from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envs.rocksample import RockSampleEnv

from experiments.rocksample_jax_2026_10 import env as rs
from experiments.rocksample_jax_2026_10 import model as tm
from experiments.rocksample_jax_2026_10 import ppo

PARAMS = rs.RockSampleParams.from_instance(5, 7)


def _jax_state_from(reference: RockSampleEnv) -> rs.EnvState:
    return rs.EnvState(
        rover=jnp.asarray(reference.rover_position, jnp.int32),
        qualities=jnp.asarray(reference.rock_qualities),
        symbol=jnp.int32(2),
        previous_action=jnp.int32(-1),
        step=jnp.int32(0),
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dynamics_and_observations_match_reference(seed):
    reference = RockSampleEnv({"diagnostics": True})
    ref_obs, _ = reference.reset(seed=seed)
    state = _jax_state_from(reference)
    np.testing.assert_allclose(rs.observation(PARAMS, state), ref_obs)
    rng = np.random.default_rng(seed)
    step = jax.jit(partial(rs.step, PARAMS))
    for t in range(100):
        # check actions are stochastic; replay the reference symbol instead
        action = int(rng.choice([0, 1, 2, 3, 4, 4, 0, 1, 3, 3, 5 + t % 7]))
        if action == 2 and reference.rover_position[0] == PARAMS.n - 1:
            action = 0
        ref_obs, ref_reward, ref_term, ref_trunc, info = reference.step(action)
        out = step(jax.random.key(t), state, jnp.int32(action))
        if action >= 5:
            out = out._replace(
                state=out.state._replace(symbol=jnp.int32(info["observation_symbol"]))
            )
            out = out._replace(obs=rs.observation(PARAMS, out.state))
        np.testing.assert_allclose(out.obs, ref_obs, err_msg=f"t={t}")
        assert float(out.reward) == ref_reward
        assert bool(out.terminated) == ref_term
        assert bool(out.truncated) == ref_trunc
        assert bool(out.illegal) == info["illegal_action"]
        np.testing.assert_array_equal(out.state.qualities, info["rock_qualities"])
        state = out.state
        if ref_term or ref_trunc:
            break


def test_exit_terminates_with_reward():
    _, state = rs.reset(PARAMS, jax.random.key(0))
    state = state._replace(rover=jnp.array([4, 2], jnp.int32))
    out = rs.step(PARAMS, jax.random.key(1), state, jnp.int32(2))
    assert bool(out.terminated) and float(out.reward) == 10.0


def test_sensor_accuracy_matches_efficiency():
    _, state = rs.reset(PARAMS, jax.random.key(0))
    rock = 6  # (3, 4), distance sqrt(13) from start (0, 2)
    keys = jax.random.split(jax.random.key(3), 20_000)
    outs = jax.vmap(lambda k: rs.step(PARAMS, k, state, jnp.int32(5 + rock)))(keys)
    truth = bool(state.qualities[rock])
    reports_good = np.asarray(outs.state.symbol == 0)
    accuracy = (reports_good == truth).mean()
    efficiency = 2.0 ** (-np.sqrt(13.0) / PARAMS.half_efficiency_distance)
    assert accuracy == pytest.approx((1 + efficiency) / 2, abs=0.01)


def test_autoreset_restarts_after_truncation():
    _, state = rs.reset(PARAMS, jax.random.key(0))
    state = state._replace(step=jnp.int32(99), rover=jnp.array([2, 2], jnp.int32))
    out = rs.step_autoreset(PARAMS, jax.random.key(1), state, jnp.int32(0))
    assert bool(out.truncated)
    assert int(out.state.step) == 0
    np.testing.assert_array_equal(out.state.rover, PARAMS.start)
    assert not np.allclose(out.final_obs, out.obs)


def test_cached_rollout_matches_windowed_training_path():
    spec = tm.TransformerSpec(obs_dim=PARAMS.obs_dim, num_actions=PARAMS.num_actions)
    params = tm.init_params(spec, jax.random.key(0))
    length = 140
    obs = jax.random.uniform(jax.random.key(1), (length, spec.obs_dim))
    episode = jnp.asarray(np.repeat([0, 1, 2], [30, 70, 40]), jnp.int32)
    cache = tm.empty_cache(spec)
    cached = []
    for t in range(length):
        if t > 0 and episode[t] != episode[t - 1]:
            cache = tm.empty_cache(spec)
        embedding, cache = tm.encode_step(spec, params, cache, obs[t])
        cached.append(embedding)
    windowed = tm.encode_window(spec, params, obs, episode)
    np.testing.assert_allclose(np.stack(cached), windowed, atol=2e-5)
    # truncated lookback window (as used in training) is still exact
    start = length - spec.lookback - 20
    tail = tm.encode_window(spec, params, obs[start:], episode[start:])
    np.testing.assert_allclose(tail[spec.lookback :], windowed[-20:], atol=2e-5)


def test_gae_bootstraps_truncation_and_zeroes_termination():
    config = ppo.PPOConfig(
        num_envs=1, num_steps=3, num_minibatches=1, gamma=0.5, gae_lambda=1.0
    )
    zeros = jnp.zeros((1, 3))
    traj = ppo.Transition(
        obs=None,
        episode_id=None,
        action=None,
        log_prob=None,
        value=jnp.array([[1.0, 2.0, 3.0]]),
        reward=jnp.array([[1.0, 1.0, 1.0]]),
        terminated=jnp.array([[False, True, False]]),
        truncated=jnp.array([[True, False, False]]),
        bootstrap_value=jnp.array([[4.0, 0.0, 0.0]]),
        episode_return=zeros,
        illegal=zeros,
    )
    advantages, _ = ppo.gae(config, traj, jnp.array([6.0]))
    np.testing.assert_allclose(advantages, [[1 + 0.5 * 4 - 1, 1 - 2, 1 + 0.5 * 6 - 3]])


def test_training_chunk_runs_and_is_vmappable():
    config = ppo.PPOConfig(num_envs=4, num_steps=16, num_minibatches=2)
    spec = ppo.make_spec(PARAMS, d_model=16)
    chunk = jax.jit(jax.vmap(ppo.make_train_chunk(config, PARAMS, spec, num_updates=2)))
    states = jax.vmap(partial(ppo.init, config, PARAMS, spec))(
        jax.random.split(jax.random.key(0), 2)
    )
    states, metrics = chunk(states)
    assert metrics["env_steps"].shape == (2, 2)
    assert int(metrics["env_steps"][0, -1]) == 2 * config.batch_size
    assert np.isfinite(np.asarray(metrics["policy_loss"])).all()
