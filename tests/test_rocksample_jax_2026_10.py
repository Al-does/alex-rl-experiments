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


# --- LPO drift + ES meta-loop -------------------------------------------------


def test_initial_lpo_objective_is_exactly_ppo():
    import numpy as np

    from experiments.rocksample_jax_2026_10 import lpo
    from experiments.rocksample_jax_2026_10 import ppo as jppo

    spec = lpo.DriftSpec()
    meta = jnp.asarray(spec.initial_params(seed=3), jnp.float32)
    ratio = jnp.tile(jnp.linspace(0.3, 2.5, 101), 3)
    advantages = jnp.repeat(jnp.array([-1.7, 0.4, 2.2]), 101)

    lpo_surrogate, lpo_entropy = lpo.objective(spec)(meta, ratio, advantages)
    ppo_surrogate, ppo_entropy = jppo.clipped_objective(jppo.PPOConfig())(
        meta, ratio, advantages
    )
    np.testing.assert_allclose(lpo_surrogate, ppo_surrogate, atol=1e-6)
    np.testing.assert_allclose(lpo_entropy, ppo_entropy, rtol=1e-6)


def test_perturbed_lpo_drift_is_nonnegative_and_zero_at_unit_ratio():
    import numpy as np

    from experiments.rocksample_jax_2026_10 import lpo

    spec = lpo.DriftSpec()
    meta = spec.initial_params(seed=0) + np.random.default_rng(1).normal(
        0, 0.5, spec.num_params
    )
    meta = jnp.asarray(meta, jnp.float32)
    ratio = jnp.linspace(0.2, 3.0, 57)
    advantages = jnp.linspace(-3.0, 3.0, 57)

    assert bool(jnp.all(lpo.drift(spec, meta, ratio, advantages) >= 0))
    np.testing.assert_allclose(
        lpo.drift(spec, meta, jnp.ones(57), advantages), 0.0, atol=1e-6
    )


def test_es_step_moves_toward_better_antithetic_member():
    import numpy as np

    from experiments.rocksample_jax_2026_10.es import ESState, OpenES

    es = OpenES(num_pairs=1, sigma=0.1, learning_rate=0.05)
    state = ESState(mean=np.zeros(2))
    noise = np.array([[1.0, -1.0]])
    np.testing.assert_allclose(es.candidates(state, noise), [[0.1, -0.1], [-0.1, 0.1]])
    stepped = es.step(state, noise, np.array([5.0, 1.0]))
    assert stepped.generation == 1
    np.testing.assert_allclose(stepped.mean, [0.05, -0.05], rtol=1e-6)


def test_meta_fitness_and_inner_seeds():
    import math

    from experiments.rocksample_jax_2026_10 import meta

    assert (
        meta.fitness_from_history([0.0, math.nan, 10.0, 20.0, 30.0, 40.0], 0.4) == 35.0
    )
    assert math.isnan(meta.fitness_from_history([], 0.2))
    assert meta.inner_seed(42, 3, 1) != meta.inner_seed(42, 3, 2)
    assert meta.inner_seed(42, 3, 1) != meta.inner_seed(42, 4, 1)


def test_full_meta_recipe_budget(tmp_path):
    from harness.context import RunContext

    from experiments.rocksample_jax_2026_10.lpo_es_drift import experiment

    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "r",
        artifacts_dir=tmp_path / "a",
    )
    inner = experiment.inner_recipe(context)
    assert inner.ppo.batch_size == 8_192
    assert inner.ppo.num_minibatches == 8
    assert inner.num_updates == 367
    assert experiment.meta_recipe(context).es.num_pairs == 8


def test_meta_smoke_runs_population_and_advances_es(tmp_path):
    import json

    import numpy as np
    from harness.context import RunContext

    from experiments.rocksample_jax_2026_10 import lpo, meta
    from experiments.rocksample_jax_2026_10 import ppo as jppo
    from experiments.rocksample_jax_2026_10.es import OpenES

    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        smoke=True,
    )
    inner = meta.InnerRecipe(
        ppo=jppo.PPOConfig(num_envs=4, num_steps=128, num_minibatches=2, num_epochs=1),
        total_env_steps=2_048,
        d_model=16,
    )
    drift = lpo.DriftSpec(hidden=4)
    recipe = meta.MetaRecipe(
        drift=drift,
        es=OpenES(num_pairs=1, sigma=0.05, learning_rate=0.02),
        generations=2,
        fitness_tail_fraction=0.5,
    )
    state = meta.run_meta_training(context, inner, recipe)

    assert state.generation == 2
    rows = [
        json.loads(line)
        for line in (context.results_dir / meta.META_PROGRESS_FILENAME)
        .read_text()
        .splitlines()
    ]
    assert [row["generation"] for row in rows] == [0, 1]
    assert all(
        row["inner_runs"] == 3 and row["inner_env_steps"] == 2_048 for row in rows
    )
    assert all(np.isfinite(row["fitness"]).all() for row in rows)


def test_ppo_baseline_recipe_matches_rllib_budget(tmp_path):
    from harness.context import RunContext

    from experiments.rocksample_jax_2026_10.ppo_n5_k7_d64 import experiment

    context = RunContext(
        experiment_dir=tmp_path,
        results_dir=tmp_path / "r",
        artifacts_dir=tmp_path / "a",
    )
    config = experiment.ppo_config(context)
    assert config.batch_size == 8_192
    assert config.batch_size // config.num_minibatches == 1_024
    steps = experiment.UPDATES_PER_CHUNK * experiment.NUM_CHUNKS * config.batch_size
    assert 5_000_000 <= steps < 5_100_000
