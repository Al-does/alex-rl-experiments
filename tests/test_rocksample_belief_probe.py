from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envs.rocksample import joint_from_marginals, update_joint

from experiments.rocksample_jax_2026_10 import belief_probe as bp
from experiments.rocksample_jax_2026_10 import env as rs
from experiments.rocksample_jax_2026_10 import model as tm
from experiments.rocksample_jax_2026_10 import ppo


def test_checkpoint_order_seed_identity_and_invalid_metadata(tmp_path):
    records = [
        {"update": 29, "env_steps": 29_360_128, "path": "d128_kl0.1/seed2.pkl"},
        {"update": 1, "env_steps": 1_048_576, "path": "d128_kl0.1/seed2/update001.pkl"},
    ]
    seed = {"seed_index": 2, "checkpoints": records}
    summary = {"arms": {"d128_kl0.1": {"seeds": [seed]}}}
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    assert [r["update"] for r in bp.checkpoint_records(tmp_path, 2)] == [1, 29]
    records[0]["path"] = "d128_kl0.1/seed1.pkl"
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="path disagrees"):
        bp.checkpoint_records(tmp_path, 2)
    records[0]["path"] = "d128_kl0.1/seed2.pkl"
    records[0]["env_steps"] = 1_048_576
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="increase strictly"):
        bp.checkpoint_records(tmp_path, 2)


@pytest.fixture(scope="module")
def collected():
    env = rs.RockSampleParams.from_instance(4, 4, episode_length=50)
    spec = tm.TransformerSpec(
        env.obs_dim, env.num_actions, d_model=8, n_layers=1, n_heads=2, context_len=4
    )
    params = bp.initialization(spec, 1, 42)
    rows = bp.collect(
        spec, env, params, params, episodes=8, key_seed=9131, scripted=True
    )
    return env, spec, params, rows, bp.targets(rows, env)


def test_exact_training_initialization_key():
    env = rs.RockSampleParams.from_instance()
    spec = tm.TransformerSpec(
        env.obs_dim, env.num_actions, d_model=8, n_layers=1, n_heads=2, context_len=4
    )
    config = ppo.PPOConfig(num_envs=1, num_steps=2, num_minibatches=1)
    for seed in range(4):
        state = ppo.init(
            config, env, spec, jax.random.split(jax.random.key(42), 4)[seed]
        )
        actual = bp.initialization(spec, seed, 42)
        for a, b in zip(
            jax.tree.leaves(actual), jax.tree.leaves(state.params), strict=True
        ):
            np.testing.assert_array_equal(a, b)


def test_decision_time_targets_and_terminal_exclusion(collected):
    env, _, _, rows, data = collected
    np.testing.assert_array_equal(data.marginals[data.times == 0], 0.5)
    assert data.exits.all()
    assert np.all(data.lengths < env.episode_length)
    np.testing.assert_array_equal(data.features, data.initialization)
    for episode in range(8):
        mask = data.groups == episode
        np.testing.assert_array_equal(
            data.times[mask], np.arange(data.lengths[episode])
        )
        j = np.full(2**env.k, 2.0**-env.k)
        for t in range(data.lengths[episode]):
            np.testing.assert_allclose(j, data.joint[mask][t], atol=1e-10)
            if not rows["terminated"][episode, t]:
                j = update_joint(
                    j,
                    rocks=np.asarray(env.rocks),
                    position=rows["position"][episode, t],
                    action=int(rows["action"][episode, t]),
                    symbol=int(rows["symbol"][episode, t]),
                    half_efficiency_distance=env.half_efficiency_distance,
                )
    np.testing.assert_allclose(data.joint, joint_from_marginals(data.marginals))
    assert np.all(data.samples == 1)
    assert np.all(data.final_belief == 0)


def test_privileged_truth_and_reward_cannot_change_targets(collected):
    env, _, _, rows, data = collected
    altered = {
        **rows,
        "qualities": ~rows["qualities"],
        "reward": np.full_like(rows["reward"], -999),
    }
    other = bp.targets(altered, env)
    np.testing.assert_array_equal(data.marginals, other.marginals)
    np.testing.assert_array_equal(data.joint, other.joint)
    np.testing.assert_array_equal(data.relevant, other.relevant)
    np.testing.assert_array_equal(data.nuisance, other.nuisance)


def test_cached_features_use_complete_episode_history(collected):
    _, spec, params, rows, _ = collected
    for episode in (0, 3):
        count = int(rows["valid"][episode].sum())
        features = tm.encode_window(
            spec,
            params,
            jnp.asarray(rows["obs"][episode, :count]),
            jnp.zeros(count, jnp.int32),
        )
        np.testing.assert_allclose(
            features, rows["features"][episode, :count], atol=3e-5
        )


def test_probe_fits_use_whole_episodes_and_constant_coordinates(collected):
    _, _, _, _, data = collected
    fit, test = bp.split(data)
    assert set(data.groups[fit]).isdisjoint(data.groups[test])
    report, predictions = bp.analyze(data, joint_battery=False)
    assert report["n_fit_episodes"] == report["n_test_episodes"] == 4
    assert report["validation"]["all_reset_targets_half"]
    assert predictions["final"].shape == data.marginals[test].shape
    constant = replace(data, marginals=np.full_like(data.marginals, 0.5))
    report, _ = bp.analyze(constant, joint_battery=False)
    assert report["marginals"]["final"]["metrics"]["r_squared"] is None


def test_scripted_sensor_histories_are_identical_across_encoders(collected):
    env, spec, params, rows, _ = collected
    other = tm.init_params(spec, jax.random.key(173))
    second = bp.collect(
        spec, env, other, params, episodes=8, key_seed=9131, scripted=True
    )
    for name in ("obs", "action", "symbol", "valid", "reward", "initialization"):
        np.testing.assert_array_equal(rows[name], second[name])
    assert not np.array_equal(rows["features"], second["features"])
