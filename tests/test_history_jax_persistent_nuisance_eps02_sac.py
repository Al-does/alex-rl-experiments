import importlib
from dataclasses import asdict, replace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_history_jax_2026_11.training import _save
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_sac_2026_11 import sac

LADDER = "belief_quotient_persistent_nuisance_eps02_jax_temp15"
FACTORS = "reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32"
CASES = [(LADDER, f"variant_{v}_transformer") for v in (1, 2, 3)] + [
    (FACTORS, f"{c}_transformer") for c in ("reward_both", "reward_factor_1")]


@pytest.mark.parametrize("study,leaf", CASES)
def test_leaf_wiring(study, leaf):
    shared = importlib.import_module(f"experiments.{study}_sac_2026_11.shared")
    module = importlib.import_module(f"experiments.{study}_sac_2026_11.{leaf}.experiment")
    parent = importlib.import_module(f"experiments.{study}_2026_11.{leaf}.experiment")
    for smoke in (False, True):
        context = type("Context", (), {"smoke": smoke})()
        with patch.object(shared.base, "run_study", side_effect=lambda *a, **kw: kw):
            reference = parent.run(context)
            result = module.run(context)
            after = parent.run(context)
        config = result["config"]
        assert asdict(reference["config"]) == asdict(after["config"])
        assert {k: asdict(config)[k] for k in asdict(reference["config"])} == asdict(reference["config"])
        assert config.algorithm == "SAC" and config.alpha == 0.
        assert not config.auto_entropy_tuning
        assert config.tau == .005 and config.gamma == .99
        assert config.gradient_steps == 64 and config.replay_rollouts == 4
        assert config.lr == config.critic_lr == 4.2e-4
        assert config.temperature == 1.5 and config.exploration_epsilon == 0.
        assert config.entropy_schedule == ((0, 0.),)
        assert config.num_envs == 19456 and config.num_minibatches == 64
        assert config.episode_length == 1024
        assert config.total_env_steps == (4096 if smoke else 2_000_000_000)
        assert result["spec"] == reference["spec"]
        assert result["representation"] == reference["representation"]
        assert result["study"] == shared.STUDY and callable(result["trainer"])
        for name in result["process"]._fields:
            np.testing.assert_array_equal(getattr(result["process"], name),
                                          getattr(reference["process"], name))
    with pytest.raises(ValueError):
        shared.run_condition(context, architecture="invalid",
            **({"variant": 1} if study == LADDER else {"condition": "reward_both"}))


@pytest.mark.parametrize("alpha", [0., .2])
@pytest.mark.parametrize("length", [0, 1, 5])
def test_critic_target_numpy(alpha, length):
    rng = np.random.default_rng(42)
    rewards = rng.normal(size=(2, length)).astype(np.float32)
    q1, q2 = rng.normal(size=(2, 2, length, 3)).astype(np.float32)
    probabilities = rng.dirichlet(np.ones(3), size=(2, length)).astype(np.float32)
    logs = np.log(probabilities)
    expected = rewards.copy()
    for b in range(2):
        for t in range(length - 1):
            expected[b, t] += .99 * np.sum(probabilities[b, t + 1] *
                (np.minimum(q1[b, t + 1], q2[b, t + 1]) - alpha * logs[b, t + 1]))
    actual = sac.critic_targets(jnp.array(rewards), jnp.array(logs), jnp.array(q1),
                                 jnp.array(q2), .99, alpha)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    if length:
        np.testing.assert_array_equal(actual[:, -1], rewards[:, -1])
    gradient = jax.grad(lambda q: sac.critic_targets(
        jnp.array(rewards), jnp.array(logs), q, jnp.array(q2), .99, alpha).sum())(jnp.array(q1))
    np.testing.assert_array_equal(gradient, 0.)


@pytest.mark.parametrize("alpha", [0., .2])
def test_actor_numpy_and_critic_stop_gradient(alpha):
    probabilities = np.array([[[.2, .8], [.6, .4]]], np.float32)
    q1 = jnp.array([[[2., 3.], [-1., 4.]]])
    q2 = jnp.array([[[1., 5.], [2., 3.]]])
    logs = jnp.log(probabilities)
    expected = (probabilities * (alpha * np.log(probabilities) -
                                np.minimum(q1, q2))).sum(-1).mean()
    np.testing.assert_allclose(sac.actor_objective(logs, q1, q2, alpha), expected, atol=1e-6)
    gradient = jax.grad(lambda q: sac.actor_objective(logs, q, q2, alpha))(q1)
    np.testing.assert_array_equal(gradient, 0.)


def test_polyak():
    old = ({"w": jnp.array([1., 2.])}, {"w": jnp.array([-1., 4.])})
    new = jax.tree.map(lambda x: x + 10, old)
    result = sac.polyak_update(old, new, .005)
    for before, after in zip(jax.tree.leaves(old), jax.tree.leaves(result)):
        np.testing.assert_allclose(after, before + .05, atol=1e-6)


def episode_records(ids, length=4):
    ids = np.asarray(ids)
    tokens = (ids[:, None] + np.arange(length)[None]) % 9
    actions = (2 * ids[:, None] + np.arange(length)[None]) % 9
    previous = np.concatenate([np.full((len(ids), 1), -1), actions[:, :-1]], 1)
    obs = jnp.concatenate([jax.nn.one_hot(tokens, 9), jax.nn.one_hot(previous, 9)], -1)
    return {"obs": obs, "action": jnp.array(actions),
            "reward": jnp.broadcast_to(jnp.array(ids, jnp.float32)[:, None], (len(ids), length))}


def test_buffer_whole_episodes_wrap_and_reconstruct():
    config = sac.TrainConfig(total_env_steps=8, num_envs=2, episode_length=4, replay_rollouts=2)
    buffer = sac.init_buffer(config)
    assert int(buffer["size"]) == 0
    assert sum(x.nbytes for x in jax.tree.leaves(buffer)) == sac.buffer_bytes(config)
    for ids in ([0, 1], [2, 3], [4, 5]):
        buffer = jax.jit(sac.insert_buffer, static_argnums=2)(buffer, episode_records(ids), 9)
        expected_ids = [0, 1] if ids == [0, 1] else ([0, 1, 2, 3] if ids == [2, 3] else [4, 5, 2, 3])
        np.testing.assert_array_equal(buffer["reward"][:len(expected_ids), 0], expected_ids)
        sample = sac.sample_buffer(buffer, jax.random.PRNGKey(5), 256, 9, 9)
        sampled_ids = np.asarray(sample["reward"][:, 0]).astype(int)
        assert set(sampled_ids) == set(expected_ids)
        expected = episode_records(sampled_ids)
        for name in expected:
            np.testing.assert_array_equal(sample[name], expected[name])
    assert int(buffer["position"]) == 2 and int(buffer["size"]) == 4
    full = sac.TrainConfig(total_env_steps=2_000_000_000, num_envs=19456, num_minibatches=64)
    assert sac.buffer_bytes(full) == 478_150_664


def test_training_checkpoint_layout_and_scaled_q(tmp_path):
    process = environment.ladder_process(1)
    spec = models.ModelSpec("transformer", process.obs_dim, process.n_actions, 8, 1)
    config = sac.TrainConfig(total_env_steps=8, num_envs=2, episode_length=4,
                            num_minibatches=2, gradient_steps=2, temperature=1.5)
    parent = models.init_params(spec, jax.random.fold_in(jax.random.PRNGKey(42), 0))
    _save(tmp_path / "parent.npz", parent)
    captured = {}

    def save(step, actor):
        captured[step] = jax.device_get(actor)
        _save(tmp_path / f"step_{step:010d}.npz", actor)

    actor, curves, summary = sac.train(spec, process, config, seed=42,
        on_checkpoint=save, critic_checkpoint_dir=tmp_path)
    expected = {"actor_loss": 33.019309997558594, "critic_loss": .5497674942016602,
                "mean_q": -43.139869689941406, "policy_entropy": 1.0397520065307617,
                "reward_per_step": 0.}
    for name, value in expected.items():
        np.testing.assert_allclose(curves[-1][name], value, rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(actor["policy"]["b"],
        [.0693768858909607, .283753901720047, -.08408339321613312], atol=1e-7)
    assert "alpha" not in curves[-1] and "mean_policy_q" not in curves[-1]
    assert sorted(captured) == [0, 8]
    with np.load(tmp_path / "parent.npz") as reference:
        for step in captured:
            with np.load(tmp_path / f"step_{step:010d}.npz") as checkpoint:
                assert set(checkpoint.files) == set(reference.files)
                for name in reference.files:
                    assert checkpoint[name].shape == reference[name].shape
                    if step == 0:
                        np.testing.assert_array_equal(checkpoint[name], reference[name])
    with np.load(tmp_path / "critic_step_0000000008.npz") as checkpoint:
        assert any("targets" in name for name in checkpoint.files)
        assert any("critics" in name for name in checkpoint.files)
        assert not any("alpha" in name for name in checkpoint.files)
    for name in ("actor_loss", "critic_loss", "mean_q", "policy_entropy"):
        assert np.isfinite(curves[-1][name])
    assert summary["env_steps"] == 8
    obs = jnp.zeros((2, 4, process.obs_dim))
    logits, _, _ = models.sequence(spec, actor, obs)
    np.testing.assert_allclose(sac.q_values(spec, actor, obs, .99), logits / .01, atol=1e-5)
    q = jnp.array([[[1., 2., 3.]]])
    fixed = {"obs": obs, "action": jnp.zeros((2, 4), jnp.int32)}
    values = sac.q_values(spec, actor, obs, .99)[..., 0]
    np.testing.assert_allclose(sac.critic_loss(spec, config, (actor, actor), fixed, values),
                               0., atol=1e-12)
    with pytest.raises(ValueError, match="fresh"):
        sac.train(spec, process, config, seed=42, initial_params=parent)
    with pytest.raises(ValueError, match="exploration"):
        sac.train(spec, process, replace(config, exploration_epsilon=.1), seed=42)


@pytest.mark.parametrize("kwargs", [
    {"gamma": 1.}, {"tau": 0.}, {"replay_rollouts": 0},
    {"gradient_steps": 0}, {"auto_entropy_tuning": True}, {"alpha": -.1},
])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        sac.TrainConfig(total_env_steps=8, **kwargs)
