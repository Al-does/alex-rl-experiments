import importlib
import math
from dataclasses import asdict
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_history_jax_2026_11.training import _save
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_sac_2026_11 import sac
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_sac_autoalpha_2026_11 import shared as auto

LADDER = "belief_quotient_persistent_nuisance_eps02_jax_temp15"
FACTORS = "reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32"
CASES = [(LADDER, f"variant_{v}_transformer") for v in (1, 2, 3)] + [
    (FACTORS, f"{c}_transformer") for c in ("reward_both", "reward_factor_1")]


@pytest.mark.parametrize("study,leaf", CASES)
def test_leaf_wiring_and_legacy_config(study, leaf):
    shared = importlib.import_module(f"experiments.{study}_sac_autoalpha_2026_11.shared")
    module = importlib.import_module(f"experiments.{study}_sac_autoalpha_2026_11.{leaf}.experiment")
    legacy = importlib.import_module(f"experiments.{study}_sac_2026_11.{leaf}.experiment")
    for smoke in (False, True):
        context = type("Context", (), {"smoke": smoke})()
        with patch.object(shared.base, "run_study", side_effect=lambda *a, **kw: kw):
            before = legacy.run(context)
            result = module.run(context)
            after = legacy.run(context)
        assert asdict(before["config"]) == asdict(after["config"])
        assert asdict(result["config"]) == {
            **asdict(before["config"]), **auto.hyperparameters(result["spec"].n_actions)}
        config = result["config"]
        assert config.auto_entropy_tuning and config.alpha_init == .01
        assert result["spec"].n_actions == (3 if study == LADDER else 9)
        assert config.target_entropy == .3 * math.log(result["spec"].n_actions)
        assert config.alpha_lr == config.critic_lr == 3e-4 and config.lr == 3e-5
        assert config.q_head_init == "zeros"
        assert config.temperature == 1.5 and config.gamma == .99 and config.tau == .005
        assert config.num_envs == 19456 and config.num_minibatches == 64
        assert config.gradient_steps == 64 and config.replay_rollouts == 4
        assert config.episode_length == 1024
        assert config.total_env_steps == (4096 if smoke else 2_000_000_000)
        assert result["spec"] == before["spec"]
        assert result["representation"] == before["representation"]
        assert result["study"] == shared.STUDY and callable(result["trainer"])
        for name in result["process"]._fields:
            np.testing.assert_array_equal(getattr(result["process"], name),
                                          getattr(before["process"], name))
    assert before["config"].alpha == 0. and not before["config"].auto_entropy_tuning
    assert before["config"].q_head_init == "default"
    assert before["config"].lr == before["config"].critic_lr == 4.2e-4
    with pytest.raises(ValueError, match="transformer"):
        shared.run_condition(context, architecture="invalid",
            **({"variant": 1} if study == LADDER else {"condition": "reward_both"}))


@pytest.mark.parametrize("probabilities,direction", [([.99, .005, .005], 1), ([1/3]*3, -1)])
def test_alpha_gradient_direction_and_detachment(probabilities, direction):
    log_probs = jnp.log(jnp.array([[probabilities]]))
    log_alpha = jnp.array(.2)
    target = auto.hyperparameters(log_probs.shape[-1])["target_entropy"]
    grad = jax.grad(sac.alpha_loss)(log_alpha, log_probs, target)
    optimizer = optax.adam(3e-4)
    updates, _ = optimizer.update(grad, optimizer.init(log_alpha), log_alpha)
    updated = optax.apply_updates(log_alpha, updates)
    assert direction * (jnp.exp(updated) - jnp.exp(log_alpha)) > 0
    np.testing.assert_array_equal(jax.grad(lambda lp: sac.alpha_loss(
        log_alpha, lp, target))(log_probs), 0.)


def test_critic_initialization_opt_in():
    process = environment.ladder_process(1)
    spec = models.ModelSpec("transformer", process.obs_dim, process.n_actions, 8, 1)
    config = sac.TrainConfig(total_env_steps=8)
    obs = jax.random.normal(jax.random.PRNGKey(7), (2, 4, process.obs_dim))
    for index in (3, 4):
        key = jax.random.fold_in(jax.random.PRNGKey(42), index)
        reference = models.init_params(spec, key)
        default = sac.init_critic_params(spec, key, config)
        zero = sac.init_critic_params(spec, key, sac.configure(config, **auto.hyperparameters(spec.n_actions)))
        for expected, actual in zip(jax.tree.leaves(reference), jax.tree.leaves(default)):
            np.testing.assert_array_equal(actual, expected)
        for name in reference:
            if name != "policy":
                for expected, actual in zip(jax.tree.leaves(reference[name]), jax.tree.leaves(zero[name])):
                    np.testing.assert_array_equal(actual, expected)
        for value in jax.tree.leaves(zero["policy"]):
            np.testing.assert_array_equal(value, 0.)
        np.testing.assert_array_equal(sac.q_values(spec, zero, obs, .99), 0.)


def test_autoalpha_training_metrics_and_checkpoint(tmp_path):
    process = environment.ladder_process(1)
    spec = models.ModelSpec("transformer", process.obs_dim, process.n_actions, 8, 1)
    config = sac.configure(sac.TrainConfig(total_env_steps=8, num_envs=2, episode_length=4,
        num_minibatches=2, gradient_steps=2, temperature=1.5), **auto.hyperparameters(spec.n_actions))
    saved = {}

    def save(step, params):
        saved[step] = jax.device_get(params)
        _save(tmp_path / f"step_{step}.npz", params)

    actor, curves, summary = sac.train(spec, process, config, seed=42,
        on_checkpoint=save, critic_checkpoint_dir=tmp_path)
    assert sorted(saved) == [0, 8] and summary["env_steps"] == 8
    with np.load(tmp_path / "step_0.npz") as before, np.load(tmp_path / "step_8.npz") as after:
        assert set(before.files) == set(after.files)
        assert not any("alpha" in key for key in after.files)
    metrics = curves[-1]
    for name in ("alpha", "log_alpha", "alpha_loss", "entropy", "mean_policy_q",
                 "mean_q", "critic_loss", "actor_loss"):
        assert np.isfinite(metrics[name])
    assert 0 < metrics["alpha"] <= config.alpha_init
    assert metrics["entropy_coeff"] == metrics["alpha"]
    assert metrics["entropy"] == metrics["policy_entropy"]
    with np.load(tmp_path / "critic_step_0000000008.npz") as checkpoint:
        assert any("targets" in key for key in checkpoint.files)
        assert any("critics" in key for key in checkpoint.files)
        assert 0 < checkpoint["['alpha']"] < config.alpha_init
        np.testing.assert_allclose(checkpoint["['alpha']"], np.exp(checkpoint["['log_alpha']"]))
        assert checkpoint["['alpha_optimizer'][0].count"] == config.gradient_steps
    obs = jnp.zeros((2, 4, spec.obs_dim))
    q1 = jnp.broadcast_to(jnp.array([-2., 1., 4.]), (2, 4, 3))
    q2 = q1 + 1.
    loss, info = sac.actor_loss(spec, config, actor, obs, q1, q2, alpha=jnp.array(.7))
    logits, _, _ = models.sequence(spec, actor, obs)
    logs = jax.nn.log_softmax(logits / 1.5)
    np.testing.assert_allclose(loss, sac.actor_objective(logs, q1, q2, .7))
    np.testing.assert_allclose(info["mean_policy_q"], (jnp.exp(logs) * q1).sum(-1).mean())
    np.testing.assert_allclose(info["mean_q"], q1.mean())


@pytest.mark.parametrize("overrides", [
    {"target_entropy": None}, {"target_entropy": -1.}, {"target_entropy": float("nan")},
    {"alpha_init": 0.}, {"alpha_init": -1.}, {"alpha_lr": 0.},
    {"alpha_lr": float("inf")}, {"q_head_init": "unknown"},
])
def test_invalid_autoalpha_config(overrides):
    with pytest.raises(ValueError):
        sac.TrainConfig(total_env_steps=8, **{**auto.hyperparameters(3), **overrides})
