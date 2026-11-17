import importlib
from dataclasses import asdict
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_ppo_2026_11 import ppo

LADDER = "belief_quotient_persistent_nuisance_eps02_jax_temp15"
FACTORS = "reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32"
CASES = [(LADDER, f"variant_{v}_transformer") for v in (1, 2, 3)] + [
    (FACTORS, f"{c}_transformer") for c in ("reward_both", "reward_factor_1")]


@pytest.mark.parametrize("study,leaf", CASES)
def test_leaf_wiring_and_parent_unchanged(study, leaf):
    shared = importlib.import_module(f"experiments.{study}_ppo_2026_11.shared")
    module = importlib.import_module(f"experiments.{study}_ppo_2026_11.{leaf}.experiment")
    parent = importlib.import_module(f"experiments.{study}_2026_11.{leaf}.experiment")
    for smoke in (False, True):
        context = type("Context", (), {"smoke": smoke})()
        with patch.object(shared.base, "run_study", side_effect=lambda *a, **kw: kw):
            before = parent.run(context)
            result = module.run(context)
            after = parent.run(context)
        assert asdict(before["config"]) == asdict(after["config"])
        assert before["config"].num_epochs == 1
        config = result["config"]
        expected = asdict(config)
        for key in ("algorithm", "gae_lambda", "vf_coeff", "value_scaling", "value_clip"):
            expected.pop(key)
        expected["num_epochs"] = 1
        assert expected == asdict(before["config"])
        assert config.num_epochs == 4 and config.gae_lambda == .95
        assert config.gamma == .99 and config.vf_coeff == .5
        assert not config.value_clip and config.algorithm == "PPO"
        assert config.entropy_schedule == ((0, 0.),)
        assert config.temperature == 1.5 and config.exploration_epsilon == 0
        assert config.num_envs == 19456 and config.num_minibatches == 64
        assert config.total_env_steps == (4096 if smoke else 2_000_000_000)
        assert result["trainer"] is ppo.train
        assert result["spec"] == before["spec"]
        assert result["representation"] == before["representation"]
        for name in before["process"]._fields:
            np.testing.assert_array_equal(getattr(result["process"], name),
                                          getattr(before["process"], name))
    with pytest.raises(ValueError):
        shared.run_condition(context, architecture="invalid",
            **({"variant": 1} if study == LADDER else {"condition": "reward_both"}))


@pytest.mark.parametrize("length", [0, 1, 7])
def test_gae_numpy(length):
    rng = np.random.default_rng(8)
    rewards = rng.normal(size=(3, length)).astype(np.float32)
    values = rng.normal(size=(3, length)).astype(np.float32)
    expected = np.zeros_like(rewards)
    future = np.zeros(3)
    for t in reversed(range(length)):
        next_value = values[:, t + 1] if t + 1 < length else 0.
        future = rewards[:, t] + .99 * next_value - values[:, t] + .99 * .95 * future
        expected[:, t] = future
    np.testing.assert_allclose(ppo.gae(jnp.array(rewards), jnp.array(values), .99, .95),
                               expected, atol=1e-6)


def test_scaled_critic_and_shared_embedding():
    spec = models.ModelSpec("transformer", 6, 3, 8, 1)
    key = jax.random.PRNGKey(42)
    params = ppo.init_params(spec, key)
    parent = models.init_params(spec, key)
    for a, b in zip(jax.tree.leaves(parent),
                    jax.tree.leaves({k: v for k, v in params.items() if k != "value"})):
        np.testing.assert_array_equal(a, b)
    assert params["value"]["w"].shape == (8, 1)
    assert np.max(np.abs(params["value"]["w"])) <= 1 / np.sqrt(8)
    np.testing.assert_allclose(ppo.scaled_value_loss(jnp.array([1., 2.]),
                                                    jnp.array([100., 200.]), .99), 0.,
                               atol=1e-12)
    records = {"obs": jnp.zeros((2, 4, 6)), "reward": jnp.ones((2, 4))}
    config = ppo.TrainConfig(total_env_steps=8, num_envs=2)
    batch = ppo.prepare_batch(spec, config, params, records)
    _, _, embedding = models.sequence(spec, params, records["obs"])
    values = models._dense(params["value"], embedding)[..., 0] / .01
    advantages = ppo.gae(records["reward"], values, .99, .95)
    np.testing.assert_allclose(batch["value_targets"], values + advantages, atol=1e-5)
    np.testing.assert_allclose(batch["advantages"].mean(), 0., atol=1e-6)
    np.testing.assert_allclose(batch["advantages"].std(), 1., atol=1e-6)
