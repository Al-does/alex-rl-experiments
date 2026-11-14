"""Softmax temperature in the JAX trainer: sampling, likelihoods, and the temp15 recipes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_history_jax_2026_11.reinforce import TrainConfig, rollout, surrogate_loss
from experiments.belief_quotient_history_jax_lumped_2026_11 import shared as lumped
from experiments.belief_quotient_history_jax_lumped_temp15_2026_11 import shared as belief15
from experiments.reward_factors_history_jax_lumped_temp15_2026_11 import shared as reward15


def _setup(arch="memoryless"):
    process = lumped.ladder_process(2)
    spec = lumped.model_spec(arch, process)
    params = models.init_params(spec, jax.random.PRNGKey(0))
    return process, spec, params


def test_tempered_logp_matches_scaled_logits_and_greedy_is_invariant():
    process, spec, params = _setup()
    key = jax.random.PRNGKey(1)
    hot = rollout(spec, process, params, key, 8, 16, temperature=1.5)
    cold = rollout(spec, process, params, key, 8, 16, temperature=1.0)
    # same key: identical obs streams at t=0, logp differs by the temperature
    logits, *_ = models.step(spec, params, models.init_carry(spec, 8), hot["obs"][0], 0)
    want = jnp.take_along_axis(jax.nn.log_softmax(logits / 1.5), hot["action"][0][:, None], -1)[:, 0]
    np.testing.assert_allclose(hot["logp"][0], want, rtol=1e-5, atol=1e-5)
    assert float(hot["entropy"].mean()) > float(cold["entropy"].mean())
    g_hot = rollout(spec, process, params, key, 8, 16, greedy=True, temperature=1.5)
    g_cold = rollout(spec, process, params, key, 8, 16, greedy=True)
    np.testing.assert_array_equal(g_hot["action"], g_cold["action"])


def test_surrogate_ratio_is_one_under_matching_temperature():
    process, spec, params = _setup()
    config = TrainConfig(total_env_steps=256, num_envs=8, episode_length=16, temperature=1.5)
    records = rollout(spec, process, params, jax.random.PRNGKey(2), 8, 16, temperature=1.5)
    batch = {"obs": records["obs"], "action": records["action"], "logp": records["logp"],
             "advantages": jnp.ones_like(records["logp"])}
    _, info = surrogate_loss(spec, config, params, batch, 0.0)
    assert info["policy_loss"] == pytest.approx(-1.0, abs=1e-5)  # ratio == 1 everywhere
    mismatched = TrainConfig(total_env_steps=256, num_envs=8, episode_length=16, temperature=1.0)
    _, info2 = surrogate_loss(spec, mismatched, params, batch, 0.0)
    assert abs(float(info2["policy_loss"]) + 1.0) > 1e-3


def test_default_temperature_is_identity():
    assert TrainConfig(total_env_steps=1).temperature == 1.0
    with pytest.raises(ValueError):
        TrainConfig(total_env_steps=1, temperature=0.0)


@pytest.mark.parametrize("pkg", [belief15, reward15])
def test_temp15_recipe_constants(pkg):
    assert pkg.TOTAL_ENV_STEPS == 4_000_000_000
    assert pkg.ENTROPY_SCHEDULE == ((0, 0.0),)
    assert pkg.TEMPERATURE == 1.5


def test_epsilon_mixture_logp_and_on_policy_ratio():
    from experiments.belief_quotient_history_jax_2026_11.reinforce import behaviour_log_probs

    logits = jnp.array([[2.0, -1.0, 0.5]])
    want = 0.95 * jax.nn.softmax(logits / 1.5) + 0.05 / 3
    np.testing.assert_allclose(jnp.exp(behaviour_log_probs(logits, 1.5, 0.05)), want, rtol=1e-6)
    np.testing.assert_array_equal(behaviour_log_probs(logits, 1.5, 0.0), jax.nn.log_softmax(logits / 1.5))
    process, spec, params = _setup()
    config = TrainConfig(total_env_steps=256, num_envs=8, episode_length=16, temperature=1.5,
                         exploration_epsilon=0.05)
    records = rollout(spec, process, params, jax.random.PRNGKey(3), 8, 16, temperature=1.5, epsilon=0.05)
    logits0, *_ = models.step(spec, params, models.init_carry(spec, 8), records["obs"][0], 0)
    expected = jnp.take_along_axis(behaviour_log_probs(logits0, 1.5, 0.05), records["action"][0][:, None], -1)[:, 0]
    np.testing.assert_allclose(records["logp"][0], expected, rtol=1e-5, atol=1e-5)
    batch = {"obs": records["obs"], "action": records["action"], "logp": records["logp"],
             "advantages": jnp.ones_like(records["logp"])}
    _, info = surrogate_loss(spec, config, params, batch, 0.0)
    assert info["policy_loss"] == pytest.approx(-1.0, abs=1e-5)
    with pytest.raises(ValueError):
        TrainConfig(total_env_steps=1, exploration_epsilon=1.0)


def test_zero_epsilon_rollout_is_unchanged():
    process, spec, params = _setup()
    key = jax.random.PRNGKey(4)
    a = rollout(spec, process, params, key, 8, 16, temperature=1.5)
    b = rollout(spec, process, params, key, 8, 16, temperature=1.5, epsilon=0.0)
    np.testing.assert_array_equal(a["action"], b["action"])


def test_eps05_recipe_constants():
    from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_eps05_2026_11 import shared as eps05

    assert eps05.EXPLORATION_EPSILON == 0.05
    assert eps05.TOTAL_ENV_STEPS == 2_000_000_000
    assert eps05.ENTROPY_SCHEDULE == ((0, 0.0),) and eps05.TEMPERATURE == 1.5


@pytest.mark.parametrize("variant", (1, 2, 3))
@pytest.mark.parametrize("architecture", ("transformer", "memoryless", "gru128", "gru92"))
def test_eps05_leaves_pass_epsilon(variant, architecture):
    import importlib
    from unittest.mock import patch

    from experiments.belief_quotient_persistent_nuisance_jax_2026_11 import shared as base

    study = "belief_quotient_persistent_nuisance_v2_jax_temp15_eps05_2026_11"
    module = importlib.import_module(f"experiments.{study}.variant_{variant}_{architecture}.experiment")
    with patch.object(base, "run_study", side_effect=lambda *a, **kw: kw):
        result = module.run(type("Context", (), {"smoke": False})())
    assert result["architecture"] == architecture and result["study"] == study
    config = result["config"]
    assert (config.exploration_epsilon, config.temperature, config.total_env_steps) == (0.05, 1.5, 2_000_000_000)
    assert config.entropy_schedule == ((0, 0.0),)

