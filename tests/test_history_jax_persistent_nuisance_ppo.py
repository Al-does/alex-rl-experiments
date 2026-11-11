from dataclasses import asdict, replace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.belief_quotient_history_jax_2026_11 import models, reinforce
from experiments.belief_quotient_persistent_nuisance_jax_2026_11 import shared as old_ladder
from experiments.belief_quotient_persistent_nuisance_ppo_jax_2026_11 import ppo
from experiments.belief_quotient_persistent_nuisance_ppo_jax_2026_11 import shared as ladder
from experiments.reward_factors_persistent_nuisance_jax_2026_11 import shared as old_factors
from experiments.reward_factors_persistent_nuisance_ppo_jax_2026_11 import shared as factors

ARCHITECTURES = ("transformer", "memoryless", "gru128", "gru92")


def test_gae_matches_reference_loop():
    rng = np.random.default_rng(0)
    rewards, values = rng.random((3, 7)), rng.random((3, 7))
    bootstrap = rng.random(3)
    gamma, lam = .9, .8
    advantages, targets = ppo.gae(jnp.asarray(rewards), jnp.asarray(values),
                                  jnp.asarray(bootstrap), gamma, lam)
    expected = np.zeros_like(rewards)
    for b in range(3):
        running = 0.
        for t in reversed(range(7)):
            following = values[b, t + 1] if t + 1 < 7 else bootstrap[b]
            running = rewards[b, t] + gamma * following - values[b, t] + gamma * lam * running
            expected[b, t] = running
    np.testing.assert_allclose(advantages, expected, atol=1e-6)
    np.testing.assert_allclose(targets, expected + values, atol=1e-6)
    # lambda = 1 with a zero critic is the truncated discounted reward-to-go.
    advantages, _ = ppo.gae(jnp.asarray(rewards), jnp.zeros((3, 7)), jnp.zeros(3), gamma, 1.)
    np.testing.assert_allclose(advantages, reinforce.discounted_returns(jnp.asarray(rewards), gamma),
                               atol=1e-6)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_rollout_matches_actor_and_training_forward(architecture):
    process = ladder.ladder_process(2)
    spec = ladder.model_spec(architecture, process)
    key = jax.random.PRNGKey(3)
    params = ppo.init_params(spec, key)
    actor = models.init_params(spec, key)
    assert set(params) - set(actor) == {"value"}
    for leaf, reference in zip(jax.tree_util.tree_leaves({k: params[k] for k in actor}),
                               jax.tree_util.tree_leaves(actor)):
        np.testing.assert_array_equal(leaf, reference)
    rollout_key = jax.random.PRNGKey(4)
    records = ppo.rollout(spec, process, params, rollout_key, 3, 64)
    reference = reinforce.rollout(spec, process, actor, rollout_key, 3, 64)
    np.testing.assert_array_equal(records["action"], reference["action"])
    np.testing.assert_allclose(records["logp"], reference["logp"], atol=1e-5)
    _, _, embedding = models.sequence(spec, params, records["obs"], records.get("chunk_starts"))
    np.testing.assert_allclose(ppo.value(params, embedding), records["value"], atol=1e-4)
    assert records["bootstrap_value"].shape == (3,)
    assert np.isfinite(records["bootstrap_value"]).all()


def test_ppo_coefficients_and_inherited_geometry():
    for smoke in (False, True):
        config, old = ladder.train_config(smoke), old_ladder.train_config(smoke)
        for name in ("total_env_steps", "num_envs", "episode_length", "num_minibatches",
                     "gamma", "clip_param", "temperature"):
            assert getattr(config, name) == getattr(old, name)
        assert old.entropy_schedule == ((0, config.entropy_coeff),)
        assert (config.num_epochs, config.lr, config.gae_lambda, config.vf_coeff,
                config.max_grad_norm, config.adam_eps) == (4, 3e-4, .95, .5, .5, 1e-5)
    assert ladder.TOTAL_ENV_STEPS == 200_000_000


@pytest.mark.parametrize("new,old,condition", [
    (ladder, old_ladder, 2), (factors, old_factors, "reward_both"),
])
def test_process_model_and_metadata_inherited(new, old, condition):
    is_ladder = new is ladder
    make = new.ladder_process if is_ladder else new.factor_process
    reference = old.ladder_process if is_ladder else old.factor_process
    for name, value in make(condition)._asdict().items():
        np.testing.assert_array_equal(value, getattr(reference(condition), name))
    process = make(condition)
    for architecture in new.ARCHITECTURES:
        assert new.model_spec(architecture, process) == old.model_spec(architecture, process)
        with patch.object(new, "run_study", side_effect=lambda *args, **kwargs: kwargs):
            kwargs = {"variant" if is_ladder else "condition": condition}
            result = new.run_condition(type("Context", (), {"smoke": False})(),
                                       architecture=architecture, **kwargs)
        assert result["recipe_extra"]["source_study"] == old.STUDY
        assert result["recipe_extra"]["environment_revision"] == "persistent_nuisance_v1"
        assert asdict(result["config"]) == asdict(ladder.train_config(False))
        assert result["representation"] == ("embedding" if is_ladder else "residual")


def test_single_iteration_updates_actor_and_critic():
    process = ladder.ladder_process(2)
    spec = ladder.model_spec("memoryless", process)
    config = replace(ladder.train_config(True), num_envs=4, num_minibatches=2,
                     episode_length=64, total_env_steps=256)
    seen = {}
    _, curves, summary = ppo.train(spec, process, config, seed=0,
                                   on_checkpoint=lambda steps, params: seen.update({steps: params}))
    assert sorted(seen) == [0, 256]
    assert summary["adam_steps"] == config.num_epochs * config.num_minibatches
    for name in ("policy", "value"):
        assert not np.allclose(seen[0][name]["w"], seen[256][name]["w"])
    record = curves[-1]
    for name in ("vf_loss", "policy_loss", "approx_kl_mean", "explained_variance", "grad_norm"):
        assert np.isfinite(record[name])
