from dataclasses import replace
import importlib
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_ppo_2026_11 import ppo
from experiments.belief_quotient_persistent_nuisance_jax_2026_11 import shared as base
from experiments.belief_quotient_ppo_separate_trunk_2026_11 import reference, shared, split_ppo
from experiments.belief_quotient_ppo_separate_trunk_2026_11.seed_queue import LEAVES


@pytest.mark.parametrize("leaf", LEAVES)
def test_leaf_wiring_and_counts(leaf):
    module = importlib.import_module(f"experiments.{shared.STUDY}.{leaf}.experiment")
    variant = 1 if leaf.startswith("trivial") else 3
    arm = leaf.removeprefix("trivial_" if variant == 1 else "full_state_")
    context = type("Context", (), {"smoke": False})()
    with (
        patch.object(jax, "default_backend", return_value="gpu"),
        patch.object(reference, "value_grid"),
        patch.object(shared.training, "run_study", side_effect=lambda *a, **kw: kw),
    ):
        result = module.run(context)
        historical = importlib.import_module(
            "experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_ppo_2026_11"
            f".variant_{variant}_transformer.experiment"
        )
        with patch.object(base, "run_study", side_effect=lambda *a, **kw: kw):
            old = historical.run(context)
    assert result["config"] == old["config"]
    assert result["config"].batch_steps == 19_922_944
    assert result["config"].iterations * result["config"].batch_steps == 2_012_217_344
    assert result["recipe_extra"]["arm"] == arm
    assert result["recipe_extra"]["variant"] == variant
    for field in old["process"]._fields:
        np.testing.assert_array_equal(getattr(result["process"], field), getattr(old["process"], field))
    expected = {"shared": 200_772, "shared_wide": 394_654, "separate": 401_284}
    init = split_ppo.init_params if arm == "separate" else ppo.init_params
    assert result["trainer"] is (split_ppo.train if arm == "separate" else ppo.train)
    assert models.parameter_count(init(result["spec"], jax.random.PRNGKey(42))) == expected[arm]


def test_split_preserves_initial_policy_and_isolates_gradients():
    spec = models.ModelSpec("transformer", 6, 3, 8, 1)
    key = jax.random.PRNGKey(42)
    params = split_ppo.init_params(spec, key)
    original = ppo.init_params(spec, key)
    obs = jax.random.normal(key, (1, 4, spec.obs_dim))
    logits, _ = split_ppo.forward(spec, params, obs)
    np.testing.assert_array_equal(logits, ppo.forward(spec, original, obs)[0])
    assert not np.array_equal(params["input"]["w"], params["critic"]["input"]["w"])

    config = ppo.TrainConfig(total_env_steps=4, num_envs=1)
    batch = dict(obs=obs, action=jnp.zeros((1, 4), dtype=int),
                 logp=jax.nn.log_softmax(logits / config.temperature)[..., 0],
                 advantages=jnp.ones((1, 4)), value_targets=jnp.zeros((1, 4)))

    def gradients(cfg, data):
        def objective(p):
            return ppo.loss(spec, cfg, p, data, 0., forward_fn=split_ppo.forward)[0]
        return jax.grad(objective)(params)

    policy = gradients(replace(config, vf_coeff=0.), batch)
    value = gradients(config, {**batch, "advantages": jnp.zeros((1, 4))})
    np.testing.assert_array_equal(policy["critic"]["input"]["w"], 0.)
    np.testing.assert_array_equal(value["input"]["w"], 0.)
    assert np.any(np.asarray(policy["input"]["w"]) != 0)
    assert np.any(np.asarray(value["critic"]["input"]["w"]) != 0)


def test_value_probe_targets_respect_terminal_and_decision_time():
    process = shared.environment.ladder_process(3)
    belief = np.array([[[.375, .3125, .3125], [.125, .1875, .6875]]])
    grid = reference.value_grid(process, gamma=.99, horizon=2, divisions=16)
    result = reference.value_target(grid, belief, divisions=16)[0, :, 0]
    next_reward = np.einsum("s,ast,t->a", belief[0, 0], process.transitions, process.reward)
    np.testing.assert_allclose(result, [.3125 + .99 * next_reward.max(), .6875], atol=1e-7)
