import numpy as np

from experiments.critic_aware_ladder_audit_2026_11.audit import probe
from experiments.critic_aware_ladder_audit_2026_11.soft import collect_prefix, replay_prefix, suffix

import jax
import jax.numpy as jnp

from experiments.belief_quotient_history_jax_2026_11 import env, models


def test_soft_suffix_excludes_current_entropy_and_stops_at_terminal():
    spec = models.ModelSpec("transformer", 3, 2, 4, 1, context_len=2)
    process = env.make_process(
        [1.0], [[[1.0]], [[1.0]]], [[1.0]], [0.5], [[0.5]]
    )
    actor = jax.tree.map(jnp.zeros_like, models.init_params(spec, jax.random.PRNGKey(0)))
    time, length, gamma = 2, 7, 0.8
    obs, belief = collect_prefix(spec, process, actor, jax.random.PRNGKey(1), 2, time, 1.5)
    carry, _ = replay_prefix(spec, actor, obs)
    values = suffix(spec, process, actor, carry, belief, jax.random.PRNGKey(2), 1.5, gamma,
                    time=time, length=length, replicates=4)
    discounts = gamma ** np.arange(length - time)
    np.testing.assert_allclose(values[0], (1 - gamma) * 0.5 * discounts.sum(), rtol=1e-6)
    np.testing.assert_allclose(values[1], (1 - gamma) * np.log(2) * discounts[1:].sum(), rtol=1e-6)


def test_forced_action_changes_the_next_reward_only():
    spec = models.ModelSpec("transformer", 4, 2, 4, 1, context_len=2)
    process = env.make_process(
        [1.0, 0.0], [[[1.0, 0.0]] * 2, [[0.0, 1.0]] * 2],
        np.eye(2), [0.0, 1.0], [[0.0, 1.0]],
    )
    actor = jax.tree.map(jnp.zeros_like, models.init_params(spec, jax.random.PRNGKey(0)))
    actor["policy"]["b"] = jnp.array([0.0, -200.0])
    gamma = 0.8
    obs, belief = collect_prefix(spec, process, actor, jax.random.PRNGKey(1), 1, 2, 1.5)
    carry, _ = replay_prefix(spec, actor, obs)
    reward, entropy = suffix(spec, process, actor, carry, belief, jax.random.PRNGKey(2), 1.5,
                             gamma, time=2, length=7, replicates=4)
    np.testing.assert_allclose(reward[:, 0, :], 0)
    np.testing.assert_allclose(reward[:, 1, :], (1 - gamma) * gamma, rtol=1e-6)
    np.testing.assert_allclose(entropy, 0, atol=1e-12)


def test_probe_uses_separate_denominators_for_rho_and_delta():
    beliefs = np.array([[0.6, 0.1, 0.3], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1], [0.1, 0.1, 0.8]])
    features = np.column_stack([beliefs[:, 2], beliefs[:, 0] - beliefs[:, 1]])
    result = probe([features, features], [beliefs, beliefs])
    np.testing.assert_allclose(result["sst"], np.var(features, axis=0) * len(features))
    assert max(result["error"]) < 1e-10
