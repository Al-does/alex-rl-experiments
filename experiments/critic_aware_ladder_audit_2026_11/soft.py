"""Monte Carlo soft Q^pi with forced first actions and a frozen SAC policy."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from experiments.belief_quotient_history_jax_2026_11 import env, models, reinforce


@partial(jax.jit, static_argnums=(0, 4, 5))
def collect_prefix(spec, process, actor, key, episodes, time, temperature):
    records = reinforce.rollout(spec, process, actor, key, episodes, time + 1,
                                track_belief=True, temperature=temperature)
    return records["obs"], records["belief"][:, -1]


@partial(jax.jit, static_argnums=(0,))
def replay_prefix(spec, params, obs):
    def step(carry, t):
        logits, _, _, next_carry = models.step(spec, params, carry, obs[:, t], t)
        return next_carry, logits

    final_carry, logits = jax.lax.scan(step, models.init_carry(spec, len(obs)),
                                       jnp.arange(obs.shape[1]))
    return final_carry, logits[-1]


def entropy_terms(logits, temperature):
    logp = jax.nn.log_softmax(logits / temperature)
    probs = jnp.exp(logp)
    return probs, -jnp.sum(probs * logp, axis=-1)


def draw_with_uniform(probs, uniform):
    # The final bin absorbs rounding error in the preceding cumulative sum.
    return jnp.sum(uniform[..., None] >= jnp.cumsum(probs, axis=-1)[..., :-1], axis=-1)


@partial(jax.jit, static_argnames=("spec", "time", "length", "replicates"))
def suffix(spec, process, actor, carry, belief, key, temperature, gamma, *,
           time, length, replicates):
    """Return scaled reward and future entropy by prefix, forced action, replica.

    Sample tokens directly from their predictive distribution and update the
    posterior, integrating the hidden state out of the per-step reward. Reuse
    the same random numbers across first actions for lower-variance contrasts.
    """
    episodes, actions = len(belief), spec.n_actions
    rows = (episodes, actions, replicates)
    n = episodes * actions * replicates
    start_action = jnp.broadcast_to(jnp.arange(actions)[None, :, None], rows).reshape(n)
    belief = jnp.repeat(belief, actions * replicates, axis=0)
    carry = jax.tree.map(lambda x: jnp.repeat(x, actions * replicates, axis=1), carry)
    zero = jnp.zeros(n, dtype=belief.dtype)

    def common_uniform(draw_key):
        return jnp.broadcast_to(
            jax.random.uniform(draw_key, (episodes, 1, replicates)), rows).reshape(n)

    def advance(previous, t):
        carry, belief, action, reward_sum, entropy_sum, key = previous
        key, token_key, action_key = jax.random.split(key, 3)
        predicted = jnp.einsum("bi,bij->bj", belief, process.transitions[action])
        token = draw_with_uniform(predicted @ process.emission, common_uniform(token_key))
        posterior = predicted * process.emission[:, token].T
        posterior = posterior / posterior.sum(axis=1, keepdims=True)
        observation = jnp.concatenate([
            jax.nn.one_hot(token, process.n_tokens), jax.nn.one_hot(action, actions),
        ], axis=-1)
        logits, _, _, carry = models.step(spec, actor, carry, observation, t)
        pi, entropy = entropy_terms(logits, temperature)
        next_action = draw_with_uniform(pi, common_uniform(action_key))
        discount = gamma ** (t - time)
        reward_sum = reward_sum + discount * (posterior @ process.reward)
        entropy_sum = entropy_sum + discount * entropy
        return (carry, posterior, next_action, reward_sum, entropy_sum, key), None

    initial = (carry, belief, start_action, belief @ process.reward, zero, key)
    final, _ = jax.lax.scan(advance, initial, jnp.arange(time + 1, length))
    return (1 - gamma) * jnp.stack([final[3].reshape(rows), final[4].reshape(rows)])


def score(prediction, estimate_a, estimate_b):
    """Direct held-out scores, with independent split covariance as a noise check."""
    p, a, b = (np.asarray(x, dtype=np.float64) for x in (prediction, estimate_a, estimate_b))
    average = (a + b) / 2
    noise = float(np.mean(np.square(a - b)) / 4)
    observed_var = float(np.mean(np.square(average - average.mean(axis=0))))
    # Correct finite-sample bias in the centered denominator for the noise estimate.
    signal_var = observed_var - (1 - 1 / len(average)) * noise
    error = float(np.mean(np.square(p - average)))
    return {
        "r2": 1 - error / observed_var if observed_var > 0 else None,
        "noise_adjusted_r2": 1 - (error - noise) / signal_var if signal_var > 0 else None,
        "bias": float(np.mean(p - average)),
        "rmse": float(np.sqrt(error)),
        "conditional_target_estimated_variance": signal_var,
        "mc_mean_estimated_noise_variance": noise,
    }


def evaluate(spec, process, trunks, temperature, gamma, alpha, seed, length, smoke):
    episodes, replicates = (6, 4) if smoke else (128, 48)
    time = length // 2
    key = jax.random.fold_in(jax.random.PRNGKey(seed), 59231)
    prefix_key, first_key, second_key = jax.random.split(key, 3)
    actor = trunks["sac_autoalpha"]
    obs, belief = collect_prefix(spec, process, actor, prefix_key, episodes, time, temperature)
    carry, actor_logits = replay_prefix(spec, actor, obs)
    pi, entropy = entropy_terms(actor_logits, temperature)
    predicted = {
        name: np.asarray(replay_prefix(spec, trunks[name], obs)[1]) for name in ("q1", "q2")
    }
    predicted["min_q"] = np.minimum(predicted["q1"], predicted["q2"])
    current_entropy_bonus = np.asarray((1 - gamma) * alpha * entropy)
    pi = np.asarray(pi)

    def policy_mean(q):
        return (pi * q).sum(axis=-1)

    averages = []
    for suffix_key in (first_key, second_key):
        sums = suffix(spec, process, actor, carry, belief, suffix_key, temperature, gamma,
                      time=time, length=length, replicates=replicates // 2)
        averages.append(np.asarray(sums).mean(axis=-1, dtype=np.float64))
    (reward_a, entropy_a), (reward_b, entropy_b) = averages
    qa, qb = reward_a + alpha * entropy_a, reward_b + alpha * entropy_b
    va, vb = policy_mean(qa) + current_entropy_bonus, policy_mean(qb) + current_entropy_bonus

    def contrasts(q):
        return q - policy_mean(q)[:, None]

    return {
        "prefix_episodes": episodes, "prefix_timestep": time,
        "episode_length": length, "replicates_per_action": replicates,
        "discount": gamma, "temperature": temperature, "alpha": alpha,
        "units": "(1-gamma) times return",
        "mean_policy_probability_by_action": pi.mean(0).tolist(),
        "mean_soft_q_by_action": ((qa + qb) / 2).mean(0).tolist(),
        "mean_reward_only_q_by_action": ((reward_a + reward_b) / 2).mean(0).tolist(),
        "mean_soft_v": float(((va + vb) / 2).mean()),
        "q": {name: score(p, qa, qb) for name, p in predicted.items()},
        "q_while_omitting_entropy": {
            name: score(p, reward_a, reward_b) for name, p in predicted.items()
        },
        "policy_relative_q": {
            name: score(contrasts(p), contrasts(qa), contrasts(qb)) for name, p in predicted.items()
        },
        "v_from_min_q": score(policy_mean(predicted["min_q"]) + current_entropy_bonus, va, vb),
        "v_from_min_q_while_omitting_entropy_in_target": score(
            policy_mean(predicted["min_q"]) + current_entropy_bonus,
            policy_mean(reward_a), policy_mean(reward_b)),
    }
