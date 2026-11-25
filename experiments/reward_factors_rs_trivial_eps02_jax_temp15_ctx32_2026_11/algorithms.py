"""Matched trainers: actor-only, shared-trunk critic, separate-trunk critic.

All three use the same clipped surrogate, four shuffled epochs over 64
whole-episode minibatches, Adam 4.2e-4, temperature 1.5, zero entropy and
batch-standardized advantages. They differ only in where advantages come from
and what the trunk is trained on:

- ``actor_only``: Monte Carlo discounted reward-to-go advantages, no value head.
- ``shared_critic``: the existing PPO (GAE 0.95, linear value head on the
  policy trunk's post-final-LayerNorm embedding, 0.5 x scaled MSE).
- ``separate_critic``: identical GAE/value loss, but the value head reads an
  independent transformer trunk of the same architecture, so the policy trunk
  never receives value-loss gradients.
"""
from dataclasses import asdict
from functools import partial

import jax
import jax.numpy as jnp

from experiments.belief_quotient_history_jax_2026_11 import models, reinforce
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_ppo_2026_11 import ppo

ARMS = ("actor_only", "shared_critic", "separate_critic")
NUM_EPOCHS = 4


class _Backend:
    def __init__(self, name, train, config_cls, extra):
        self.name, self.train, self._cls, self.extra = name, train, config_cls, extra

    def configure(self, config):
        return self._cls(**{**asdict(config), "num_epochs": NUM_EPOCHS})


def separate_init(spec, key):
    params = models.init_params(spec, key)
    critic = models.init_params(spec, jax.random.fold_in(key, 23))
    del critic["policy"]
    critic["value"] = models._linear(jax.random.fold_in(key, 17), spec.width, 1)
    params["critic"] = critic
    return params


def critic_values(spec, params, obs):
    critic = {**params["critic"], "policy": params["policy"]}  # policy head unused here
    _, _, embedding = models.sequence(spec, critic, obs)
    return models._dense(params["critic"]["value"], embedding)[..., 0]


def separate_prepare_batch(spec, config, params, records):
    obs = records["obs"]
    chunks = obs.reshape(config.num_minibatches, -1, *obs.shape[1:])
    values = jax.lax.map(lambda o: critic_values(spec, params, o) / (1 - config.gamma),
                         chunks).reshape(obs.shape[:2])
    advantages = ppo.gae(records["reward"], values, config.gamma, config.gae_lambda)
    return jax.lax.stop_gradient({
        "advantages": (advantages - advantages.mean()) / jnp.maximum(1e-4, advantages.std()),
        "value_targets": advantages + values,
    })


def separate_loss(spec, config, params, batch, entropy_coeff):
    actor = {k: v for k, v in params.items() if k != "critic"}
    policy_loss, info = reinforce.surrogate_loss(spec, config, actor, batch, entropy_coeff)
    critic = {**params["critic"], "policy": jax.lax.stop_gradient(params["policy"])}
    _, _, embedding = models.sequence(spec, critic, batch["obs"])
    predictions = models._dense(params["critic"]["value"], embedding)[..., 0]
    vf_loss = ppo.scaled_value_loss(predictions, batch["value_targets"], config.gamma)
    targets = batch["value_targets"]
    variance = jnp.var(targets)
    explained_variance = jnp.where(
        variance > 0, 1 - jnp.var(targets - predictions / (1 - config.gamma))
        / jnp.maximum(variance, 1e-12), 0.)
    return policy_loss + config.vf_coeff * vf_loss, {
        **info, "vf_loss": vf_loss, "explained_variance": explained_variance}


def _separate_iteration(spec, process, config, optimizer):
    return reinforce.make_iteration(spec, process, config, optimizer, loss_fn=separate_loss,
                                    prepare_batch=separate_prepare_batch)


BACKENDS = {
    "actor_only": _Backend("actor_only", reinforce.train, reinforce.TrainConfig,
                           {"advantages": "standardized Monte Carlo discounted reward-to-go",
                            "critic": None}),
    "shared_critic": _Backend("shared_critic", ppo.train, ppo.TrainConfig,
                              {"advantages": "standardized GAE(0.95)",
                               "critic": "linear head on the policy trunk's post-final-LayerNorm embedding"}),
    "separate_critic": _Backend(
        "separate_critic",
        partial(reinforce.train, init_fn=separate_init, iteration_fn=_separate_iteration),
        ppo.TrainConfig,
        {"advantages": "standardized GAE(0.95)",
         "critic": "linear head on an independent 64x4 transformer trunk (no shared parameters)"}),
}
