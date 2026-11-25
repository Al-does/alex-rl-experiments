"""Shared-trunk PPO with terminal-episode GAE and a scaled linear critic."""
from dataclasses import asdict, dataclass
from functools import partial

import jax
import jax.numpy as jnp

from experiments.belief_quotient_history_jax_2026_11 import models, reinforce


@dataclass(frozen=True)
class TrainConfig(reinforce.TrainConfig):
    algorithm: str = "PPO"
    num_epochs: int = 4
    gae_lambda: float = 0.95
    vf_coeff: float = 0.5
    value_scaling: str = "1-gamma"
    value_clip: bool = False


def configure(config):
    return TrainConfig(**{**asdict(config), "num_epochs": 4})


def init_params(spec, key):
    params = models.init_params(spec, key)
    params["value"] = models._linear(jax.random.fold_in(key, 17), spec.width, 1)
    return params


def gae(rewards, values, gamma, gae_lambda):
    next_values = jnp.concatenate([values[:, 1:], jnp.zeros_like(values[:, :1])], axis=1)
    deltas = rewards + gamma * next_values - values

    def body(future, delta):
        advantage = delta + gamma * gae_lambda * future
        return advantage, advantage

    _, advantages = jax.lax.scan(body, jnp.zeros(rewards.shape[0]), deltas.T, reverse=True)
    return advantages.T


def scaled_value_loss(predictions, targets, gamma):
    return jnp.mean((predictions - (1 - gamma) * targets) ** 2)


def forward(spec, params, obs):
    logits, _, embedding = models.sequence(spec, params, obs)
    return logits, models._dense(params["value"], embedding)[..., 0]


def prepare_batch(spec, config, params, records, *, forward_fn=forward):
    obs = records["obs"]

    def predict(observations):
        _, predictions = forward_fn(spec, params, observations)
        return predictions / (1 - config.gamma)

    chunks = obs.reshape(config.num_minibatches, -1, *obs.shape[1:])
    values = jax.lax.map(predict, chunks).reshape(obs.shape[:2])
    advantages = gae(records["reward"], values, config.gamma, config.gae_lambda)
    targets = advantages + values
    return jax.lax.stop_gradient({
        "advantages": (advantages - advantages.mean()) / jnp.maximum(1e-4, advantages.std()),
        "value_targets": targets,
    })


def loss(spec, config, params, batch, entropy_coeff, *, forward_fn=forward):
    logits, predictions = forward_fn(spec, params, batch["obs"])
    behaviour = reinforce.behaviour_log_probs(
        logits, config.temperature, config.exploration_epsilon)
    logp = jnp.take_along_axis(behaviour, batch["action"][..., None], -1)[..., 0]
    ratio = jnp.exp(logp - batch["logp"])
    clipped = jnp.clip(ratio, 1 - config.clip_param, 1 + config.clip_param)
    policy_loss = -jnp.minimum(batch["advantages"] * ratio,
                               batch["advantages"] * clipped).mean()
    log_probs = jax.nn.log_softmax(logits / config.temperature)
    entropy = -(jnp.exp(log_probs) * log_probs).sum(-1).mean()
    vf_loss = scaled_value_loss(predictions, batch["value_targets"], config.gamma)
    targets = batch["value_targets"]
    variance = jnp.var(targets)
    explained_variance = jnp.where(
        variance > 0, 1 - jnp.var(targets - predictions / (1 - config.gamma)) /
        jnp.maximum(variance, 1e-12), 0.)
    return policy_loss + config.vf_coeff * vf_loss - entropy_coeff * entropy, {
        "policy_loss": policy_loss, "entropy": entropy,
        "vf_loss": vf_loss, "explained_variance": explained_variance,
    }


def make_iteration(spec, process, config, optimizer, *, forward_fn=forward):
    return reinforce.make_iteration(
        spec, process, config, optimizer, loss_fn=partial(loss, forward_fn=forward_fn),
        prepare_batch=partial(prepare_batch, forward_fn=forward_fn))


train = partial(reinforce.train, init_fn=init_params, iteration_fn=make_iteration)
