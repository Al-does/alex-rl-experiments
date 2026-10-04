"""JAX port of the two Pusher-B transformers.

Both legacy models share one block stack: pre-RMSNorm, causal RoPE attention
(interleaved pairs, base 10k) with biased qkv/output, gated-GELU MLP
(``gelu(gate(x)) * up(x)``), final RMSNorm, N(0, 0.02) weights and zero
biases. ``supervised_model.PusherBTransformer`` embeds tokens {0, 1, BOS} and
unembeds with a bias-free linear (``lm`` head). The RLlib
``FactoredReproductionActorCritic`` embeds one-hot observations with a
bias-free linear plus a learned BOS vector for the all-zero first
observation, which is exactly a 3-row embedding table; its policy/value (and
optional next-token aux) heads use PyTorch's default Linear init.

:func:`encode` is the full causal forward; :func:`encode_step` is the
KV-cached rollout path and gives the same embeddings (see tests).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from experiments.pusher_b_jax_2026_10.process import (
    CONTEXT_LENGTH,
    TOKEN_COUNT,
    VOCAB_SIZE,
)


@dataclass(frozen=True)
class ModelSpec:
    vocab: int = VOCAB_SIZE
    context_length: int = CONTEXT_LENGTH
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_mlp: int = 512
    init_std: float = 0.02
    norm_eps: float = 1e-5
    rope_base: float = 10_000.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def _normal(key, shape, std):
    return std * jax.random.normal(key, shape)


def _linear(key, fan_in, fan_out, std):
    return {"w": _normal(key, (fan_in, fan_out), std), "b": jnp.zeros(fan_out)}


def _torch_default_linear(key, fan_in, fan_out):
    bound = 1.0 / math.sqrt(fan_in)
    w_key, b_key = jax.random.split(key)
    return {
        "w": jax.random.uniform(w_key, (fan_in, fan_out), minval=-bound, maxval=bound),
        "b": jax.random.uniform(b_key, (fan_out,), minval=-bound, maxval=bound),
    }


def init_params(
    spec: ModelSpec,
    key: jax.Array,
    *,
    lm_head: bool = False,
    actor_critic: bool = False,
    aux_head: bool = False,
) -> dict:
    d, std = spec.d_model, spec.init_std
    keys = iter(jax.random.split(key, 8 + 5 * spec.n_layers))
    params = {
        "embed": _normal(next(keys), (spec.vocab, d), std),
        "blocks": [
            {
                "attn_norm": jnp.ones(d),
                "qkv": _linear(next(keys), d, 3 * d, std),
                "out": _linear(next(keys), d, d, std),
                "mlp_norm": jnp.ones(d),
                "gate": _linear(next(keys), d, spec.d_mlp, std),
                "up": _linear(next(keys), d, spec.d_mlp, std),
                "down": _linear(next(keys), spec.d_mlp, d, std),
            }
            for _ in range(spec.n_layers)
        ],
        "final_norm": jnp.ones(d),
    }
    if lm_head:
        params["lm"] = _normal(next(keys), (d, spec.vocab), std)
    if actor_critic:
        params["policy"] = _torch_default_linear(next(keys), d, TOKEN_COUNT)
        params["value"] = _torch_default_linear(next(keys), d, 1)
    if aux_head:
        params["aux"] = _torch_default_linear(next(keys), d, TOKEN_COUNT)
    return params


def parameter_count(params) -> int:
    return sum(x.size for x in jax.tree.leaves(params))


def _dense(p, x):
    return x @ p["w"] + p["b"]


def _rms_norm(scale, x, eps):
    return x * jax.lax.rsqrt((x * x).mean(-1, keepdims=True) + eps) * scale


def _rope(spec: ModelSpec, x, positions):
    """x: (..., T, H, Dh); positions: (T,)."""

    half = spec.head_dim // 2
    inv_freq = 1.0 / (spec.rope_base ** (jnp.arange(half, dtype=jnp.float32) / half))
    angles = positions.astype(jnp.float32)[:, None] * inv_freq[None, :]
    cos, sin = jnp.cos(angles)[:, None, :], jnp.sin(angles)[:, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return jnp.stack([even * cos - odd * sin, even * sin + odd * cos], -1).reshape(x.shape)


def _mlp(spec, block, x):
    h = _rms_norm(block["mlp_norm"], x, spec.norm_eps)
    hidden = jax.nn.gelu(_dense(block["gate"], h), approximate=False) * _dense(block["up"], h)
    return x + _dense(block["down"], hidden)


def _qkv(spec, block, x):
    h = _rms_norm(block["attn_norm"], x, spec.norm_eps)
    q, k, v = jnp.split(_dense(block["qkv"], h), 3, axis=-1)
    shape = x.shape[:-1] + (spec.n_heads, spec.head_dim)
    return q.reshape(shape), k.reshape(shape), v.reshape(shape)


def encode(spec: ModelSpec, params: dict, tokens: jax.Array) -> jax.Array:
    """Causal embeddings for ``tokens`` (B, T) -> (B, T, d), final norm applied."""

    batch, length = tokens.shape
    positions = jnp.arange(length)
    x = params["embed"][tokens]
    for block in params["blocks"]:
        q, k, v = _qkv(spec, block, x)
        q, k = _rope(spec, q, positions), _rope(spec, k, positions)
        attended = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        x = x + _dense(block["out"], attended.reshape(batch, length, -1))
        x = _mlp(spec, block, x)
    return _rms_norm(params["final_norm"], x, spec.norm_eps)


def lm_logits(params, embedding):
    return embedding @ params["lm"]


def policy_value(params, embedding):
    return _dense(params["policy"], embedding), _dense(params["value"], embedding)[..., 0]


class KVCache(NamedTuple):
    k: jax.Array  # (n_layers, B, context, H, Dh), keys already rotated
    v: jax.Array


def empty_cache(spec: ModelSpec, batch: int) -> KVCache:
    shape = (spec.n_layers, batch, spec.context_length, spec.n_heads, spec.head_dim)
    return KVCache(jnp.zeros(shape), jnp.zeros(shape))


def encode_step(
    spec: ModelSpec,
    params: dict,
    cache: KVCache,
    tokens: jax.Array,
    position: jax.Array,
) -> tuple[jax.Array, KVCache]:
    """Embed ``tokens`` (B,) at shared ``position`` for a synchronised batch."""

    batch = tokens.shape[0]
    x = params["embed"][tokens][:, None, :]
    pos = jnp.reshape(position, (1,))
    valid = jnp.arange(spec.context_length) <= position
    new_k, new_v = [], []
    for layer, block in enumerate(params["blocks"]):
        q, k, v = _qkv(spec, block, x)
        q, k = _rope(spec, q, pos), _rope(spec, k, pos)
        k_cache = jax.lax.dynamic_update_slice_in_dim(cache.k[layer], k, position, axis=1)
        v_cache = jax.lax.dynamic_update_slice_in_dim(cache.v[layer], v, position, axis=1)
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k_cache) / math.sqrt(spec.head_dim)
        scores = jnp.where(valid[None, None, None, :], scores, -jnp.inf)
        attended = jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(scores, -1), v_cache)
        x = x + _dense(block["out"], attended.reshape(batch, 1, -1))
        x = _mlp(spec, block, x)
        new_k.append(k_cache)
        new_v.append(v_cache)
    embedding = _rms_norm(params["final_norm"], x[:, 0], spec.norm_eps)
    return embedding, KVCache(jnp.stack(new_k), jnp.stack(new_v))


def observation_tokens(obs: jax.Array) -> jax.Array:
    """Delay-one one-hot observations (..., 2) -> token ids, all-zero -> BOS."""

    return jnp.where(obs.sum(-1) < 0.5, TOKEN_COUNT, obs.argmax(-1)).astype(jnp.int32)
