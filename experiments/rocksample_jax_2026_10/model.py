"""JAX port of ``learners.models.transformer.TransformerModel``.

Same architecture as the RLlib model: input projection, pre-LN RoPE blocks
whose attention is a hard causal band of ``context_len`` keys, final
LayerNorm, linear policy and value heads. The windowed path
(:func:`encode_window`) and the KV-cached rollout path (:func:`encode_step`)
give identical embeddings; attention never crosses an episode boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class TransformerSpec:
    obs_dim: int
    num_actions: int
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    context_len: int = 32

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("d_model / n_heads must be an even integer")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def cache_len(self) -> int:
        return self.context_len + 1

    @property
    def lookback(self) -> int:
        return self.n_layers * self.context_len


def _linear(key: jax.Array, fan_in: int, fan_out: int) -> dict[str, jax.Array]:
    bound = 1.0 / math.sqrt(fan_in)
    w_key, b_key = jax.random.split(key)
    return {
        "w": jax.random.uniform(w_key, (fan_in, fan_out), minval=-bound, maxval=bound),
        "b": jax.random.uniform(b_key, (fan_out,), minval=-bound, maxval=bound),
    }


def _layer_norm_params(width: int) -> dict[str, jax.Array]:
    return {"scale": jnp.ones(width), "bias": jnp.zeros(width)}


def init_params(spec: TransformerSpec, key: jax.Array) -> dict:
    """PyTorch-default (Kaiming-uniform) initialisation."""

    d = spec.d_model
    keys = jax.random.split(key, 3 + 4 * spec.n_layers)
    blocks = []
    for layer in range(spec.n_layers):
        k0, k1, k2, k3 = keys[3 + 4 * layer : 7 + 4 * layer]
        blocks.append(
            {
                "ln1": _layer_norm_params(d),
                "qkv": _linear(k0, d, 3 * d),
                "proj": _linear(k1, d, d),
                "ln2": _layer_norm_params(d),
                "fc1": _linear(k2, d, 4 * d),
                "fc2": _linear(k3, 4 * d, d),
            }
        )
    return {
        "input": _linear(keys[0], spec.obs_dim, d),
        "blocks": blocks,
        "final_norm": _layer_norm_params(d),
        "policy": _linear(keys[1], d, spec.num_actions),
        "value": _linear(keys[2], d, 1),
    }


def _dense(p, x):
    return x @ p["w"] + p["b"]


def _layer_norm(p, x, eps: float = 1e-5):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps) * p["scale"] + p["bias"]


def _rope(x: jax.Array, positions: jax.Array) -> jax.Array:
    """Rotate interleaved (even, odd) pairs; x is (..., length, head_dim)."""

    half = x.shape[-1] // 2
    inv_freq = 1.0 / (10000.0 ** (jnp.arange(half, dtype=jnp.float32) / half))
    angles = positions.astype(jnp.float32)[:, None] * inv_freq[None, :]
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = jnp.stack([even * cos - odd * sin, even * sin + odd * cos], axis=-1)
    return rotated.reshape(x.shape)


def _mlp(block, x):
    hidden = jax.nn.gelu(
        _dense(block["fc1"], _layer_norm(block["ln2"], x)), approximate=False
    )
    return x + _dense(block["fc2"], hidden)


def _split_heads(spec: TransformerSpec, x: jax.Array) -> jax.Array:
    return x.reshape(x.shape[0], spec.n_heads, spec.head_dim).transpose(1, 0, 2)


def heads(params: dict, embedding: jax.Array) -> tuple[jax.Array, jax.Array]:
    logits = _dense(params["policy"], embedding)
    value = _dense(params["value"], embedding)[..., 0]
    return logits, value


def encode_window(
    spec: TransformerSpec,
    params: dict,
    obs: jax.Array,
    episode_ids: jax.Array,
) -> jax.Array:
    """Embeddings for one env's sequence ``obs`` (L, obs_dim).

    Positions attend to at most ``context_len`` earlier positions with the same
    episode id. Entries with episode id ``-1`` are padding.
    """

    length = obs.shape[0]
    index = jnp.arange(length)
    band = (index[None, :] <= index[:, None]) & (
        index[:, None] - index[None, :] <= spec.context_len
    )
    same = (episode_ids[:, None] == episode_ids[None, :]) & (episode_ids[None, :] >= 0)
    mask = (band & same) | jnp.eye(length, dtype=bool)
    x = _dense(params["input"], obs)
    scale = 1.0 / math.sqrt(spec.head_dim)
    for block in params["blocks"]:
        q, k, v = jnp.split(
            _dense(block["qkv"], _layer_norm(block["ln1"], x)), 3, axis=-1
        )
        q = _rope(_split_heads(spec, q), index)
        k = _rope(_split_heads(spec, k), index)
        v = _split_heads(spec, v)
        scores = jnp.where(mask[None], (q @ k.transpose(0, 2, 1)) * scale, -jnp.inf)
        attention = jax.nn.softmax(scores, axis=-1) @ v
        x = x + _dense(block["proj"], attention.transpose(1, 0, 2).reshape(length, -1))
        x = _mlp(block, x)
    return _layer_norm(params["final_norm"], x)


class KVCache(NamedTuple):
    k: jax.Array  # (n_layers, n_heads, cache_len, head_dim), un-rotated keys
    v: jax.Array
    length: jax.Array  # () int32 valid trailing slots


def empty_cache(spec: TransformerSpec) -> KVCache:
    shape = (spec.n_layers, spec.n_heads, spec.cache_len, spec.head_dim)
    return KVCache(jnp.zeros(shape), jnp.zeros(shape), jnp.int32(0))


def encode_step(
    spec: TransformerSpec,
    params: dict,
    cache: KVCache,
    obs: jax.Array,
) -> tuple[jax.Array, KVCache]:
    """Embed one observation for one env, updating its KV cache."""

    length = jnp.minimum(cache.length + 1, spec.cache_len)
    slots = jnp.arange(spec.cache_len)
    valid = slots >= spec.cache_len - length
    key_positions = slots
    query_position = jnp.array([spec.cache_len - 1])
    scale = 1.0 / math.sqrt(spec.head_dim)
    x = _dense(params["input"], obs)[None, :]
    new_k, new_v = [], []
    for layer, block in enumerate(params["blocks"]):
        q, k, v = jnp.split(
            _dense(block["qkv"], _layer_norm(block["ln1"], x)), 3, axis=-1
        )
        q, k, v = _split_heads(spec, q), _split_heads(spec, k), _split_heads(spec, v)
        k_cache = jnp.concatenate([cache.k[layer][:, 1:], k], axis=1)
        v_cache = jnp.concatenate([cache.v[layer][:, 1:], v], axis=1)
        scores = (
            _rope(q, query_position) @ _rope(k_cache, key_positions).transpose(0, 2, 1)
        ) * scale
        scores = jnp.where(valid[None, None, :], scores, -jnp.inf)
        attention = jax.nn.softmax(scores, axis=-1) @ v_cache
        x = x + _dense(block["proj"], attention.transpose(1, 0, 2).reshape(1, -1))
        x = _mlp(block, x)
        new_k.append(k_cache)
        new_v.append(v_cache)
    embedding = _layer_norm(params["final_norm"], x)[0]
    return embedding, KVCache(jnp.stack(new_k), jnp.stack(new_v), length)
