"""JAX ports of the four history-study policy encoders.

Each architecture mirrors the corresponding PyTorch encoder in RL-Harness,
including PyTorch default initialization, LayerNorm epsilon ``1e-5`` and exact
GELU. ``step`` is the cached rollout path and ``sequence`` the training path;
both return logits plus pre- and post-final-norm embeddings. A value head is
omitted because these REINFORCE recipes use no critic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp

KINDS = ("transformer", "memoryless", "gru")


@dataclass(frozen=True)
class ModelSpec:
    kind: str
    obs_dim: int
    n_actions: int
    width: int
    n_layers: int
    n_heads: int = 1
    context_len: int = 32
    chunk_len: int = 32
    feature_count: int | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if self.width % self.n_heads:
            raise ValueError("width must be divisible by n_heads")

    @property
    def head_dim(self) -> int:
        return self.width // self.n_heads


def architecture_spec(architecture: str, *, obs_dim: int, n_actions: int,
                      n_tokens: int, context_len: int) -> ModelSpec:
    if architecture == "transformer":
        return ModelSpec("transformer", obs_dim, n_actions, 64, 4, 1, context_len)
    if architecture == "memoryless":
        return ModelSpec("memoryless", obs_dim, n_actions, 64, 6, feature_count=n_tokens)
    if architecture == "gru128":
        return ModelSpec("gru", obs_dim, n_actions, 128, 2, chunk_len=32)
    if architecture == "gru92":
        return ModelSpec("gru", obs_dim, n_actions, 92, 4, chunk_len=32)
    raise ValueError(f"unknown architecture {architecture!r}")


def _uniform(key, shape, bound):
    return jax.random.uniform(key, shape, jnp.float32, -bound, bound)


def _linear(key, fan_in, fan_out):
    weight_key, bias_key = jax.random.split(key)
    bound = 1.0 / math.sqrt(fan_in)
    return {"w": _uniform(weight_key, (fan_in, fan_out), bound),
            "b": _uniform(bias_key, (fan_out,), bound)}


def _norm(width):
    return {"scale": jnp.ones((width,)), "bias": jnp.zeros((width,))}


def _dense(params, x):
    return x @ params["w"] + params["b"]


def _layer_norm(params, x):
    mean = x.mean(-1, keepdims=True)
    variance = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + 1e-5) * params["scale"] + params["bias"]


def _mlp(params, x):
    return _dense(params["fc2"], jax.nn.gelu(_dense(params["fc1"], x), approximate=False))


def init_params(spec: ModelSpec, key: jax.Array) -> dict:
    keys = iter(jax.random.split(key, 4 * spec.n_layers + 4))
    width = spec.width
    in_dim = spec.feature_count if spec.kind == "memoryless" else spec.obs_dim
    params = {"input": _linear(next(keys), in_dim, width), "final_norm": _norm(width)}
    blocks = []
    for _ in range(spec.n_layers):
        if spec.kind == "transformer":
            blocks.append({
                "ln1": _norm(width), "qkv": _linear(next(keys), width, 3 * width),
                "proj": _linear(next(keys), width, width), "ln2": _norm(width),
                "mlp": {"fc1": _linear(next(keys), width, 4 * width),
                        "fc2": _linear(next(keys), 4 * width, width)},
            })
        elif spec.kind == "memoryless":
            blocks.append({"ln": _norm(width),
                           "mlp": {"fc1": _linear(next(keys), width, 4 * width),
                                   "fc2": _linear(next(keys), 4 * width, width)}})
        else:
            bound = 1.0 / math.sqrt(width)
            k = jax.random.split(next(keys), 4)
            blocks.append({"w_i": _uniform(k[0], (width, 3 * width), bound),
                           "b_i": _uniform(k[1], (3 * width,), bound),
                           "w_h": _uniform(k[2], (width, 3 * width), bound),
                           "b_h": _uniform(k[3], (3 * width,), bound)})
    params["blocks"] = blocks
    params["policy"] = _linear(next(keys), width, spec.n_actions)
    return params


def parameter_count(params) -> int:
    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)))


def init_carry(spec: ModelSpec, batch: int):
    if spec.kind == "transformer":
        shape = (spec.n_layers, batch, spec.context_len + 1, spec.n_heads, spec.head_dim)
        return {"k": jnp.zeros(shape), "v": jnp.zeros(shape)}
    if spec.kind == "gru":
        return jnp.zeros((spec.n_layers, batch, spec.width))
    return jnp.zeros((batch, 0))


def _rope(x, positions):
    """Interleaved RoPE (base 10,000) over the last axis; positions broadcast."""

    half = x.shape[-1] // 2
    inv_freq = 1.0 / (10000.0 ** (jnp.arange(half, dtype=jnp.float32) / half))
    angles = positions.astype(jnp.float32)[..., None] * inv_freq
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    even, odd = x[..., 0::2], x[..., 1::2]
    return jnp.stack([even * cos - odd * sin, even * sin + odd * cos], -1).reshape(x.shape)


def _gru_cell(block, x, h):
    i_r, i_z, i_n = jnp.split(x @ block["w_i"] + block["b_i"], 3, -1)
    h_r, h_z, h_n = jnp.split(h @ block["w_h"] + block["b_h"], 3, -1)
    reset = jax.nn.sigmoid(i_r + h_r)
    update = jax.nn.sigmoid(i_z + h_z)
    candidate = jnp.tanh(i_n + reset * h_n)
    return (1.0 - update) * candidate + update * h


def _gru_stack(blocks, x, hidden):
    outputs = []
    for layer, block in enumerate(blocks):
        x = _gru_cell(block, x, hidden[layer])
        outputs.append(x)
    return x, jnp.stack(outputs)


def _head(params, residual):
    embedding = _layer_norm(params["final_norm"], residual)
    return _dense(params["policy"], embedding), residual, embedding


def step(spec: ModelSpec, params, carry, obs, t):
    """One decision for a synchronized batch at episode step ``t``."""

    if spec.kind == "memoryless":
        x = _dense(params["input"], obs[..., : spec.feature_count])
        for block in params["blocks"]:
            x = x + _mlp(block["mlp"], _layer_norm(block["ln"], x))
        return (*_head(params, x), carry)
    if spec.kind == "gru":
        x, hidden = _gru_stack(params["blocks"], _dense(params["input"], obs), carry)
        return (*_head(params, x), hidden)

    batch = obs.shape[0]
    cache_len = spec.context_len + 1
    slot = t % cache_len
    valid = jnp.arange(cache_len) <= t
    x = _dense(params["input"], obs)
    keys, values = [], []
    for layer, block in enumerate(params["blocks"]):
        q, k, v = jnp.split(_dense(block["qkv"], _layer_norm(block["ln1"], x)), 3, -1)
        heads = (batch, spec.n_heads, spec.head_dim)
        q, k, v = _rope(q.reshape(heads), t), _rope(k.reshape(heads), t), v.reshape(heads)
        k_cache = carry["k"][layer].at[:, slot].set(k)
        v_cache = carry["v"][layer].at[:, slot].set(v)
        scores = jnp.einsum("bhd,bshd->bhs", q, k_cache) / math.sqrt(spec.head_dim)
        weights = jax.nn.softmax(jnp.where(valid, scores, -jnp.inf), -1)
        attended = jnp.einsum("bhs,bshd->bhd", weights, v_cache).reshape(batch, spec.width)
        x = x + _dense(block["proj"], attended)
        x = x + _mlp(block["mlp"], _layer_norm(block["ln2"], x))
        keys.append(k_cache)
        values.append(v_cache)
    return (*_head(params, x), {"k": jnp.stack(keys), "v": jnp.stack(values)})


def _banded_attention(q, k, v, band):
    """Causal attention where query i sees keys ``i - band .. i`` (exact)."""

    batch, length, n_heads, head_dim = q.shape
    pad = (-length) % band
    q, k, v = (jnp.pad(a, ((0, 0), (0, pad), (0, 0), (0, 0))) for a in (q, k, v))
    n_blocks = (length + pad) // band

    def blocks(a):
        return a.reshape(batch, n_blocks, band, n_heads, head_dim)

    qb, kb, vb = blocks(q), blocks(k), blocks(v)

    def with_previous(a):
        previous = jnp.concatenate([jnp.zeros_like(a[:, :1]), a[:, :-1]], 1)
        return jnp.concatenate([previous, a], 2)

    keys, values = with_previous(kb), with_previous(vb)
    scores = jnp.einsum("bnqhd,bnkhd->bnhqk", qb, keys) / math.sqrt(head_dim)
    query = jnp.arange(band)[:, None] + band
    key = jnp.arange(2 * band)[None, :]
    offset = query - key
    mask = (offset >= 0) & (offset <= band)
    first_block = (jnp.arange(n_blocks) == 0)[:, None, None]
    mask = mask[None] & ~(first_block & (key < band)[None])
    weights = jax.nn.softmax(jnp.where(mask[None, :, None], scores, -jnp.inf), -1)
    out = jnp.einsum("bnhqk,bnkhd->bnqhd", weights, values)
    return out.reshape(batch, n_blocks * band, n_heads * head_dim)[:, :length]


def sequence(spec: ModelSpec, params, obs, chunk_starts=None):
    """Training forward over complete episodes ``obs[B, T, D]``.

    GRU episodes are split into ``chunk_len`` windows starting from the
    recorded rollout hidden state ``chunk_starts[B, L, n_chunks, H]``,
    matching RLlib truncated BPTT.
    """

    if spec.kind == "memoryless":
        x = _dense(params["input"], obs[..., : spec.feature_count])
        for block in params["blocks"]:
            x = x + _mlp(block["mlp"], _layer_norm(block["ln"], x))
        return _head(params, x)
    if spec.kind == "gru":
        batch, length, _ = obs.shape
        if length % spec.chunk_len:
            raise ValueError("episode length must be a multiple of chunk_len")
        n_chunks = length // spec.chunk_len
        x = _dense(params["input"], obs).reshape(batch, n_chunks, spec.chunk_len, spec.width)
        hidden = jax.lax.stop_gradient(jnp.moveaxis(chunk_starts, 0, 1))  # (L, B, C, H)

        def body(hidden, x_t):
            out, hidden = _gru_stack(params["blocks"], x_t, hidden)
            return hidden, out

        _, outputs = jax.lax.scan(body, hidden, jnp.moveaxis(x, 2, 0))
        x = jnp.moveaxis(outputs, 0, 2).reshape(batch, length, spec.width)
        return _head(params, x)

    batch, length, _ = obs.shape
    positions = jnp.arange(length)[:, None]
    x = _dense(params["input"], obs)
    for block in params["blocks"]:
        q, k, v = jnp.split(_dense(block["qkv"], _layer_norm(block["ln1"], x)), 3, -1)
        heads = (batch, length, spec.n_heads, spec.head_dim)
        q, k = _rope(q.reshape(heads), positions), _rope(k.reshape(heads), positions)
        attended = _banded_attention(q, k, v.reshape(heads), spec.context_len)
        x = x + _dense(block["proj"], attended)
        x = x + _mlp(block["mlp"], _layer_norm(block["ln2"], x))
    return _head(params, x)
