"""Exact cached continuation of the frozen four-block HMM transformer."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_causal_decoding_2026_11 import decoders


def logits(spec, params, h, cache, t, site):
    if site not in (1, 3):
        raise ValueError("expected site 1 or 3")
    x = h
    for layer in range(site + 1, spec.n_layers):
        p = params["blocks"][layer]
        q, k, v = jnp.split(models._dense(p["qkv"], models._layer_norm(p["ln1"], x)), 3, -1)
        shape = (len(x), spec.n_heads, spec.head_dim)
        q, k = (models._rope(a.reshape(shape), t) for a in (q, k))
        keys = cache["k"][layer].at[:, t % (spec.context_len + 1)].set(k)
        values = cache["v"][layer].at[:, t % (spec.context_len + 1)].set(v.reshape(shape))
        scores = jnp.einsum("bhd,bshd->bhs", q, keys) / math.sqrt(spec.head_dim)
        valid = jnp.arange(spec.context_len + 1) <= t
        weights = jax.nn.softmax(jnp.where(valid, scores, -jnp.inf), -1)
        attended = jnp.einsum("bhs,bshd->bhd", weights, values).reshape(len(x), spec.width)
        x = x + models._dense(p["proj"], attended)
        x = x + models._mlp(p["mlp"], models._layer_norm(p["ln2"], x))
    return models._dense(params["policy"], models._layer_norm(params["final_norm"], x))


def contexts(spec, params, sites, episodes, times, site):
    """Reconstruct pre-query KV slots from intact, complete episode residuals."""
    episodes, times = np.asarray(episodes), np.asarray(times)
    if not len(times) or episodes.shape != times.shape or np.any(times < 0):
        raise ValueError("nonempty aligned episode/time indices required")
    slots = np.arange(spec.context_len + 1)
    positions = times[:, None] - 1 - ((times[:, None] - 1 - slots) % len(slots))
    valid = positions >= 0
    caches = models.init_carry(spec, len(times))
    for layer in range(site + 1, spec.n_layers):
        x = jnp.asarray(np.asarray(sites)[episodes[:, None], positions.clip(0), layer - 1])
        p = params["blocks"][layer]
        _, k, v = jnp.split(models._dense(p["qkv"], models._layer_norm(p["ln1"], x)), 3, -1)
        shape = (len(times), len(slots), spec.n_heads, spec.head_dim)
        k = models._rope(k.reshape(shape), jnp.asarray(positions.clip(0))[..., None])
        caches["k"] = caches["k"].at[layer].set(jnp.where(jnp.asarray(valid)[..., None, None], k, 0.))
        caches["v"] = caches["v"].at[layer].set(jnp.where(jnp.asarray(valid)[..., None, None], v.reshape(shape), 0.))
    return caches


def batch_logits(spec, params, h, cache, times, site):
    def one(x, c, t):
        return logits(spec, params, x[None], jax.tree_util.tree_map(lambda a: a[:, None], c), t, site)[0]
    return jax.vmap(one, (0, {"k": 1, "v": 1}, 0))(h, cache, times)


def jacobians(spec, params, h, cache, times, site, transform=lambda z: z):
    def one(x, c, t):
        z = logits(spec, params, x[None], jax.tree_util.tree_map(lambda a: a[:, None], c), t, site)[0]
        return transform(z)
    return jax.vmap(jax.jacrev(one), (0, {"k": 1, "v": 1}, 0))(h, cache, times)


def action_basis(spec, params, h, cache, times, site, rank, transform=lambda z: z, exclude=None):
    matrix = np.asarray(jacobians(spec, params, h, cache, times, site, transform)).reshape(-1, spec.width)
    if exclude is not None:
        matrix = matrix - (matrix @ exclude) @ exclude.T
    _, singular, axes = np.linalg.svd(matrix, full_matrices=False)
    count = min(rank, int(np.sum(singular > singular[0] * decoders.RCOND)))
    if not count:
        raise ValueError("policy Jacobian has no supported directions")
    return jnp.asarray(axes[:count].T), singular.tolist()


def null_write(spec, params, h, direction, cache, t, site, transform):
    times = jnp.full((len(h),), t)
    jac = jacobians(spec, params, h, cache, times, site, transform)
    inverse = jnp.linalg.pinv(jac, rtol=decoders.RCOND)
    return direction - jnp.einsum("bij,bjk,bk->bi", inverse, jac, direction)


def donor_index(h, target, current, bank_h, bank_b, basis, preserve, scale, tolerance=.15):
    if not len(bank_h):
        return jnp.zeros(len(h), jnp.int32), jnp.zeros(len(h), bool)
    target_error = jnp.max(jnp.abs((bank_b[None] - target[:, None]) * (1 - preserve)), -1)
    preserve_error = jnp.max(jnp.abs((bank_b[None] - current[:, None]) * preserve), -1)
    difference = bank_h[None] - h[:, None]
    complement = difference - (difference @ basis) @ basis.T
    distance = jnp.mean((complement / scale) ** 2, -1) + preserve_error ** 2
    valid = (target_error <= tolerance) & (preserve_error <= tolerance)
    return jnp.argmin(jnp.where(valid, distance, jnp.inf), -1), valid.any(-1)


def grid_actions(policy, belief):
    if policy.quotient:
        index = jnp.rint(jnp.clip(belief[..., 2], 0., 1.) * policy.n).astype(jnp.int32)
    else:
        b = jnp.maximum(belief, 0.)
        b = b / b.sum(-1, keepdims=True)
        ij = jnp.rint(b[..., :2] * policy.n).astype(jnp.int32)
        i, j = ij[..., 0], ij[..., 1]
        over = i + j > policy.n
        i, j = (jnp.where(over & (b[..., 0] > b[..., 1]), policy.n-j, i),
                jnp.where(over & (b[..., 0] <= b[..., 1]), policy.n-i, j))
        index = jnp.asarray(policy.lookup)[i, j]
    return jnp.asarray(policy.actions)[index]
