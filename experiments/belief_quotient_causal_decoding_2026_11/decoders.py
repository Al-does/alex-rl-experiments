"""Experimental decoders and local causal writes for the frozen HMM policies."""

import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_default_matmul_precision", "highest")
RCOND = 1e-6


def affine(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("finite nonempty fitting rows required")
    coef = np.linalg.lstsq(np.c_[x, np.ones(len(x))], y, rcond=RCOND)[0]
    w, b = map(jnp.asarray, (coef[:-1], coef[-1]))
    return lambda h: h @ w + b


def optimize(loss, initial, steps=300, rate=.01):
    optimizer = optax.adam(rate)
    def body(carry, _):
        p, state = carry
        value, grad = jax.value_and_grad(loss)(p)
        updates, state = optimizer.update(grad, state, p)
        return (optax.apply_updates(p, updates), state), value
    (p, _), values = jax.lax.scan(body, (initial, optimizer.init(initial)), None, steps)
    return p, values


def kl(p, q):
    return jnp.sum(p * (jnp.log(jnp.maximum(p, 1e-8)) - jnp.log(jnp.maximum(q, 1e-8))), -1)


def mlp(x, y, seed=42, width=64, steps=300):
    x, y = jnp.asarray(x), jnp.asarray(y)
    if not len(x):
        raise ValueError("nonempty fitting rows required")
    center, scale = x.mean(0), jnp.maximum(x.std(0), .01)
    key1, key2 = jax.random.split(jax.random.PRNGKey(seed))
    initial = (jax.random.normal(key1, (x.shape[-1], width)) / jnp.sqrt(x.shape[-1]),
               jnp.zeros(width), jax.random.normal(key2, (width, y.shape[-1])) * .01, y.mean(0))
    def apply(p, h):
        a, b, c, d = p
        return jnp.tanh((h - center) / scale @ a + b) @ c + d
    p, _ = optimize(lambda p: jnp.mean((apply(p, x) - y) ** 2), initial, steps)
    return lambda h: apply(p, h)


def policy_implied(beliefs, logits, temperature=1.5, steps=100):
    """Return inverse policy map; beliefs have shape (rows, factors, states).

    The inverse is generally nonidentifiable. Uniform initialization is fixed,
    not initialized at the held-out Bayes posterior.
    """
    shape = beliefs.shape[1:]
    forward = affine(np.asarray(beliefs).reshape(len(beliefs), -1), logits)
    probabilities = jax.nn.softmax(jnp.asarray(logits) / temperature)
    initial = jnp.zeros((np.prod(shape), logits.shape[-1]))
    # Refit the map by policy KL rather than squared logit error.
    b = jnp.asarray(beliefs).reshape(len(beliefs), -1)
    w, _ = optimize(lambda w: kl(probabilities, jax.nn.softmax((forward(b) + b @ w) / temperature)).mean(), initial)
    def decode(z):
        p = jax.nn.softmax(z / temperature)
        def loss(u):
            q = jax.nn.softmax(u, -1).reshape(len(z), -1)
            return kl(p, jax.nn.softmax((forward(q) + q @ w) / temperature)).mean()
        u, _ = optimize(loss, jnp.zeros((len(z), *shape)), steps, .05)
        return jax.nn.softmax(u, -1)
    decode.policy_map = lambda q: jax.nn.softmax((forward(q.reshape(len(q), -1)) + q.reshape(len(q), -1) @ w) / temperature)
    return decode


def band_features(history, offsets=(0, 1, 4, 8)):
    """Concatenate causal residual positions; caller excludes startup rows."""
    return jnp.concatenate([jnp.take(history, jnp.maximum(jnp.arange(history.shape[-2]) - lag, 0), axis=-2)
                            for lag in offsets], -1)


def das(h, donor, policy, target_policy, non_target=lambda p: p, beta=1., rank=4, steps=300, seed=42,
        targeted=lambda p: p):
    if not 1 <= rank <= h.shape[-1] or not len(h):
        raise ValueError("rank must fit the nonempty activation matrix")
    initial = jax.random.normal(jax.random.PRNGKey(seed), (h.shape[-1], rank))
    original = policy(h)
    def basis(p):
        return jnp.linalg.qr(p, mode="reduced")[0]
    def loss(p):
        u = basis(p)
        written = policy(h + (donor - h) @ u @ u.T)
        return (kl(targeted(target_policy), targeted(written)) + beta * kl(non_target(original), non_target(written))).mean()
    p, values = optimize(loss, initial, steps)
    return basis(p), values[-1]


def donor_write(h, target, current, bank_h, bank_b, basis, preserve, tolerance=.05):
    """Match actual bank rows; abstain when preserved coordinates lack support."""
    if not len(bank_h):
        return jnp.zeros_like(h), jnp.zeros(h.shape[:-1], bool)
    distance = jnp.sum((bank_b[None] - target[:, None]) ** 2, -1)
    error = jnp.max(jnp.abs((bank_b[None] - current[:, None]) * preserve), -1)
    valid = error <= tolerance
    index = jnp.argmin(jnp.where(valid, distance, jnp.inf), -1)
    supported = valid.any(-1)
    delta = (bank_h[index] - h) @ basis @ basis.T
    return jnp.where(supported[:, None], delta, 0.), supported


def gradient_write(h, target, decoder, policy, non_target=lambda p: p, beta=1., gamma=.01, steps=60):
    intact = non_target(policy(h))
    def loss(delta):
        return (jnp.mean((decoder(h + delta) - target) ** 2)
                + beta * kl(intact, non_target(policy(h + delta))).mean() + gamma * jnp.mean(delta ** 2))
    return optimize(loss, jnp.zeros_like(h), steps, .03)[0]


def null_write(h, direction, non_target_logits):
    jacobian = jax.vmap(jax.jacrev(non_target_logits))(h)
    inverse = jnp.linalg.pinv(jacobian, rtol=RCOND)
    return direction - jnp.einsum("bij,bjk,bk->bi", inverse, jacobian, direction)


def contextual_donor(h, target, current, bank_h, bank_b, basis, preserve,
                     scale, tolerance=.15, full=False):
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if not len(bank_h):
        return jnp.zeros_like(h), jnp.zeros(len(h), bool)
    target_error = jnp.max(jnp.abs((bank_b[None] - target[:, None]) * (1 - preserve)), -1)
    preserve_error = jnp.max(jnp.abs((bank_b[None] - current[:, None]) * preserve), -1)
    valid = (target_error <= tolerance) & (preserve_error <= tolerance)
    difference = bank_h[None] - h[:, None]
    complement = difference - (difference @ basis) @ basis.T
    context = jnp.mean((complement / scale) ** 2, -1) + preserve_error ** 2
    index = jnp.argmin(jnp.where(valid, context, jnp.inf), -1)
    supported = valid.any(-1)
    delta = bank_h[index] - h
    if not full:
        delta = (delta @ basis) @ basis.T
    return jnp.where(supported[:, None], delta, 0.), supported
