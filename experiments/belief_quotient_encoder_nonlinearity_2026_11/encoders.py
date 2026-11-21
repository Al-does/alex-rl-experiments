"""Encoder-only ablation of the historical Eq. (4) activation write."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_default_matmul_precision", "highest")


def r_squared(predicted, actual):
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    denominator = np.square(actual - actual.mean(0)).sum()
    return float(1 - np.square(predicted - actual).sum() / denominator) if denominator else None


@dataclass
class Encoder:
    kind: str
    params: tuple
    center: jax.Array
    scale: jax.Array
    output_center: jax.Array
    output_scale: jax.Array
    diagnostics: dict

    def __call__(self, belief):
        if self.kind == "affine":
            return belief @ self.params[0] + self.params[1]
        w1, b1, w2, b2 = self.params
        hidden = jnp.tanh(((belief - self.center) / self.scale) @ w1 + b1)
        return (hidden @ w2 + b2) * self.output_scale + self.output_center

    def displacement(self, current, target):
        if self.kind == "affine":
            return (target - current) @ self.params[0]
        return self(target) - self(current)


def fit(x, h, xv, hv, kind, *, rcond=1e-10, seed=42, max_epochs=200,
        patience=15, batch_size=1024):
    x, h, xv, hv = [np.asarray(a, dtype=np.float64) for a in (x, h, xv, hv)]
    if (any(a.ndim != 2 or not len(a) or not np.isfinite(a).all() for a in (x, h, xv, hv))
            or len(x) != len(h) or len(xv) != len(hv)
            or x.shape[1] != xv.shape[1] or h.shape[1] != hv.shape[1]):
        raise ValueError("finite, nonempty, aligned train/validation matrices required")
    if kind not in ("affine", "mlp") or min(max_epochs, patience, batch_size) < 1:
        raise ValueError("valid encoder kind and positive training budgets required")
    if kind == "affine":
        coefficients = np.linalg.lstsq(np.c_[x, np.ones(len(x))], h, rcond=rcond)[0]
        return Encoder(kind, tuple(map(jnp.asarray, (coefficients[:-1], coefficients[-1]))),
                       None, None, None, None, dict(rcond=rcond, train_rows=len(x), validation_rows=len(xv)))
    center, scale = x.mean(0), np.maximum(x.std(0), 1e-4)
    output_center, output_scale = h.mean(0), np.maximum(h.std(0), 1e-4)
    x, h, xv, hv = [jnp.asarray(a, jnp.float32) for a in
                     ((x - center) / scale, (h - output_center) / output_scale,
                      (xv - center) / scale, (hv - output_center) / output_scale)]
    width = 2 * h.shape[1]
    key1, key2 = jax.random.split(jax.random.PRNGKey(seed))
    params = (jax.random.normal(key1, (x.shape[1], width)) / jnp.sqrt(x.shape[1]),
              jnp.zeros(width), jax.random.normal(key2, (width, h.shape[1])) / jnp.sqrt(width),
              jnp.zeros(h.shape[1]))
    optimizer = optax.adamw(1e-3, weight_decay=1e-4)
    state = optimizer.init(params)

    def loss(p, features, targets):
        w1, b1, w2, b2 = p
        return jnp.mean(jnp.square(jnp.tanh(features @ w1 + b1) @ w2 + b2 - targets))

    @jax.jit
    def epoch(p, state, indices):
        def step(carry, batch):
            p, state = carry
            gradient = jax.grad(loss)(p, x[batch], h[batch])
            updates, state = optimizer.update(gradient, state, p)
            return (optax.apply_updates(p, updates), state), None
        return jax.lax.scan(step, (p, state), indices)[0]

    validation_loss = jax.jit(lambda p: loss(p, xv, hv))
    initial = best = float(validation_loss(params))
    best_params, best_epoch, stale, history = params, 0, 0, []
    rng = np.random.default_rng(seed)
    for number in range(1, max_epochs + 1):
        indices = rng.permutation(len(x))
        indices = np.resize(indices, ((len(x) + batch_size - 1) // batch_size) * batch_size)
        params, state = epoch(params, state, jnp.asarray(indices.reshape(-1, batch_size)))
        value = float(validation_loss(params))
        if not np.isfinite(value):
            raise ValueError("nonfinite validation loss")
        if number == 1 or number % 10 == 0:
            history.append([number, value])
        if value < best - 1e-5:
            best, best_params, best_epoch, stale = value, params, number, 0
        else:
            stale += 1
        if stale >= patience:
            break
    diagnostics = dict(architecture="Linear-tanh-Linear", hidden_width=width,
                       weight_decay=1e-4, learning_rate=1e-3, seed=seed, batch_size=batch_size,
                       max_epochs=max_epochs, patience=patience, min_delta=1e-5,
                       best_epoch=best_epoch, stopped_epoch=number, initial_validation_loss=initial,
                       best_validation_loss=best, validation_curve=history,
                       train_rows=len(x), validation_rows=len(xv))
    return Encoder(kind, best_params, *map(jnp.asarray, (center, scale, output_center, output_scale)), diagnostics)
