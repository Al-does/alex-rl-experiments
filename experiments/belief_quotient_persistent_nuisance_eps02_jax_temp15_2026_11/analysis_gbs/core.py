"""Predictive distillation. This module accepts symbols and distributions only."""
from dataclasses import dataclass
from functools import partial
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax


@dataclass(frozen=True)
class Spec:
    kind: str
    k: int
    tokens: int
    actions: int
    outputs: int
    width: int = 32
    output_groups: int = 1

    def __post_init__(self):
        if self.kind not in ("affine", "gru", "tanh", "hmm"):
            raise ValueError(self.kind)
        if self.k < 0 or min(self.tokens, self.actions, self.outputs, self.width) < 1:
            raise ValueError("invalid dimensions")


def init(spec, seed):
    keys = iter(jax.random.split(jax.random.PRNGKey(seed), 12))

    def normal(shape, scale=.1):
        return jax.random.normal(next(keys), shape) * scale

    k = spec.k
    p = {"g0": normal((k,)), "h": normal((k, spec.width)),
         "hb": jnp.zeros(spec.width), "o": normal((spec.width, spec.outputs)),
         "ob": jnp.zeros(spec.outputs)}
    symbols = spec.tokens * spec.actions
    if spec.kind == "hmm":
        p.update(prior=normal((k+1,)), T=normal((spec.actions, k+1, k+1)),
                 E=normal((k+1, spec.tokens)))
    elif spec.kind == "affine":
        p.update(A=jnp.tile(jnp.eye(k)[None] * .8, (symbols, 1, 1)) +
                 normal((symbols, k, k), .01), b=normal((symbols, k)))
    else:
        gates = 3 if spec.kind == "gru" else 1
        p.update(W=normal((symbols, gates*k)), R=normal((k, gates*k)),
                 bias=jnp.zeros(gates*k))
    return p


def initial(spec, p):
    return jax.nn.softmax(p["prior"])[:-1] if spec.kind == "hmm" else p["g0"]


def update(spec, p, g, x, reset=False):
    """Reset emits from the learned prior without a fictitious transition."""
    if spec.kind == "hmm":
        b = jnp.concatenate((g, 1-g.sum(-1, keepdims=True)), -1)
        T, E = jax.nn.softmax(p["T"], -1), jax.nn.softmax(p["E"], -1)
        b = jnp.where(reset, b, jnp.einsum("...i,...ij->...j", b, T[x // spec.tokens]))
        b = b * jnp.moveaxis(E[:, x % spec.tokens], 0, -1)
        return (b / b.sum(-1, keepdims=True))[..., :-1]
    if spec.kind == "affine":
        return jnp.einsum("...i,...ij->...j", g, p["A"][x]) + p["b"][x]
    if spec.kind == "tanh":
        return jnp.tanh(g @ p["R"] + p["W"][x] + p["bias"])
    xz, xr, xn = jnp.split(p["W"][x] + p["bias"], 3, -1)
    hz, hr, hn = jnp.split(g @ p["R"], 3, -1)
    z, r = jax.nn.sigmoid(xz+hz), jax.nn.sigmoid(xr+hr)
    return z*g + (1-z)*jnp.tanh(xn+r*hn)


@partial(jax.jit, static_argnums=0)
def states(spec, p, x):
    def body(g, row):
        t, symbol = row
        g = update(spec, p, g, symbol, t == 0)
        return g, g
    g = jnp.broadcast_to(initial(spec, p), (x.shape[0], spec.k))
    _, gs = jax.lax.scan(body, g, (jnp.arange(x.shape[1]), x.T))
    return jnp.swapaxes(gs, 0, 1)


def log_predictions(spec, p, x):
    g = states(spec, p, x)
    logits = jnp.tanh(g @ p["h"] + p["hb"]) @ p["o"] + p["ob"]
    if spec.output_groups > 1:
        logits = logits.reshape(*logits.shape[:-1], spec.output_groups, -1)
    return jax.nn.log_softmax(logits, -1)


@partial(jax.jit, static_argnums=0)
def loss(spec, p, x, target):
    return jnp.mean(jnp.sum(jax.scipy.special.xlogy(target, target) -
                            target * log_predictions(spec, p, x), -1))


def fit(spec, train_x, train_target, val_x, val_target, *, seed=0, lr=.003,
        updates=2000, batch=32):
    """Whole-episode minibatches; validation selects the best saved iterate."""
    if len(train_x) == 0 or len(val_x) == 0 or updates < 1 or batch < 1:
        raise ValueError("nonempty train/validation and positive updates/batch required")
    for x, y in ((train_x, train_target), (val_x, val_target)):
        if x.shape != y.shape[:2] or np.prod(y.shape[2:]) != spec.outputs:
            raise ValueError("symbol/target shapes differ")
        if not np.isfinite(y).all() or (y < 0).any() or not np.allclose(y.sum(-1), 1, atol=1e-5):
            raise ValueError("targets must be distributions")
    p = init(spec, seed)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(lr))
    opt = optimizer.init(p)

    @jax.jit
    def step(p, opt, x, y):
        value, grad = jax.value_and_grad(loss, 1)(spec, p, x, y)
        delta, opt = optimizer.update(grad, opt, p)
        return optax.apply_updates(p, delta), opt, value

    x, y = jnp.asarray(train_x), jnp.asarray(train_target)
    rng = np.random.default_rng(seed)
    best, best_p, best_i = float("inf"), None, None
    start = time.perf_counter()
    for i in range(updates):
        ix = rng.integers(len(x), size=batch)
        p, opt, value = step(p, opt, x[ix], y[ix])
        if (i+1) % max(1, updates//10) == 0 or i == updates-1:
            v = float(loss(spec, p, val_x, val_target))
            if np.isfinite(v) and v < best:
                best, best_p, best_i = v, p, i+1
    if best_p is None:
        raise FloatingPointError("all validation losses nonfinite")
    jax.block_until_ready(value)
    seconds = time.perf_counter()-start
    xb, yb = x[ix], y[ix]
    jax.block_until_ready((xb, yb))
    t = time.perf_counter()
    p, opt, value = step(p, opt, xb, yb)
    jax.block_until_ready(value)
    step_seconds = time.perf_counter()-t
    return best_p, {"val_kl": best, "selected_update": best_i, "seconds": seconds,
                    "step_seconds": step_seconds, "seed": seed, "lr": lr}
