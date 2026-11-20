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


def train_steps(spec):
    """Independent Adam states and clipping, with a candidate-axis vmap."""
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(1.))

    def step(p, opt, x, y, lr):
        value, grad = jax.value_and_grad(loss, 1)(spec, p, x, y)
        delta, opt = optimizer.update(grad, opt, p)
        delta = jax.tree.map(lambda d: d*lr, delta)
        return optax.apply_updates(p, delta), opt, value

    return optimizer, jax.jit(step), jax.jit(jax.vmap(step))


def fit_many(spec, train_x, train_target, val_x, val_target, *, seeds=(0, 1, 2),
             lrs=(.001, .003), updates=20000, val_interval=1000, batch=32):
    """Train every init × rate concurrently, retaining each candidate's best."""
    if len(train_x) == 0 or len(val_x) == 0 or min(updates, val_interval, batch) < 1:
        raise ValueError("nonempty train/validation and positive updates/batch required")
    if not seeds or not lrs or min(lrs) <= 0:
        raise ValueError("nonempty seeds and positive learning rates required")
    for x, y in ((train_x, train_target), (val_x, val_target)):
        if x.shape != y.shape[:2] or np.prod(y.shape[2:]) != spec.outputs:
            raise ValueError("symbol/target shapes differ")
        if not np.isfinite(y).all() or (y < 0).any() or not np.allclose(y.sum(-1), 1, atol=1e-5):
            raise ValueError("targets must be distributions")
    grid = [(seed, lr) for lr in lrs for seed in seeds]
    p = jax.vmap(lambda seed: init(spec, seed))(jnp.asarray([s for s, _ in grid]))
    optimizer, _, step = train_steps(spec)
    opt = jax.vmap(optimizer.init)(p)
    rates = jnp.asarray([lr for _, lr in grid])
    validate = jax.jit(jax.vmap(lambda p, x, y: loss(spec, p, x, y), in_axes=(0, None, None)))
    val_x, val_target = jnp.asarray(val_x), jnp.asarray(val_target)
    x, y = jnp.asarray(train_x), jnp.asarray(train_target)
    rngs = [np.random.default_rng(seed) for seed, _ in grid]
    best, best_i = np.full(len(grid), np.inf), np.zeros(len(grid), dtype=int)
    best_p = p
    start = time.perf_counter()
    for i in range(updates):
        ix = np.stack([rng.integers(len(x), size=batch) for rng in rngs])
        p, opt, value = step(p, opt, x[ix], y[ix], rates)
        if (i+1) % val_interval == 0 or i == updates-1:
            values = np.asarray(validate(p, val_x, val_target))
            improved = np.isfinite(values) & (values < best)
            best = np.where(improved, values, best)
            best_i = np.where(improved, i+1, best_i)
            best_p = jax.tree.map(lambda old, new: jnp.where(
                jnp.asarray(improved).reshape((-1,) + (1,)*(new.ndim-1)), new, old), best_p, p)
    if not np.isfinite(best).any():
        raise FloatingPointError("all validation losses nonfinite")
    jax.block_until_ready(value)
    seconds = time.perf_counter()-start
    xb, yb = x[ix], y[ix]
    jax.block_until_ready((xb, yb))
    t = time.perf_counter()
    p, opt, value = step(p, opt, xb, yb, rates)
    jax.block_until_ready(value)
    step_seconds = time.perf_counter()-t
    results = []
    for i, (seed, lr) in enumerate(grid):
        metric = {"seed": seed, "lr": lr, "group_seconds": seconds,
                  "vmapped_step_seconds": step_seconds, "parallel_candidates": len(grid),
                  "updates": updates, "val_interval": val_interval}
        if np.isfinite(best[i]):
            metric.update(val_kl=float(best[i]), selected_update=int(best_i[i]))
            results.append((jax.tree.map(lambda a: a[i], best_p), metric))
        else:
            metric["error"] = "all validation losses nonfinite"
            results.append((None, metric))
    return results


def fit(spec, train_x, train_target, val_x, val_target, *, seed=0, lr=.003,
        updates=20000, val_interval=1000, batch=32):
    """Single-candidate convenience wrapper for focused identification tests."""
    return fit_many(spec, train_x, train_target, val_x, val_target, seeds=(seed,),
                    lrs=(lr,), updates=updates, val_interval=val_interval, batch=batch)[0]
