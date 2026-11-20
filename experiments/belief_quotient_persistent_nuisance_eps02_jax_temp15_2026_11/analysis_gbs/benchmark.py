"""CPU/GPU wall time for full-length fit steps, excluding compilation."""
import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import core


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    rows = []
    for tokens, actions, k in ((3, 3, 6), (9, 9, 10)):
        for kind in ("affine", "gru", "tanh", "hmm"):
            spec = core.Spec(kind, k, tokens, actions, actions, 128)
            params = core.init(spec, 0)
            x = jax.random.randint(jax.random.PRNGKey(1), (32, 1024), 0, tokens*actions)
            y = jnp.full((32, 1024, actions), 1/actions)
            optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(.001))
            state = optimizer.init(params)
            @jax.jit
            def step(params, state, x, y):
                value, grad = jax.value_and_grad(core.loss, 1)(spec, params, x, y)
                delta, state = optimizer.update(grad, state, params)
                return optax.apply_updates(params, delta), state, value
            params, state, value = step(params, state, x, y)
            jax.block_until_ready(value)
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                params, state, value = step(params, state, x, y)
                jax.block_until_ready(value)
                times.append(time.perf_counter()-start)
            row = {"tokens": tokens, "class": kind, "k": k, "batch": 32,
                   "length": 1024, "width": 128, "median_step_seconds": float(np.median(times))}
            rows.append(row)
            print(json.dumps(row), flush=True)
    print(json.dumps({"devices": list(map(str, jax.devices())), "steps": rows}))


if __name__ == "__main__":
    main()
