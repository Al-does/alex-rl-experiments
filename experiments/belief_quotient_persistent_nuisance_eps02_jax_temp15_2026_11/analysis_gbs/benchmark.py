"""CPU/GPU wall time for full-length fit steps, excluding compilation."""
import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import core


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    rows = []
    for tokens, actions, k in ((3, 3, 6), (9, 9, 10)):
        for kind in ("affine", "gru", "tanh", "hmm"):
            spec = core.Spec(kind, k, tokens, actions, actions, 128)
            grid = [(seed, lr) for lr in (.001, .003) for seed in range(3)]
            params = jax.vmap(lambda seed: core.init(spec, seed))(jnp.asarray([seed for seed, _ in grid]))
            x = jax.random.randint(jax.random.PRNGKey(1), (32, 1024), 0, tokens*actions)
            y = jnp.full((32, 1024, actions), 1/actions)
            optimizer, single_step, parallel_step = core.train_steps(spec)
            state = jax.vmap(optimizer.init)(params)
            rates = jnp.asarray([lr for _, lr in grid])
            xs, ys = jnp.broadcast_to(x, (6, *x.shape)), jnp.broadcast_to(y, (6, *y.shape))
            # Both paths get the same six independent states and data. Warm both
            # compilations before timing a complete six-candidate update round.
            singles = [(jax.tree.map(lambda a: a[i], params), jax.tree.map(lambda a: a[i], state)) for i in range(6)]
            for i in range(6):
                p1, s1, value = single_step(*singles[i], x, y, rates[i])
                jax.block_until_ready(value)
                singles[i] = p1, s1
            params, state, value = parallel_step(params, state, xs, ys, rates)
            jax.block_until_ready(value)
            times, sequential_times = [], []
            for _ in range(args.repeats):
                start = time.perf_counter()
                for i in range(6):
                    p1, s1, value = single_step(*singles[i], x, y, rates[i])
                    jax.block_until_ready(value)
                    singles[i] = p1, s1
                sequential_times.append(time.perf_counter()-start)
                start = time.perf_counter()
                params, state, value = parallel_step(params, state, xs, ys, rates)
                jax.block_until_ready(value)
                times.append(time.perf_counter()-start)
            row = {"tokens": tokens, "class": kind, "k": k, "batch": 32,
                   "length": 1024, "width": 128, "candidates": 6,
                   "sequential_round_seconds": float(np.median(sequential_times)),
                   "vmapped_round_seconds": float(np.median(times)),
                   "throughput_speedup": float(np.median(sequential_times)/np.median(times))}
            rows.append(row)
            print(json.dumps(row), flush=True)
    print(json.dumps({"devices": list(map(str, jax.devices())), "steps": rows}))


if __name__ == "__main__":
    main()
