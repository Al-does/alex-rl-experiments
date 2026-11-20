"""Compare a historical core against current full/chunked fits on one device."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import jax
import jax.numpy as jnp
import numpy as np

from . import core


def baseline(ref):
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    path = Path(core.__file__).resolve().relative_to(root)
    source = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=root, text=True)
    module = types.ModuleType("gbs_profile_baseline")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{ref}:{path}", "exec"), module.__dict__)
    return module


def emit(row):
    print(json.dumps(row), flush=True)


def timings(old, args):
    cases = {"ladder2": (3, 3, 2), "ladder6": (3, 3, 6), "rr10": (9, 9, 10)}
    for tokens, actions, k in (cases[case] for case in args.cases):
        for control in (False, True):
            for kind in args.classes:
                row = dict(tokens=tokens, k=k, kind=kind, control=control, width=args.width,
                           candidates=6, batch=32, episode_length=1024)
                for label, module in (("before", old), ("after_full", core), ("after_chunk", core)):
                    if label not in args.modes:
                        continue
                    print(f"timing tokens={tokens} k={k} class={kind} control={control} mode={label}", file=sys.stderr, flush=True)
                    batch = args.batch if label == "after_chunk" else 32
                    groups = actions if control else 1
                    spec = module.Spec(kind, k, tokens, actions, actions*tokens if control else actions,
                                       args.width, groups)
                    params = jax.vmap(lambda seed: module.init(spec, seed))(jnp.array([0, 1, 2, 0, 1, 2]))
                    optimizer, _, step = module.train_steps(spec)
                    state = jax.vmap(optimizer.init)(params)
                    rates = jnp.array([.001]*3+[.003]*3)
                    x = jax.random.randint(jax.random.PRNGKey(3), (6, batch, 1024), 0, tokens*actions)
                    shape = (6, batch, 1024, actions, tokens) if control else (6, batch, 1024, actions)
                    y = jnp.full(shape, 1/(tokens if control else actions))
                    count = 1024//args.chunk_length if label == "after_chunk" else 1
                    if label == "after_chunk":
                        block = module.chunk_block(spec, optimizer)
                        xs = jnp.moveaxis(x.reshape(6, batch, count, args.chunk_length), 2, 0)
                        ys = jnp.moveaxis(y.reshape(6, batch, count, args.chunk_length, *shape[3:]), 2, 0)
                        carry = jnp.zeros((6, batch, k))
                        def execute(p, s):
                            p, s, _, value = block(p, s, carry, xs, ys,
                                                   jnp.arange(count)*args.chunk_length, rates)
                            return p, s, value
                    else:
                        def execute(p, s):
                            return step(p, s, x, y, rates)
                    start = time.perf_counter()
                    if label == "after_chunk":
                        jax.block_until_ready((params, state, carry, xs, ys, rates))
                        block = block.lower(params, state, carry, xs, ys, jnp.arange(count)*args.chunk_length, rates).compile()
                    else:
                        jax.block_until_ready((params, state, x, y, rates))
                        step = step.lower(params, state, x, y, rates).compile()
                    compile_seconds = time.perf_counter()-start
                    print(f"compiled seconds={compile_seconds:.6f}", file=sys.stderr, flush=True)
                    start = time.perf_counter()
                    params, state, value = execute(params, state)
                    jax.block_until_ready(value)
                    print(f"warmup seconds={time.perf_counter()-start:.6f}", file=sys.stderr, flush=True)
                    samples = []
                    for _ in range(args.repeats):
                        start = time.perf_counter()
                        params, state, value = execute(params, state)
                        jax.block_until_ready(value)
                        samples.append((time.perf_counter()-start)/count)
                    vx, vy = x[0, :32].repeat(4, 0), y[0, :32].repeat(4, 0)
                    validate = jax.jit(jax.vmap(lambda p: module.loss(spec, p, vx, vy)))
                    start = time.perf_counter()
                    jax.block_until_ready(validate(params))
                    validation_compile = time.perf_counter()-start
                    start = time.perf_counter()
                    jax.block_until_ready(validate(params))
                    validation = time.perf_counter()-start
                    seconds = float(np.median(samples))
                    row[label] = dict(batch=batch, chunk_length=args.chunk_length if label == "after_chunk" else 1024,
                                      step_seconds=seconds, training_group_seconds=seconds*args.updates,
                                      validation_seconds=validation, compile_seconds=compile_seconds,
                                      validation_compile_seconds=validation_compile,
                                      projected_group_seconds=seconds*args.updates+
                                      validation*int(np.ceil(args.updates/args.val_interval))+
                                      compile_seconds+validation_compile)
                    jax.clear_caches()
                emit(row)


def comparison(old, args):
    with np.load(args.compare_data) as arrays:
        arrays = dict(arrays)
    for label, module in (("before", old), ("after", core)):
        if args.after_only and label == "before":
            continue
        spec = module.Spec("gru", 2, 3, 3, 3, args.width)
        kwargs = {"batch": 32} if label == "before" else {"chunk_length": args.chunk_length, "batch": args.batch}
        start = time.perf_counter()
        results = module.fit_many(spec, arrays["train_x"], arrays["train_y"], arrays["val_x"], arrays["val_y"],
                                  updates=args.updates, val_interval=args.val_interval, **kwargs)
        metrics = [metric for _, metric in results]
        emit(dict(label=label, seconds=time.perf_counter()-start, metrics=metrics,
                  best_val_kl=min(m["val_kl"] for m in metrics)))
        jax.clear_caches()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--compare-data", type=Path, help="NPZ with train_x/train_y/val_x/val_y policy arrays")
    parser.add_argument("--after-only", action="store_true", help="reuse an already measured validation baseline")
    parser.add_argument("--modes", nargs="+", choices=("before", "after_full", "after_chunk"),
                        default=["before", "after_full", "after_chunk"])
    parser.add_argument("--classes", nargs="+", default=["affine", "gru", "tanh", "hmm"])
    parser.add_argument("--cases", nargs="+", choices=("ladder2", "ladder6", "rr10"),
                        default=["ladder2", "ladder6", "rr10"])
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--chunk-length", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--updates", type=int, default=20000)
    parser.add_argument("--val-interval", type=int, default=1000)
    args = parser.parse_args()
    old = baseline(args.baseline_ref)
    emit(dict(devices=list(map(str, jax.devices())), baseline_ref=args.baseline_ref, config=vars(args) | {"compare_data": str(args.compare_data)}))
    (comparison if args.compare_data else timings)(old, args)


if __name__ == "__main__":
    main()
