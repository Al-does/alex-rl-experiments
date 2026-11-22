"""Counterfactual cross-host fitted-map swap; original Eq.(4) rollout/budgets."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from .control_audit import digest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--maps", type=Path)
    args = parser.parse_args()
    if not args.label.replace("_", "").isalnum():
        parser.error("alphanumeric/underscore label required")
    os.environ.update(JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    import jax.numpy as jnp
    import numpy as np
    from . import run
    from experiments.reward_factors_replay_edit_2026_11 import transfer
    repo = Path(__file__).resolve().parents[2]
    root = repo / "artifacts" / f"block2_swap_{args.label}"
    (root / "maps").mkdir(parents=True, exist_ok=True)
    run.ROOT = root
    original_prepare, original_tables = run.prepare, transfer.task_tables
    snapshots, table_hashes = {}, {}
    active_test = None

    def prepare(rows, site):
        native = original_prepare(rows, site)
        flat = {f"{name}__{key}": np.asarray(value) for name, mapping in native.items() for key, value in mapping.items()}
        np.savez(root / "maps" / f"{active_test}.npz", **flat)
        snapshot = dict(native={key: digest(value) for key, value in flat.items()})
        if args.maps:
            with np.load(args.maps / f"{active_test}.npz") as loaded:
                supplied = {key: loaded[key] for key in loaded.files}
            if supplied.keys() != flat.keys() or any(supplied[key].shape != flat[key].shape or supplied[key].dtype != flat[key].dtype for key in flat):
                raise ValueError("counterfactual map keys, shapes or dtypes differ")
            replacement = {name: {key: jnp.asarray(supplied[f"{name}__{key}"]) for key in mapping} for name, mapping in native.items()}
            snapshot["supplied"] = {key: digest(value) for key, value in supplied.items()}
        else:
            replacement = native
        snapshots[active_test] = snapshot
        return replacement

    def tables(*arguments, **kwargs):
        result = original_tables(*arguments, **kwargs)
        table_hashes[active_test] = [digest(np.asarray(value, np.float32)) for value in result]
        return result

    run.prepare, transfer.task_tables = prepare, tables
    cpus = sorted(os.sched_getaffinity(0))[:4]
    try:
        for active_test in ("xor_eps02", "xor_v2"):
            for method in ("eq4", "eq4_joint"):
                run.worker((active_test, 42, method, 3, cpus, False))
    finally:
        run.prepare, transfer.task_tables = original_prepare, original_tables
    records = []
    for path in sorted((root / "results").glob("*.json")):
        row = json.loads(path.read_text())
        records.append(dict(test=row["test"], method=row["method"], checkpoint=row["checkpoint"]["sha256"], metrics=row["metrics"]))
    if len(records) != 4 or set(snapshots) != {"xor_eps02", "xor_v2"}:
        raise ValueError("four newly evaluated controls required")
    result = dict(label=args.label, map_source=str(args.maps) if args.maps else "native", snapshots=snapshots,
                  table_hashes=table_hashes, records=records,
                  wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root / "swap_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Cross-host map experiment complete: {args.label}, 4 full-budget controls", flush=True)


if __name__ == "__main__":
    main()
