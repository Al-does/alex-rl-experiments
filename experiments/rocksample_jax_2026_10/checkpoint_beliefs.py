"""Probe every recorded checkpoint of a RockSample sweep trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np

from experiments.rocksample_jax_2026_10 import belief_probe as bp
from experiments.rocksample_jax_2026_10 import env as rs
from experiments.rocksample_jax_2026_10 import model as tm


def parameter_hash(params: dict) -> str:
    digest = hashlib.sha256()
    for value in jax.tree.leaves(params):
        array = np.asarray(value)
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def evaluate(
    spec: tm.TransformerSpec, env: rs.RockSampleParams, params: dict, init: dict,
    *, episodes: int, full_battery: bool,
) -> dict:
    reports = {}
    for name, scripted in (("on_policy", False), ("common_checks", True)):
        print(f"  {name}: collect {episodes} episodes", flush=True)
        rows = bp.collect(
            spec, env, params, init, episodes=episodes, key_seed=9131,
            scripted=scripted,
        )
        reports[name], _ = bp.analyze(
            bp.targets(rows, env), joint_battery=full_battery and not scripted,
        )
        reports[name]["history_sha256"] = hashlib.sha256(
            b"".join(rows[key].tobytes() for key in ("obs", "action", "valid"))
        ).hexdigest()
    return reports


def independent_behavior(
    spec: tm.TransformerSpec, env: rs.RockSampleParams, params: dict,
    *, episodes: int,
) -> dict:
    rows = bp.collect(
        spec, env, params, params, episodes=episodes, key_seed=27183,
    )
    data = bp.targets(rows, env)
    good = rows["qualities"][:, 0].astype(bool)
    sampled = data.samples > 0
    result = bp.behavior(data)
    result["key_seed"] = 27183
    result["initial_good_counts"] = good.sum(axis=0).tolist()
    result["sample_given_initial_good"] = (
        (sampled & good).sum(axis=0) / good.sum(axis=0)
    ).tolist()
    result["sample_given_initial_bad"] = (
        (sampled & ~good).sum(axis=0) / (~good).sum(axis=0)
    ).tolist()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_results", type=Path)
    parser.add_argument("--seed", type=int, choices=range(4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--updates", nargs="+", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    records = bp.checkpoint_records(args.run_results, args.seed)
    if args.updates:
        available = {record["update"] for record in records}
        if not set(args.updates) <= available:
            raise ValueError("requested update not present in checkpoint metadata")
        selected = [r for r in records if r["update"] in args.updates]
    else:
        selected = records
    directory = bp.ROOT / "artifacts/belief_checkpoints" / args.run_results.name
    directory.mkdir(parents=True, exist_ok=True)
    recipe = json.loads((args.run_results / "resolved_recipe.json").read_text())
    if recipe["instance"] != [5, 7] or recipe["num_seeds"] != 4:
        raise ValueError("expected the matched four-seed RockSample[5,7] sweep")
    arm = next(a for a in recipe["arms"] if a["name"] == "d128_kl0.1")
    env = rs.RockSampleParams.from_instance()
    spec = tm.TransformerSpec(
        env.obs_dim, env.num_actions, arm["d_model"], arm["n_layers"],
        arm["n_heads"], arm["context_len"],
    )
    init = bp.initialization(spec, args.seed, recipe["seed"], recipe["num_seeds"])
    episodes = 16 if args.smoke else 512
    metadata = {
        "schema": 2,
        "run_id": args.run_results.name,
        "seed": args.seed,
        "smoke": args.smoke,
        "episodes_per_distribution": episodes,
        "env": asdict(env), "spec": asdict(spec),
        "experiment_revision": bp.revision(bp.ROOT),
        "harness_revision": bp.revision(bp.ROOT.parents[2] / "rl-harness"),
        "jax_version": jax.__version__,
        "sources": {
            str(path.relative_to(bp.ROOT)): bp.sha256(path)
            for path in (Path(__file__), bp.ROOT / "belief_probe.py")
        },
        "recipe_sha256": bp.sha256(args.run_results / "resolved_recipe.json"),
        "checkpoint_summary_sha256": bp.sha256(args.run_results / "summary.json"),
        "initialization_parameter_sha256": parameter_hash(init),
        "protocol": "512 complete episodes/distribution, 256 fit/256 test, split seed 831; "
        "CV seed 551 uses fit episodes only; raw affine predictions; 200 fixed-probe "
        "episode bootstrap resamples. Full joint null battery at final only. "
        "On-policy histories vary with the checkpoint; common-check histories are "
        "identical across every encoder. Reset rows included; exit ends histories.",
    }
    print(f"{args.run_results.name} seed {args.seed}: initialization", flush=True)
    report = {
        **metadata, "update": 0, "env_steps": 0,
        "source": {"kind": "exact sweep/PPO initialization reconstruction"},
        **evaluate(spec, env, init, init, episodes=episodes, full_battery=False),
    }
    (args.output / "update000.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    for record in selected:
        update = record["update"]
        print(f"{args.run_results.name} seed {args.seed}: update {update}", flush=True)
        params, steps, _, provenance = bp.checkpoint(
            directory, args.seed, args.download, run_results=args.run_results,
            update=update,
        )
        if params["input"]["w"].shape != (env.obs_dim, spec.d_model) or len(
            params["blocks"]
        ) != spec.n_layers:
            raise ValueError("checkpoint architecture disagrees with recipe")
        final = update == records[-1]["update"]
        report = {
            **metadata, "update": update, "env_steps": steps,
            "source": provenance, "parameter_sha256": parameter_hash(params),
            **evaluate(
                spec, env, params, init, episodes=episodes,
                full_battery=final and not args.smoke,
            ),
        }
        if final:
            print("  independent rollout confirmation", flush=True)
            report["independent_behavior"] = independent_behavior(
                spec, env, params, episodes=16 if args.smoke else 4096,
            )
        path = args.output / f"update{update:03}.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
