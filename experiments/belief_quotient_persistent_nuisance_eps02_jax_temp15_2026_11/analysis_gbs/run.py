"""One condition per process; no provider provisioning is performed."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time

import jax
import numpy as np

from harness.context import RunContext, new_run_id
from harness.storage.b2 import B2StorageConfig, upload_run_artifacts
from . import compare, core, data


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parser(study):
    p = argparse.ArgumentParser(description=__doc__)
    conditions = ("variant_1", "variant_2", "variant_3") if study == data.LADDER else ("reward_both", "reward_factor_1")
    p.add_argument("--condition", choices=conditions, required=True)
    p.add_argument("--seeds", type=int, nargs="+", choices=(42, 43, 44), default=[42, 43, 44])
    p.add_argument("--control", action="store_true")
    p.add_argument("--classes", nargs="+", choices=("affine", "gru", "tanh", "hmm"), default=["affine", "gru", "tanh", "hmm"])
    p.add_argument("--ks", nargs="+", type=int, default=list(range(7 if study == data.LADDER else 11)))
    p.add_argument("--widths", nargs="+", type=int, default=[32, 128])
    p.add_argument("--lrs", nargs="+", type=float, default=[.001, .003])
    p.add_argument("--inits", type=int, default=3)
    p.add_argument("--updates", type=int, default=20000)
    p.add_argument("--val-interval", type=int, default=1000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--rollout-batch", type=int, default=32)
    p.add_argument("--train-episodes", type=int, default=2048)
    p.add_argument("--val-episodes", type=int, default=128)
    p.add_argument("--test-episodes", type=int, default=256)
    p.add_argument("--probe-updates", type=int, default=300)
    p.add_argument("--tolerance", type=float, default=1e-3, help="absolute test KL nats above best k, separately per class")
    p.add_argument("--run-name", default=None)
    p.add_argument("--smoke", action="store_true", help="ignored output location, not a scientific result")
    p.add_argument("--no-upload", action="store_true")
    return p


def main(study=data.LADDER):
    args = parser(study).parse_args()
    if min(args.train_episodes, args.val_episodes, args.test_episodes, args.inits,
           args.updates, args.val_interval, args.batch, args.rollout_batch, args.probe_updates) < 1:
        raise ValueError("counts must be positive")
    if min(args.ks) < 0 or min(args.widths) < 1 or min(args.lrs) <= 0 or args.tolerance < 0:
        raise ValueError("invalid sweep grid")
    here = data.ROOT / study / "analysis_gbs"
    name = args.run_name or f"{args.condition}-{'control' if args.control else 'policy'}-{new_run_id()}"
    if Path(name).name != name:
        raise ValueError("run-name must be a single path component")
    results = here / (".smoke" if args.smoke else "results") / name
    artifacts = here / "artifacts" / name
    results.mkdir(parents=True, exist_ok=False)
    artifacts.mkdir(parents=True, exist_ok=False)
    context = RunContext(here, results, artifacts, run_id=name, smoke=args.smoke)
    def revision(path):
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    manifest = {"schema_version": 2, "run_id": name, "status": "running", "command": sys.argv,
                "started_at": datetime.now(timezone.utc).isoformat(), "config": vars(args),
                "git": {"commit": revision(data.ROOT.parent),
                        "library": revision(data.ROOT.parent.parent / "rl-harness"),
                        "dirty": bool(subprocess.check_output(["git", "-C", str(data.ROOT.parent), "status", "--porcelain"], text=True).strip())},
                "versions": {"jax": jax.__version__, "numpy": np.__version__, "python": platform.python_version()},
                "devices": list(map(str, jax.devices()))}
    write(results / "run_manifest.json", manifest)
    rows = []
    try:
        for seed in args.seeds:
            start = time.perf_counter()
            agent_spec, process, params, recipe, provenance = data.restore(study, args.condition, seed, here / "artifacts" / "checkpoints")
            print(f"verified {args.condition} seed={seed} step={provenance['final_step']} sha256={provenance['sha256']}", flush=True)
            splits = {}
            for i, (split, count) in enumerate((("train", args.train_episodes), ("val", args.val_episodes), ("test", args.test_episodes))):
                splits[split] = data.collect(agent_spec, process, params, episodes=count,
                                            seed=seed*100+i+10000, batch=args.rollout_batch)
                truth = data.truth_targets(splits[split], process, study == data.LADDER)
                splits[split].update({f"truth_{key}": value for key, value in truth.items()})
                np.savez_compressed(artifacts / f"seed{seed}_{split}.npz", **splits[split])
            print(f"collected seed={seed} episodes={args.train_episodes}/{args.val_episodes}/{args.test_episodes} length=1024 seconds={time.perf_counter()-start:.3f}", flush=True)
            target = "all_predictive" if args.control else "policy"
            groups = process.n_actions if args.control else 1
            outputs = process.n_actions*process.n_tokens if args.control else process.n_actions
            # No truth object crosses this boundary: only integer histories and
            # predictive distributions are passed into core.fit.
            for kind, k in itertools.product(args.classes, args.ks):
                best = None
                candidates = []
                for width in args.widths:
                    spec = core.Spec(kind, k, process.n_tokens, process.n_actions, outputs, width, groups)
                    try:
                        group = core.fit_many(spec, splits["train"]["x"], splits["train"][target],
                                              splits["val"]["x"], splits["val"][target], seeds=tuple(range(args.inits)),
                                              lrs=tuple(args.lrs), updates=args.updates,
                                              val_interval=args.val_interval, batch=args.batch)
                    except FloatingPointError as error:
                        failed = {"width": width, "error": str(error)}
                        candidates.append(failed)
                        write(results / f"candidates_seed{seed}_{kind}_k{k}_width{width}.json", failed)
                        continue
                    for fitted, metric in group:
                        metric["width"] = width
                        candidates.append(metric)
                        if fitted is not None and (best is None or metric["val_kl"] < best[2]["val_kl"]):
                            best = spec, fitted, metric
                    write(results / f"candidates_seed{seed}_{kind}_k{k}_width{width}.json",
                          {"condition": args.condition, "seed": seed, "class": kind, "k": k,
                           "width": width, "candidates": [metric for _, metric in group]})
                if best is None:
                    write(results / f"seed{seed}_{kind}_k{k}_failed.json", {"candidates": candidates})
                    raise FloatingPointError(f"all candidates failed for {kind}, k={k}")
                spec, fitted, selected = best
                with (artifacts / f"seed{seed}_{kind}_k{k}.pkl").open("wb") as f:
                    pickle.dump({"spec": asdict(spec), "params": jax.tree.map(np.asarray, fitted)}, f)
                test_kl = float(core.loss(spec, fitted, splits["test"]["x"], splits["test"][target]))
                row = {"condition": args.condition, "seed": seed, "control": args.control,
                       "class": kind, "k": k, "test_kl": test_kl, "selected": selected,
                       "candidates": candidates, "checkpoint": provenance, "status": "fitted"}
                output = results / f"seed{seed}_{kind}_k{k}.json"
                write(output, row)
                train_g = np.asarray(core.states(spec, fitted, splits["train"]["x"]))
                test_g = np.asarray(core.states(spec, fitted, splits["test"]["x"]))
                ladder = study == data.LADDER
                train_truth = {key[6:]: value for key, value in splits["train"].items() if key.startswith("truth_")}
                test_truth = {key[6:]: value for key, value in splits["test"].items() if key.startswith("truth_")}
                row.update({"status": "completed",
                       "geometry": compare.geometry(train_g, test_g, train_truth, test_truth, mlp_updates=args.probe_updates),
                       "commutation": compare.commutation(spec, fitted, train_g, test_g, splits["train"]["belief"],
                                                          splits["test"]["belief"], splits["test"]["x"], process),
                       "hmm_matrices": compare.matrix_error(spec, fitted, process, ladder)})
                rows.append(row)
                write(output, row)
                print(f"fit condition={args.condition} seed={seed} class={kind} k={k} val_kl={selected['val_kl']:.8g} test_kl={test_kl:.8g} vmapped_step_seconds={selected['vmapped_step_seconds']:.6f} parallel_candidates={selected['parallel_candidates']}", flush=True)
        elbows = []
        for seed, kind in itertools.product(args.seeds, args.classes):
            subset = [r for r in rows if r["seed"] == seed and r["class"] == kind]
            minimum = min(r["test_kl"] for r in subset)
            elbows.append({"seed": seed, "class": kind, "best_test_kl": minimum,
                           "elbow_k": min(r["k"] for r in subset if r["test_kl"] <= minimum+args.tolerance)})
        write(results / "summary.json", {"elbow_rule": "smallest swept k with test KL <= minimum test KL within class + absolute tolerance (nats)",
              "tolerance": args.tolerance, "elbows": elbows, "fits": len(rows),
              "control_target": "all executed-action-conditional next-token distributions, equally weighted over actions" if args.control else None,
              "quotient_caveat": "Exact Reward-State Quotient filter driven by merged observations in every ladder condition; not an exact lumping of Full-State Quotient dynamics."})
        manifest["status"] = "completed"
        manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
        write(results / "run_manifest.json", manifest)
        if not args.no_upload and B2StorageConfig.from_env() is not None:
            manifest["remote_artifacts"] = upload_run_artifacts(context, experiment_module=f"experiments.{study}.analysis_gbs.run")
            write(results / "run_manifest.json", manifest)
        print(f"completed results={results} fits={len(rows)}", flush=True)
    except Exception as e:
        manifest.update(status="failed", error=repr(e))
        write(results / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
