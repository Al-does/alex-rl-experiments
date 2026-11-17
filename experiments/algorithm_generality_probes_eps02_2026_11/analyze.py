"""SHA-verified CPU probes with paper-compatible streams and matched initialization."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[name] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

import numpy as np

from .paper_probe import coarse_filter, fit_score, flatten, runtime
from .paper_coarse import merged_filter
from .paper_constraints import constrain_predictions, constraint_method
from .paper_within import refit, score, null_basis

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SEEDS = (42, 44, 46, 49, 52)
FINAL = 2012217344
VERSION = "eps02-algorithm-generality-v1"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def discover():
    result = []
    for algorithm in ("ppo", "sac_autoalpha"):
        study = f"belief_quotient_persistent_nuisance_eps02_jax_temp15_{algorithm}_2026_11"
        for variant in (1, 2, 3):
            runs = []
            for manifest in (REPO / "experiments" / study / f"variant_{variant}_transformer" / "results").glob("*/run_manifest.json"):
                m = read(manifest)
                seed = m["runtime"]["seed"]
                if seed not in SEEDS or m["runtime"]["smoke"]:
                    continue
                if m["status"] != "completed" or m["remote_artifacts"]["status"] != "completed":
                    raise ValueError(f"incomplete run {manifest}")
                recipe = read(manifest.with_name("resolved_recipe.json"))
                if recipe["training"]["temperature"] != 1.5 or recipe["environment_revision"] != "persistent_nuisance_eps02":
                    raise ValueError(f"unexpected recipe {manifest}")
                runs.append(dict(algorithm=algorithm, variant=variant, seed=seed, run_id=m["run_id"],
                                 results_path=str(manifest.parent.relative_to(REPO))))
            if sorted(r["seed"] for r in runs) != list(SEEDS):
                raise ValueError(f"expected exactly five seeds: {study} variant {variant}")
            result.extend(sorted(runs, key=lambda r: r["seed"]))
    return result


def actor_arrays(saved, template, algorithm):
    import jax
    leaves, tree = jax.tree_util.tree_flatten_with_path(template)
    expected = {jax.tree_util.keystr(k): v for k, v in leaves}
    extras = {"['value']['w']", "['value']['b']"} if algorithm == "ppo" else set()
    if set(saved) != set(expected) | extras:
        raise ValueError(f"parameter keys differ: missing={set(expected)-set(saved)}, extra={set(saved)-set(expected)}")
    if extras and (saved["['value']['w']"].shape != (64, 1) or saved["['value']['b']"].shape != (1,)):
        raise ValueError("invalid PPO value head shape")
    values = []
    for key, leaf in leaves:
        value = saved[jax.tree_util.keystr(key)]
        if value.shape != leaf.shape or value.dtype != leaf.dtype or not np.isfinite(value).all():
            raise ValueError(f"invalid actor leaf {jax.tree_util.keystr(key)}")
        values.append(jax.numpy.asarray(value))
    return jax.tree_util.tree_unflatten(tree, values)


def restore(run, step, spec):
    import jax
    from harness.storage.b2 import B2StorageConfig
    from experiments.belief_quotient_history_jax_2026_11 import models
    remote = read(REPO / run["results_path"] / "run_manifest.json")["remote_artifacts"]
    client = B2StorageConfig.from_env().s3_client()
    durability = json.loads(client.get_object(Bucket=remote["bucket"], Key=remote["canonical_manifest_key"])["Body"].read())
    relative = f"checkpoints/step_{step:010d}.npz"
    entries = [f for f in durability["files"] if f["relative_path"] == relative]
    if len(entries) != 1:
        raise ValueError(f"missing or duplicate durable checkpoint {relative}")
    entry = entries[0]
    path = ROOT / "artifacts" / "checkpoints" / run["run_id"] / Path(relative).name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        client.download_file(remote["bucket"], entry["key"], str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
        raise ValueError(f"checkpoint integrity mismatch: {path}")
    with np.load(path) as saved:
        params = actor_arrays(saved, models.init_params(spec, jax.random.PRNGKey(0)), run["algorithm"])
    return params, dict(step=step, sha256=digest, size_bytes=entry["size_bytes"],
                        bucket=remote["bucket"], key=entry["key"], manifest_key=remote["canonical_manifest_key"])


def within(params, features, beliefs, classes):
    targets = [np.concatenate([b, b[:, 2:3], b[:, :1]-b[:, 1:2]], axis=1) for b in beliefs]
    w, bias = np.asarray(params["policy"]["w"], float), np.asarray(params["policy"]["b"], float)
    basis = null_basis(w.T)
    result = {"fit_counts": np.bincount(classes[0], minlength=3).tolist(),
              "test_counts": np.bincount(classes[1], minlength=3).tolist(),
              "head_null_dimension": basis.shape[1], "metrics": {}}
    for name, x in {"full_h": features, "head_blind": [h @ basis for h in features],
                    "logits": [h @ w + bias for h in features]}.items():
        predicted = refit(x[0], targets[0], x[1], *classes)
        valid = np.isfinite(predicted).all(1)
        result["metrics"][name] = {}
        for target, sl in (("belief", slice(0, 3)), ("rho", slice(3, 4)), ("delta", slice(4, 5))):
            result["metrics"][name][target] = dict(score(targets[1][valid, sl], predicted[valid, sl], classes[1][valid]),
                excluded_rows=int((~valid).sum()), per_class={str(k): score(targets[1][valid & (classes[1] == k), sl],
                    predicted[valid & (classes[1] == k), sl]) for k in range(3)})
    result["one_class"] = bool(min(np.count_nonzero(result["fit_counts"]), np.count_nonzero(result["test_counts"])) < 2)
    return result


def run_one(task):
    run, smoke, cpu = task
    os.sched_setaffinity(0, {cpu})
    start = time.monotonic()
    jax, rollout, replay = runtime()
    from experiments.belief_quotient_history_jax_2026_11.models import ModelSpec
    from experiments.belief_quotient_history_jax_2026_11.env import make_process
    from analysis.probes.linear import global_mse_metrics, r2_score
    output = ROOT / ("artifacts/smoke" if smoke else "artifacts/report_details/probes/agents") / f"{run['algorithm']}_variant{run['variant']}_seed{run['seed']}.json"
    if output.exists():
        cached = read(output)
        if cached["version"] != VERSION:
            raise ValueError("incompatible cached output")
        return str(output)
    recipe = read(REPO / run["results_path"] / "resolved_recipe.json")
    spec, process = ModelSpec(**recipe["model"]), make_process(**recipe["process"])
    params, final_provenance = restore(run, FINAL, spec)
    init, init_provenance = restore(run, 0, spec)
    counts = (960, 1920) if smoke else (60000, 80000)
    with jax.default_matmul_precision("highest"):
        keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(run["seed"]), 2))
        raw = [rollout(spec, process, params, key, -(-n // 960), 1024) for key, n in zip(keys, counts)]
        features = [flatten(r["embedding"], 64, n) for r, n in zip(raw, counts)]
        beliefs = [flatten(r["belief"], 64, n) for r, n in zip(raw, counts)]
        classes = [flatten(r["action"], 64, n) for r, n in zip(raw, counts)]
        targets = [dict(belief=b, rho=b[:, 2:3], delta=b[:, :1]-b[:, 1:2]) for b in beliefs]
        if run["variant"] in (2, 3):
            merged = [flatten(merged_filter(np.asarray(r["obs"]), recipe["process"])[..., 2:3], 64, n)
                      for r, n in zip(raw, counts)]
            if run["variant"] == 2:
                exact = [flatten(coarse_filter(np.asarray(r["obs"]), recipe["process"])[..., None], 64, n)
                         for r, n in zip(raw, counts)]
                for a, b in zip(exact, merged):
                    np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
            for t, c in zip(targets, merged):
                t["coarse_b" if run["variant"] == 2 else "merged_token_rho"] = c
        scalars = {}
        for name in targets[0]:
            metrics, predicted = fit_score(*features, targets[0][name], targets[1][name])
            record = dict(raw=metrics, error=metrics["error"])
            if metrics["error"] > 1:
                target = "coarse_b" if name == "merged_token_rho" else name
                bounded = constrain_predictions(predicted, target)
                constrained = global_mse_metrics(bounded, targets[1][name])
                constrained["r_squared"] = r2_score(bounded, targets[1][name])
                constrained["error"] = 1-constrained["r_squared"]
                record.update(error=constrained["error"], constrained=constrained, method=constraint_method(target))
            scalars[name] = record
        own = np.asarray(replay(spec, params, raw[1]["obs"], "embedding"))
        replay_error = float(np.max(np.abs(own - np.asarray(raw[1]["embedding"]))))
        if replay_error >= 1e-4:
            raise ValueError(f"self-replay differs: {replay_error}")
        matched_features = [flatten(replay(spec, init, r["obs"], "embedding"), 64, n) for r, n in zip(raw, counts)]
        trained, matched = within(params, features, beliefs, classes), within(init, matched_features, beliefs, classes)
    manifest = read(REPO / run["results_path"] / "run_manifest.json")
    result = dict(version=VERSION, **run, smoke=smoke, checkpoints=[final_provenance, init_provenance],
                  scalars=scalars, trained=trained, matched_init=matched, final_self_replay_max_abs=replay_error,
                  greedy_reward=float(flatten(raw[1]["reward"], 64, counts[1]).mean()),
                  fit_rows=counts[0], test_rows=counts[1], warmup=64, ridge=1e-6,
                  coarse_target="exact quotient" if run["variant"] == 2 else "merged-token filter; NOT exact quotient",
                  cpu_affinity=sorted(os.sched_getaffinity(0)), wall_seconds=time.monotonic()-start,
                  training_revisions=manifest["git"], analysis_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  analysis_source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob("*.py"))},
                  harness_revision=subprocess.check_output(["git", "-C", str(REPO.parent / "rl-harness"), "rev-parse", "HEAD"], text=True).strip(),
                  jax_version=jax.__version__, numpy_version=np.__version__)
    write(output, result)
    print(f"DONE {run['algorithm']} variant{run['variant']} seed{run['seed']} {result['wall_seconds']:.2f}s", flush=True)
    jax.clear_caches()
    return str(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    cpus = sorted(os.sched_getaffinity(0))
    if not 1 <= args.workers <= len(cpus):
        parser.error("workers must fit available CPU affinity")
    runs = discover()
    if args.smoke:
        run_one((next(r for r in runs if r["algorithm"] == "ppo" and r["variant"] == 2), True, cpus[0]))
        return
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1) as pool:
        for start in range(0, len(runs), args.workers):
            list(pool.map(run_one, [(r, False, cpus[i]) for i, r in enumerate(runs[start:start+args.workers])], chunksize=1))


if __name__ == "__main__":
    main()
