"""Reproduce the archived probe and replay the same histories across trunks."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

# The archived adapter configures the JAX CPU runtime and BLAS thread limits.
from experiments.algorithm_generality_probes_eps02_2026_11 import analyze as archive
from experiments.algorithm_generality_probes_eps02_2026_11.paper_probe import flatten
import jax
import numpy as np

from analysis.probes.linear import fit_affine_probe
from experiments.belief_quotient_history_jax_2026_11 import env, models
from harness.storage.b2 import B2StorageConfig

ROOT = Path(__file__).resolve().parent
SEEDS = (42, 44, 46)
ALGORITHMS = ("reinforce", "ppo", "sac_autoalpha")
CHECKPOINTS = ROOT / "artifacts" / "checkpoints"


def find_runs(variant, seed):
    runs = {
        run["algorithm"]: ROOT.parents[1] / run["results_path"]
        for run in archive.discover()
        if run["variant"] == variant and run["seed"] == seed
    }
    pattern = (
        "belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11/"
        f"variant_{variant}_transformer/results/*/run_manifest.json"
    )
    for path in (ROOT.parent).glob(pattern):
        manifest = archive.read(path)
        if (manifest["runtime"]["seed"] == seed and not manifest["runtime"]["smoke"]
                and manifest["status"] == "completed"):
            if "reinforce" in runs:
                raise ValueError(f"Multiple REINFORCE checkpoints for {seed=} {variant=}")
            runs["reinforce"] = path.parent
    if set(runs) != set(ALGORITHMS):
        raise ValueError(f"Incomplete cohort for {seed=} {variant=}")
    return runs


def restore(run, names, client):
    manifest = archive.read(run / "run_manifest.json")
    remote = manifest["remote_artifacts"]
    body = client.get_object(Bucket=remote["bucket"], Key=remote["canonical_manifest_key"])["Body"]
    with body:
        durability = json.load(body)
    arrays, metadata = {}, {}
    for name in names:
        prefix = "step" if name == "step" else "critic_step"
        relative = f"checkpoints/{prefix}_{archive.FINAL:010d}.npz"
        entry, = [item for item in durability["files"] if item["relative_path"] == relative]
        path = CHECKPOINTS / manifest["run_id"] / Path(relative).name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            client.download_file(remote["bucket"], entry["key"], str(path))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (digest, path.stat().st_size) != (entry["sha256"], entry["size_bytes"]):
            raise ValueError(f"Checkpoint integrity mismatch: {path}")
        with np.load(path, allow_pickle=False) as saved:
            arrays[name] = dict(saved)
        metadata[name] = {key: entry[key] for key in ("key", "sha256", "size_bytes")}
    return arrays, {
        "run_id": manifest["run_id"], "git": manifest["git"], "step": archive.FINAL,
        "canonical_manifest": remote["canonical_manifest_key"], "files": metadata,
    }


def load(variant, seed):
    client = B2StorageConfig.from_env().s3_client()
    reference = None
    trunks, checkpoints = {}, {}
    for algorithm, run in find_runs(variant, seed).items():
        recipe = archive.read(run / "resolved_recipe.json")
        spec = models.ModelSpec(**recipe["model"])
        settings = (spec, recipe["process"], recipe["training"]["temperature"],
                    recipe["training"]["episode_length"], recipe["training"]["gamma"])
        if reference is not None and reference != settings:
            raise ValueError(f"Training environments/config differ for {seed=} {variant=}")
        reference = settings
        names = ("step",)
        if algorithm == "sac_autoalpha":
            names = (*names, "critic")
        saved, checkpoints[algorithm] = restore(run, names, client)
        template = models.init_params(spec, jax.random.PRNGKey(0))
        trunks[algorithm] = archive.actor_arrays(saved["step"], template, algorithm)
        if algorithm == "sac_autoalpha":
            critics = saved["critic"]
            for index in range(2):
                prefix = f"['critics'][{index}]"
                weights = {key.removeprefix(prefix): value for key, value in critics.items()
                           if key.startswith(prefix)}
                trunks[f"q{index + 1}"] = archive.actor_arrays(weights, template, "sac_autoalpha")
            alpha = float(critics["['alpha']"])
            np.testing.assert_allclose(alpha, np.exp(critics["['log_alpha']"]), rtol=1e-6)
            if not 0 < alpha < np.inf or recipe["training"]["exploration_epsilon"] != 0:
                raise ValueError("Unexpected SAC entropy or exploration parameters")
    spec, process, temperature, length, gamma = reference
    return spec, env.make_process(**process), temperature, length, gamma, alpha, trunks, checkpoints


def targets(belief):
    return np.column_stack([belief[:, 2], belief[:, 0] - belief[:, 1]]).astype(np.float64)


def probe(features, belief):
    train, test = map(targets, belief)
    weight, bias = fit_affine_probe(features[0], train, ridge=1e-6)
    sse = np.square(features[1] @ weight + bias - test).sum(0)
    denominator = np.square(test - test.mean(0)).sum(0)
    return {"sse": sse.tolist(), "sst": denominator.tolist(),
            "error": (sse / denominator).tolist(), "test_rows": len(test)}


def window(value, start, end, rows):
    return flatten(value[:, :end], start, rows)


def matched(spec, process, trunks, seed, length, smoke, reference=None):
    runtime, rollout, replay = archive.runtime()
    keys = runtime.random.split(runtime.random.fold_in(runtime.random.PRNGKey(seed), 2))
    counts = (1920, 2880) if smoke else (60000, 80000)
    protocols = {"64:600": (600, (None, None), (4, 6) if smoke else (32, 40))}
    if reference is None:
        protocols[f"64:{length}"] = (length, counts, tuple(-(-n // (length - 64)) for n in counts))
    records = {}
    for role in ALGORITHMS:
        records[role] = {}
        for name, (stop, rows, episodes) in protocols.items():
            streams = [rollout(spec, process, trunks[role], key, n, length)
                       for key, n in zip(keys, episodes)]
            y = [window(r["belief"], 64, stop, n) for r, n in zip(streams, rows)]
            scores = {}
            if reference is not None:
                from .value_reference import build_targets, score_targets
                values, variance_shares = build_targets(reference, y)
            for trunk, params in trunks.items():
                features = []
                for r, n in zip(streams, rows):
                    x = replay(spec, params, r["obs"], "embedding")
                    if trunk == role:
                        np.testing.assert_allclose(x, r["embedding"], atol=1e-4, rtol=0)
                        x = r["embedding"]
                    features.append(window(x, 64, stop, n))
                scores[trunk] = (probe(features, y) if reference is None
                                 else score_targets(features, values))
            if reference is None:
                for result in scores.values():
                    np.testing.assert_array_equal(result["sst"], scores[role]["sst"])
                records[role][name] = scores
            else:
                records[role][name] = {"scores": scores, "nonlinear_variance_shares": variance_shares}
    return records


def run_one(task):
    variant, seed, cpu, smoke, soft, hard_reference, output = task
    os.sched_setaffinity(0, {cpu})
    spec, process, temperature, length, gamma, alpha, trunks, checkpoints = load(variant, seed)
    with jax.default_matmul_precision("highest"):
        result = {"variant": variant, "seed": seed, "checkpoints": checkpoints,
                  "coordinates": ["rho", "delta"], "learned_alpha": alpha,
                  "jax_version": jax.__version__, "numpy_version": np.__version__}
        if soft:
            from .soft import evaluate
            result["soft"] = evaluate(spec, process, trunks, temperature, gamma, alpha, seed, length, smoke)
        elif hard_reference is not None:
            from .value_reference import restore_reference
            reference = restore_reference(hard_reference, variant, process, gamma)
            result["hard_reference"] = {
                **reference.metadata, "matched": matched(spec, process, trunks, seed, length, smoke, reference)
            }
        else:
            result["matched"] = matched(spec, process, trunks, seed, length, smoke)
    path = output / f"variant{variant}_seed{seed}.json"
    archive.write(path, result)
    print(f"Completed {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--soft", action="store_true", help="run SAC soft-value targets")
    mode.add_argument("--hard-reference", type=Path, help="run matched probes on original PBVI targets")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.glob("variant*_seed*.json")):
        parser.error("use a clean output directory to avoid replacing previous results")
    cpus = sorted(os.sched_getaffinity(0))
    jobs = [(2, 42)] if args.smoke else [(v, s) for v in (1, 2, 3) for s in SEEDS]
    workers = min(len(cpus), len(jobs), args.workers)
    if workers < 1:
        parser.error("workers must be positive")
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                             max_tasks_per_child=1) as pool:
        for start in range(0, len(jobs), workers):
            tasks = [(v, s, cpus[idx], args.smoke, args.soft, args.hard_reference, args.output)
                     for idx, (v, s) in enumerate(jobs[start:start + workers])]
            list(pool.map(run_one, tasks))


if __name__ == "__main__":
    main()
