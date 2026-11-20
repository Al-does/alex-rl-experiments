"""Run on CPU: python -m experiments.belief_quotient_replay_edit_2026_11.run."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
STUDY = "belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11"
HISTORICAL = "45517b13a67d746048a002f9db52ea55a654b219"
SEEDS = (42, 43, 44)
PHASE_SEEDS = dict(fit=42, selection=43, test=44, tune=1042, confirmation=2042)


def archive(path):
    full_path = f"experiments/{STUDY}/{path}"
    raw = subprocess.check_output(["git", "show", f"{HISTORICAL}:{full_path}"], cwd=REPO)
    return json.loads(raw), dict(path=full_path, revision=HISTORICAL, sha256=hashlib.sha256(raw).hexdigest())


def worker(task):
    variant, seed, cpu, smoke = task
    os.sched_setaffinity(0, {cpu})
    start = time.monotonic()

    import jax
    import numpy as np

    from experiments.algorithm_generality_probes_eps02_2026_11.analyze import FINAL, read, restore, write
    from experiments.algorithm_generality_probes_eps02_2026_11.steer_core import AFFINE_RCOND
    from experiments.belief_quotient_history_jax_2026_11.models import ModelSpec
    from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.environment import ladder_process

    from . import closed_loop, replay

    output = ROOT / ("artifacts/smoke" if smoke else "results") / f"v{variant}_seed{seed}.json"
    if output.exists():
        raise FileExistsError(output)
    leaf = f"variant_{variant}_transformer"
    old_report, old_source = archive(f"{leaf}/results/belief_steering_v2_seed{seed}_stable.json")
    path = Path("experiments", STUDY, leaf, "results", old_report["checkpoint"]["run_id"])
    manifest = read(REPO / path / "run_manifest.json")
    recipe = read(REPO / path / "resolved_recipe.json")
    if (manifest["status"], manifest["runtime"]["seed"], recipe["environment_revision"], recipe["training"]["temperature"]) != (
            "completed", seed, "persistent_nuisance_eps02", 1.5):
        raise ValueError(f"unexpected checkpoint recipe: {path}")
    spec = ModelSpec(**recipe["model"])
    process = ladder_process(variant)
    for field, value in process._asdict().items():
        np.testing.assert_allclose(value, recipe["process"][field], atol=1e-6)
    src = dict(results_path=str(path), algorithm="reinforce", run_id=old_report["checkpoint"]["run_id"])
    print(f"START {variant=} {seed=}", flush=True)
    with jax.default_matmul_precision("highest"):
        final, ckpt = restore(src, FINAL, spec)
        init, init_ckpt = restore(src, 0, spec)
        if ckpt["sha256"] != old_report["checkpoint"]["sha256"]:
            raise ValueError(f"final checkpoint differs from archive: {path}")
        episodes, length = (2, 256) if smoke else (14, 1024)
        def history(params, split):
            return replay.records(spec, process, params, PHASE_SEEDS[split], episodes, length)
        fit, select, test = (history(final, s) for s in ("fit", "selection", "test"))
        np.testing.assert_allclose(replay.filtered(process, test["obs"], variant)[..., :3], test["belief"], atol=1e-5)
        a = dict(final=replay.invariance(spec, final, process, variant, fit, test),
                 init=replay.invariance(spec, init, process, variant, history(init, "fit"), history(init, "test")))
        print(f"A {variant=} {seed=} {time.monotonic()-start:.1f}s", flush=True)
        def pairs(record, offset, patch):
            return replay.collect_pairs(spec, final, process, variant, record, 7000 + offset, patch)
        _, fit_pairs = pairs(fit, 0, False)
        select_b, select_pairs = pairs(select, 1, variant == 2)
        b, test_pairs = pairs(test, 2, True)
        encoders, c = replay.fit_encoders(fit_pairs, select_pairs, test_pairs)
        print(f"BC {variant=} {seed=} {time.monotonic()-start:.1f}s", flush=True)
        d = None
        if variant == 2:
            d = closed_loop.compare(spec, process, final, fit, encoders, select_b, old_report, smoke)
            if not smoke:
                actual = d["sites"]["3"]
                archived_parity = {}
                for key in ("intact", "random"):
                    ref = next(r for r in old_report["confirmation"] if r["condition"] == key)["metrics"]
                    np.testing.assert_equal(actual[key]["occupancy"], ref["occupancy"])
                for mode in ("rho_flip", "rho_point", "mirror"):
                    if mode == "mirror":
                        ref = next(r for r in old_report["confirmation"]
                                   if r["site"] == 3 and r["condition"] == "curve_mirror" and r["strength"] == 1.)
                    else:
                        ref = next(r for r in old_report["confirmation"] if r["site"] == 3 and r["condition"] == f"selected_{mode}")
                    row = actual[mode]["eq4"]["confirmation"][str(float(ref["strength"]))]
                    archived_parity[mode] = {
                        metric: dict(archive=ref["metrics"][metric], cpu=row[metric],
                                     difference=row[metric] - ref["metrics"][metric])
                        for metric in ("occupancy", "action_change")}
                d["archived_eq4_comparison"] = archived_parity
            reference, reference_source = archive("results/belief_steering_reference.json")
            d.update(exact_filter_reference=reference["variants"]["2"], exact_filter_reference_source=reference_source)
    report = dict(A=a, B=b, C=c, D=d, provenance=dict(
        seed=seed, variant=variant, run_id=src["run_id"], result_path=str(path), archived_steering=old_source,
        checkpoint_final=ckpt, checkpoint_init=init_ckpt, rcond=AFFINE_RCOND, matmul_precision="highest",
        phase_seeds=PHASE_SEEDS, lag_grid=replay.LAGS, warmup=replay.WARMUP, fit_episodes=episodes, length=length,
        probability_temperature=1.5, high_drop_threshold=replay.DROP, jax=jax.__version__, numpy=np.__version__,
        replay_suffix=closed_loop.SUFFIX, model=recipe["model"], results_scope="smoke" if smoke else "main",
        source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob("*.py")},
        wall_seconds=time.monotonic() - start))
    write(output, report)
    print(f"DONE {variant=} {seed=} {time.monotonic()-start:.1f}s {output.name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=SEEDS)
    parser.add_argument("--variants", type=int, nargs="+", choices=(2, 3), default=(2, 3))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        parser.error("duplicate seed or variant")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    cores = sorted(os.sched_getaffinity(0))
    jobs = [(variant, seed) for seed in args.seeds for variant in args.variants]
    # Affinity follows each spawned process, not the task index.
    with ProcessPoolExecutor(max_workers=min(len(jobs), args.workers, len(cores)),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        list(pool.map(worker, ((variant, seed, cores[i % len(cores)], args.smoke)
                              for i, (variant, seed) in enumerate(jobs))))


if __name__ == "__main__":
    main()
