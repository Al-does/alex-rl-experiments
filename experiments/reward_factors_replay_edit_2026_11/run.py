"""CPU entry point for the three-seed moving-reward and XOR replay study."""

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
REF = "3504bfb72455a411b725ef368a18ff892760555f"
ARCHIVE = "experiments/two_factor_coupled_moving_reward_steering_v2_jax_2026_11/results"
STUDY = "reward_factors_persistent_nuisance_{}_jax_temp15_ctx32_2026_11"
SEEDS = (42, 43, 44)
COHORTS = ("eps02", "v2", "marginal")
SPLIT = dict(fit=42, selection=43, test=44)


def archive(path):
    data = subprocess.check_output(["git", "show", f"{REF}:{path}"], cwd=REPO)
    return json.loads(data), dict(revision=REF, path=str(path), sha256=hashlib.sha256(data).hexdigest())


def write_result(path, report):
    def encode(value):
        return json.dumps(value, allow_nan=False, separators=(",", ":"))

    sections = []
    for name, records in report.items():
        rows = ",\n".join(f"    {encode(key)}: {encode(value)}" for key, value in records.items())
        sections.append(f"  {encode(name)}: {{\n{rows}\n  }}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text("{\n" + ",\n".join(sections) + "\n}\n")
    temp.replace(path)


def source(cohort, seed):
    study = STUDY.format("v2" if cohort == "v2" else "eps02")
    condition = "reward_factor_1" if cohort == "marginal" else "reward_both"
    parent = Path("experiments", study, condition + "_transformer", "results")
    if cohort == "marginal":
        matches = [p.parent for p in (REPO / parent).glob("*/run_manifest.json")
                   if json.loads(p.read_text())["runtime"]["seed"] == seed]
        if len(matches) != 1:
            raise ValueError(matches)
        path, archived, provenance = matches[0].relative_to(REPO), None, None
    else:
        tag = "ctx32_2026_11" if cohort == "v2" else "eps02_ctx32_2026_11"
        archived, provenance = archive(f"{ARCHIVE}/{tag}__reward_both__seed{seed}.json")
        path = parent / archived["source"]["run_id"]
    return path, condition, archived, provenance


def worker(task):
    cohort, seed, cpu, smoke = task
    os.sched_setaffinity(0, set(cpu))
    start = time.monotonic()

    import jax
    import numpy as np

    from experiments.algorithm_generality_probes_eps02_2026_11.analyze import FINAL, read, restore, write
    from experiments.algorithm_generality_probes_eps02_2026_11 import steer_moving_base as base
    from experiments.belief_quotient_history_jax_2026_11.models import ModelSpec
    from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment as eps02
    from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import environment as v2
    from experiments.reward_factors_persistent_nuisance_jax_2026_11.shared import factor_process
    from . import replay, transfer

    output = ROOT / ("artifacts/pilot" if smoke else "results") / f"{cohort}_seed{seed}.json"
    if output.exists():
        raise FileExistsError(output)
    path, condition, archived, provenance = source(cohort, seed)
    if (REPO / path / "resolved_recipe.json").exists():
        recipe = read(REPO / path / "resolved_recipe.json")
        manifest = read(REPO / path / "run_manifest.json")
        locator = path
    else:
        recipe, _ = archive(path / "resolved_recipe.json")
        manifest, _ = archive(path / "run_manifest.json")
        locator = (ROOT / "artifacts" / "metadata" / path.name).relative_to(REPO)
        write(REPO / locator / "run_manifest.json", manifest)
    if (manifest["status"], manifest["runtime"]["seed"], recipe["training"]["temperature"]) != ("completed", seed, 1.5):
        raise ValueError(path)
    spec = ModelSpec(**recipe["model"])
    env = v2 if cohort == "v2" else eps02
    process = factor_process(condition, env)
    for k, v in process._asdict().items():
        np.testing.assert_allclose(v, recipe["process"][k], atol=1e-6)

    src = dict(results_path=str(locator), run_id=path.name, algorithm="reinforce")
    length, nt, nf = (256, 2, 2) if smoke else (base.HORIZON, 12, 24)
    with jax.default_matmul_precision("highest"):
        params, checkpoint = restore(src, FINAL, spec)
        if archived and checkpoint["sha256"] != archived["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash differs from archived transfer: {path}")
        def samples(p, split, count):
            return replay.collect(spec, p, process, SPLIT[split], count, length)
        held = samples(params, "test", nt)
        report = {}
        if cohort == "marginal":
            init, init_checkpoint = restore(src, 0, spec)
            report["E"] = {
                "final": replay.replay_cases(spec, params, env, held, 7002, factor_ids=(1,), part_e=True)[0],
                "initialization": replay.replay_cases(spec, init, env, samples(init, "test", nt), 7002,
                                                     factor_ids=(1,), part_e=True)[0],
                "initial_checkpoint": init_checkpoint}
        else:
            a, b, pairs = replay.replay_cases(spec, params, env, held, 7002, patches=True, max_episodes=nt)
            def paired(split, count, offset):
                return replay.replay_cases(spec, params, env, samples(params, split, count), 7000 + offset,
                                           max_episodes=count)[2]
            c, maps = replay.fit(paired("fit", nf, 0), paired("selection", nt, 1), pairs)
            report.update(A=a, B=b, C=c)
            print(f"{cohort} seed {seed}: offline complete ({time.monotonic()-start:.0f}s)", flush=True)
            rows = base.collect_rows(spec, params, process, rows=256 if smoke else archived["fit_rows"],
                                     seed=archived["fit_seed"])
            report["D"] = {}
            for name in (("moving", "xor") if cohort == "eps02" else ("xor",)):
                report["D"][name] = transfer.compare(spec, params, process, rows, maps, archived, name, smoke)
                print(f"{cohort} seed {seed}: {name} complete ({time.monotonic()-start:.0f}s)", flush=True)
    report["provenance"] = dict(cohort=cohort, seed=seed, checkpoint=checkpoint, run_id=path.name,
                                archive=provenance, splits=SPLIT, replay_fit_episodes=nf,
                                replay_test_episodes=nt, replay_length=length, warmup=replay.WARMUP,
                                precision="highest", rcond=replay.RCOND, jax=jax.__version__,
                                scope="pilot" if smoke else "main")
    if archived and not smoke:
        fields = ("fit_rows", "fit_seed", "tune_seed", "confirm_seed", "tune_episodes",
                  "confirm_episodes", "episode_horizon")
        report["provenance"]["archived_comparison_budgets"] = {k: archived[k] for k in fields}
    write_result(output, report)
    print(f"DONE {output.name} ({time.monotonic()-start:.0f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, nargs="+", default=SEEDS)
    parser.add_argument("--cohort", choices=COHORTS, nargs="+", default=COHORTS)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", JAX_PLATFORMS="cpu")
    cpus = sorted(os.sched_getaffinity(0))
    count = min(args.workers, len(cpus), len(args.seed) * len(args.cohort))
    jobs = [(c, s) for c in args.cohort for s in args.seed]
    # Assign one CPU group per static lane, not per future, so jobs cannot collide.
    lanes = [[(c, s, cpus[i::count], args.pilot) for c, s in jobs[i::count]] for i in range(count)]
    with ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context("spawn")) as pool:
        list(pool.map(lane, lanes))


def lane(tasks):
    for task in tasks:
        worker(task)


if __name__ == "__main__":
    main()
