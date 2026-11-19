"""Run the three block-4 twin steering cohorts on local CPU."""

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
XOR_ROOT = ROOT.parent / "reward_factors_twin_steering_2026_11"
LADDER_REF = "45517b13a67d746048a002f9db52ea55a654b219"
XOR_REF = "3504bfb72455a411b725ef368a18ff892760555f"
XOR_ARCHIVE = "experiments/two_factor_coupled_moving_reward_steering_v2_jax_2026_11/results"
LADDER_STUDY = "belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11"
SEEDS = (42, 43, 44)
COHORTS = ("ladder", "v2_xor", "eps02_xor")


def archive(revision, path):
    raw = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=REPO)
    return json.loads(raw), dict(revision=revision, path=path, sha256=hashlib.sha256(raw).hexdigest())


def old_record(cohort, seed):
    from experiments.algorithm_generality_probes_eps02_2026_11.steer_core import MODES

    if cohort == "ladder":
        leaf = "variant_2_transformer"
        path = f"experiments/{LADDER_STUDY}/{leaf}/results/belief_steering_v2_seed{seed}_stable.json"
        record, source = archive(LADDER_REF, path)
        ckpt = record["checkpoint"]
        keep = ("occupancy", "occupancy_se", "action_change", "shift_norm", "relative_shift")
        metrics = lambda r: {k: r["metrics"][k] for k in keep}
        table = {(r["condition"], r["site"], r["strength"]): r for r in record["confirmation"]}
        comparisons = {}
        for mode in MODES:
            row, = [r for r in record["confirmation"] if r["site"] == 3 and r["condition"] == f"selected_{mode}"]
            comparisons[mode] = dict(selected_strength=row["strength"], selected=metrics(row),
                                     full_strength=metrics(table[f"curve_{mode}", 3, 1.]))
        result = dict(intact=metrics(table["intact", -1, 0.]), random=metrics(table["random", -1, 0.]),
                      comparisons=comparisons)
        expected = ckpt["sha256"]
        study, run_id = ckpt["study"], ckpt["run_id"]
    else:
        tag = "ctx32_2026_11" if cohort == "v2_xor" else "eps02_ctx32_2026_11"
        path = f"{XOR_ARCHIVE}/{tag}__reward_both__seed{seed}.json"
        record, source = archive(XOR_REF, path)
        old = record["rewards"]["xor"]["confirmation"]
        result = dict(controls={key: old[key] for key in ("intact", "random", "exact_belief_discounted_map")},
                      comparisons={target: old["comparisons"][f"discounted_{target}_block_4"]
                                   for target in ("marginal", "joint")},
                      matched_null={target: old["matched_null"][f"discounted_{target}"]
                                    for target in ("marginal", "joint")})
        expected = record["checkpoint_sha256"]
        study, leaf, run_id = (record["source"][k] for k in ("study", "leaf", "run_id"))
    return dict(result, cohort=cohort, seed=seed, study=study, leaf=leaf, run_id=run_id,
                expected_checkpoint_sha256=expected, archived_report=source)


def training_metadata(baseline, root, read, write):
    path = Path("experiments", baseline["study"], baseline["leaf"], "results", baseline["run_id"])
    if (REPO / path / "run_manifest.json").is_file():
        return path, read(REPO / path / "resolved_recipe.json"), read(REPO / path / "run_manifest.json")
    revision = baseline["archived_report"]["revision"]
    recipe, _ = archive(revision, path / "resolved_recipe.json")
    manifest, _ = archive(revision, path / "run_manifest.json")
    cached = root / "artifacts" / "training_metadata" / baseline["run_id"]
    write(cached / "run_manifest.json", manifest)
    return cached.relative_to(REPO), recipe, manifest


def worker(task):
    cohort, seed, smoke, cpu = task
    os.sched_setaffinity(0, set(cpu))

    import jax
    import numpy as np

    from experiments.algorithm_generality_probes_eps02_2026_11.analyze import FINAL, read, restore, write
    from experiments.belief_quotient_history_jax_2026_11.models import ModelSpec
    from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment as eps02
    from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import environment as v2
    from experiments.reward_factors_persistent_nuisance_jax_2026_11.shared import factor_process
    from experiments.reward_factors_twin_steering_2026_11 import xor

    from . import ladder, twins

    start = time.monotonic()
    source_sha256 = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted([*ROOT.glob("*.py"), *XOR_ROOT.glob("*.py")])}
    root = ROOT if cohort == "ladder" else XOR_ROOT
    output = root / ("artifacts/smoke" if smoke else "results") / f"{cohort}_seed{seed}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    baseline = old_record(cohort, seed)
    source, recipe, manifest = training_metadata(baseline, root, read, write)
    assert manifest["status"] == "completed" and manifest["runtime"]["seed"] == seed
    spec = ModelSpec(**recipe["model"])
    environment = v2 if cohort == "v2_xor" else eps02
    process = environment.ladder_process(2) if cohort == "ladder" else factor_process("reward_both", environment)
    for field, value in process._asdict().items():
        np.testing.assert_allclose(value, recipe["process"][field], atol=1e-6)
    with jax.default_matmul_precision("highest"):
        params, checkpoint = restore(dict(baseline, results_path=str(source), algorithm="reinforce"), FINAL, spec)
        if checkpoint["sha256"] != baseline["expected_checkpoint_sha256"]:
            raise ValueError("checkpoint digest differs from archived result")
        if cohort == "ladder":
            report = ladder.run(spec, process, params, baseline, smoke)
        else:
            report = xor.run(spec, process, params, baseline, environment.emission_matrix(), smoke)
    report["provenance"] = dict(
        checkpoint=checkpoint, paper_block=twins.SITE + 1, residual_projection=False, rcond=twins.RCOND,
        jax=jax.__version__, numpy=np.__version__, device=str(jax.devices()[0]), cpus=sorted(cpu),
        fit_seed=twins.FIT_SEED, heldout_seed=twins.TEST_SEED,
        tune_seed=twins.TUNE_SEED, confirmation_seed=twins.CONFIRM_SEED, smoke=smoke,
        regime_length=twins.REGIME_LENGTH, wall_seconds=time.monotonic() - start,
        source_sha256=source_sha256)
    if not smoke:
        controls = (report, baseline) if cohort == "ladder" else (report["controls"], baseline["controls"])
        key = "occupancy" if cohort == "ladder" else "total_reward"
        for condition in ("intact", "random"):
            np.testing.assert_allclose(controls[0][condition][key], controls[1][condition][key], rtol=0, atol=1e-7)
        if cohort != "ladder":
            np.testing.assert_allclose(controls[0]["map"][key],
                                       controls[1]["exact_belief_discounted_map"][key], rtol=0, atol=1e-7)
    write(output, report)
    print(f'DONE {cohort} seed {seed}: {report["provenance"]["wall_seconds"]:.1f}s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS, nargs="+", default=COHORTS)
    parser.add_argument("--seeds", choices=SEEDS, type=int, nargs="+", default=SEEDS)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cores = sorted(os.sched_getaffinity(0))
    jobs = [(c, s) for c in args.cohort for s in args.seeds]
    workers = min(args.workers, len(jobs))
    if not 1 <= workers <= len(cores):
        parser.error("number of workers must fit CPU affinity")
    os.environ.update(JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                             max_tasks_per_child=1) as pool:
        for offset in range(0, len(jobs), workers):
            batch = [(*job, args.smoke, cores[i::workers])
                     for i, job in enumerate(jobs[offset:offset + workers])]
            list(pool.map(worker, batch))


if __name__ == "__main__":
    main()
