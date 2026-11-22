"""Same-host full-budget triangulation of saved and current B4 XOR runners."""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import platform

REPO = Path(__file__).resolve().parents[2]
RUNNERS = {
    "saved_encoder": "experiments.reward_factors_encoder_nonlinearity_2026_11.run",
    "current_encoder": "experiments.reward_factors_block2_steering_2026_11.encoder_run",
    "current_causal": "experiments.reward_factors_block2_steering_2026_11.run",
}


def digest(array):
    import numpy as np
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes()).hexdigest()


def worker(job):
    name, test, cpus = job
    os.sched_setaffinity(0, set(cpus))
    import numpy as np
    from experiments.algorithm_generality_probes_eps02_2026_11 import steer_moving_base as base
    from experiments.belief_quotient_encoder_nonlinearity_2026_11 import encoders
    runner = importlib.import_module(RUNNERS[name])
    runner.ROOT = REPO / "artifacts" / f"block2_audit_{name}"
    runner.ROOT.mkdir(parents=True, exist_ok=True)
    fits, encoder_fits = [], []
    original_affine, original_encoder = base.affine_fit, encoders.fit

    def affine(features, targets):
        result = original_affine(features, targets)
        if len(features) == 60000:
            fits.append(dict(shape=[len(features), features.shape[1], targets.shape[1]],
                             features=digest(np.asarray(features, np.float64)),
                             targets=digest(np.asarray(targets, np.float64)),
                             weight_float32=digest(np.asarray(result[0], np.float32)),
                             bias_float32=digest(np.asarray(result[1], np.float32))))
        return result

    def encoder(*args, **kwargs):
        result = original_encoder(*args, **kwargs)
        if result.kind == "affine":
            encoder_fits.append(dict(shape=list(result.params[0].shape),
                                     weight_float32=digest(np.asarray(result.params[0])),
                                     bias_float32=digest(np.asarray(result.params[1]))))
        return result

    base.affine_fit, encoders.fit = affine, encoder
    try:
        if name == "current_causal":
            for method in ("eq4", "eq4_joint"):
                runner.worker((test, 42, method, 3, cpus, False))
        else:
            runner.ARMS = (("affine", "exact"),)
            runner.worker((test, 42, 3, False) if name == "current_encoder" else (test, 42, False))
    finally:
        base.affine_fit, encoders.fit = original_affine, original_encoder
    records = []
    for path in sorted((runner.ROOT / "results").glob(f"*{test}*seed42*.json")):
        row = json.loads(path.read_text())
        records.append(dict(file=str(path.relative_to(REPO)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            representation=row.get("representation", "joint" if row.get("method") == "eq4_joint" else "marginal"),
                            checkpoint=row["checkpoint"]["sha256"], metrics=row["metrics"]))
    if len(records) != 2:
        raise ValueError(f"{name}/{test}: expected two full-budget controls, got {len(records)}")
    return dict(runner=name, test=test, fits=fits, encoder_fits=encoder_fits, records=records)


def main():
    os.environ.update(JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cpus = sorted(os.sched_getaffinity(0))[:6]
    jobs = [(name, test, cpus[i % 3::3]) for i, (name, test) in enumerate(
            (name, test) for name in RUNNERS for test in ("xor_eps02", "xor_v2"))]
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = list(pool.map(worker, jobs))
    result = dict(platform=platform.platform(), runners=RUNNERS, rows=rows,
                  audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    destination = Path(__file__).parent / "results" / "control_audit_seed42.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Same-host control audit complete: {sum(len(row['records']) for row in rows)} full-budget controls", flush=True)


if __name__ == "__main__":
    main()
