"""Hash-verified PR202 greedy histories and archived-probe reproduction."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from analysis.probes.linear import fit_affine_probe, probe_predict, r2_score, global_mse_metrics
from harness.storage.b2 import B2StorageConfig
from experiments.belief_quotient_history_jax_2026_11.continuation import load_params
from experiments.belief_quotient_history_jax_2026_11.env import make_process
from experiments.belief_quotient_history_jax_2026_11.models import ModelSpec
from experiments.belief_quotient_history_jax_2026_11.probe import _probe_rollout
from .metrics import bootstrap, score

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "belief_quotient_history_jax_lumped_temp15_2026_11"
RAW = ROOT / "artifacts" / "temp15"
TOLERANCE = 1e-2
TOLERANCE_REASON = "Historical discrepancy threshold max(0.01, 5 * CPU episode-bootstrap SE), now annotations only by user authorization. No archived-parity hard stops. Primary matched comparisons use verified checkpoints and identical CPU histories."


def runs():
    return sorted(SOURCE.glob("variant_*/results/*-seed42-temp15-4000m"))


def gate_passes(check):
    deltas = np.asarray([check["delta"], check["ratio_delta"]])
    threshold = max(TOLERANCE, 5 * check.get("bootstrap_se", 0.0))
    return bool(np.isfinite(deltas).all() and (np.abs(deltas) <= threshold).all())


def restore(run, step, *, raw_root=None):
    recipe = json.loads((run / "resolved_recipe.json").read_text())
    manifest = json.loads((run / "run_manifest.json").read_text())
    remote = manifest["remote_artifacts"]
    directory = (RAW if raw_root is None else raw_root) / run.parent.parent.name
    directory.mkdir(parents=True, exist_ok=True)
    storage = B2StorageConfig.from_env()
    client = storage.s3_client()
    path = directory / "durability_manifest.json"
    if not path.exists():
        client.download_file(remote["bucket"], remote["canonical_manifest_key"], str(path))
    durability = json.loads(path.read_text())
    name = f"step_{step:010d}.npz"
    entries = [f for f in durability["files"] if f["relative_path"] == f"checkpoints/{name}"]
    assert len(entries) == 1, (name, len(entries))
    entry = entries[0]
    path = directory / name
    if not path.exists():
        client.download_file(remote["bucket"], entry["key"], str(path))
    assert path.stat().st_size == entry["size_bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    spec = ModelSpec(**recipe["model"])
    return spec, make_process(**recipe["process"]), load_params(path, spec), {
        "run_id": manifest["run_id"], "source_git": manifest["git"],
        "manifest_key": remote["canonical_manifest_key"], "checkpoint": entry,
        "recipe_sha256": hashlib.sha256((run / "resolved_recipe.json").read_bytes()).hexdigest(),
        "durability_sha256": hashlib.sha256((directory / "durability_manifest.json").read_bytes()).hexdigest()}


def rows(records, name, count):
    array = np.asarray(records[name])[:, 64:]
    return array.reshape(-1, *array.shape[2:])[:count]


def histories(run, step, spec, process, params, *, raw_root=None):
    result = []
    keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(42), 2))
    for split, count, key in zip(("fit", "test"), (60000, 80000), keys):
        path = (RAW if raw_root is None else raw_root) / run.parent.parent.name / f"{step}_{split}.npz"
        if not path.exists():
            records = _probe_rollout(spec, process, params, key, -(-count // 960), 1024)
            np.savez_compressed(path, **{k: np.asarray(records[k]) for k in ("obs", "action", "embedding", "belief")})
        with np.load(path) as saved:
            result.append({k: saved[k] for k in saved.files})
    return result


def diagnosis():
    run = runs()[0]
    spec, process, params, provenance = restore(run, 0)
    point = json.loads((run / "condition_summary.json").read_text())["checkpoint_reports"][0]
    result = {"arm": run.parent.parent.name, "archived": point, "provenance": provenance, "cpu": {}}
    for precision in ("default", "highest"):
        jax.clear_caches()
        with jax.default_matmul_precision(precision):
            keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(42), 2))
            data = [_probe_rollout(spec, process, params, key, -(-count // 960), 1024)
                    for count, key in zip((60000, 80000), keys)]
            w, bias = fit_affine_probe(rows(data[0], "embedding", 60000), rows(data[0], "belief", 60000), ridge=1e-6)
            predicted = probe_predict(w, bias, rows(data[1], "embedding", 80000))
            result["cpu"][precision] = {
                "r2": r2_score(predicted, rows(data[1], "belief", 80000)),
                "action_fractions": (np.bincount(rows(data[1], "action", 80000), minlength=3)/80000).tolist(),
                "reward_per_step": float(rows(data[1], "reward", 80000).mean())}
    (ROOT / "results" / "ladder_temp15_precision_diagnosis.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result["cpu"], indent=2), flush=True)


def final_diagnosis():
    run = runs()[0]
    point = json.loads((run / "condition_summary.json").read_text())["checkpoint_reports"][-1]
    step = point["checkpoint_step"]
    spec, process, params, provenance = restore(run, step)
    train, test = histories(run, step, spec, process, params)
    result = {"step": step, "provenance": provenance, "archived_r2": point["targets"]["belief"]["r_squared"]}
    for dtype in (np.float64, np.float32):
        x, y = rows(train, "embedding", 60000).astype(dtype), rows(train, "belief", 60000).astype(dtype)
        xm, ym = x.mean(0), y.mean(0)
        centered = x-xm
        design = np.concatenate([centered, np.sqrt(dtype(1e-6))*np.eye(x.shape[1], dtype=dtype)])
        response = np.concatenate([y-ym, np.zeros((x.shape[1], 3), dtype=dtype)])
        weight = np.linalg.lstsq(design, response, rcond=None)[0]
        pred = (rows(test, "embedding", 80000).astype(dtype)-xm) @ weight+ym
        result[np.dtype(dtype).name] = {"r2": score(rows(test, "belief", 80000), pred)["r2"],
            "centered_design_condition": float(np.linalg.cond(centered.astype(float))),
            "ridge_augmented_condition": float(np.linalg.cond(design.astype(float)))}
        if dtype == np.float64:
            result["global_bootstrap"] = bootstrap(rows(test, "belief", 80000), pred,
                np.zeros(80000, dtype=int), np.arange(80000)//960, n=10000)
    result["history_device_independence"] = "Identical fixed actions remove policy branching, but categorical sampling uses floating-point log/Gumbel comparisons. Threefry keys/bits are intended to be portable; JAX/XLA floating-point sampling and environment execution are not guaranteed bitwise identical across CPU/GPU. No archived raw histories are available here to verify identity."
    (ROOT / "results" / "ladder_temp15_final_diagnosis.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


def gate():
    checks = []
    assert len(runs()) == 12
    for run in runs():
        archived = json.loads((run / "condition_summary.json").read_text())["checkpoint_reports"]
        for point in (archived[0], archived[-1]):
            step = point["checkpoint_step"]
            spec, process, params, provenance = restore(run, step)
            train, test = histories(run, step, spec, process, params)
            x, y = rows(train, "embedding", 60000), rows(train, "belief", 60000)
            w, bias = fit_affine_probe(x, y, ridge=1e-6)
            actual = r2_score(probe_predict(w, bias, rows(test, "embedding", 80000)), rows(test, "belief", 80000))
            expected = point["targets"]["belief"]["r_squared"]
            ratio = global_mse_metrics(probe_predict(w, bias, rows(test, "embedding", 80000)), rows(test, "belief", 80000))["global_mse_ratio"]
            expected_ratio = point["targets"]["belief"]["global_mse_ratio"]
            uncertainty = bootstrap(rows(test, "belief", 80000), probe_predict(w, bias, rows(test, "embedding", 80000)),
                np.zeros(80000, dtype=int), np.arange(80000)//960, n=2000)
            check = {"arm": run.parent.parent.name, "step": step, "archived": expected,
                     "reproduced": actual, "delta": actual-expected, "provenance": provenance,
                     "archived_1_minus_r2": expected_ratio, "reproduced_1_minus_r2": ratio,
                     "ratio_delta": ratio-expected_ratio,
                     "bootstrap_se": uncertainty["r2_se"], "bootstrap": uncertainty,
                     "hard_stop_threshold": max(TOLERANCE, 5*uncertainty["r2_se"]),
                     "archived_policy": point["policy"],
                     "cpu_action_fractions": (np.bincount(rows(test, "action", 80000), minlength=3)/80000).tolist()}
            checks.append(check)
            check["annotation"] = None if gate_passes(check) else "archived global R² not reproduced on CPU"
            (ROOT / "results" / "ladder_temp15_gate.json").write_text(json.dumps({"tolerance": TOLERANCE, "tolerance_reason": TOLERANCE_REASON, "checks": checks}, indent=2)+"\n")
            print(f"GATE {check['arm']} step={step} archived={expected:.12f} reproduced={actual:.12f} delta={actual-expected:+.3e}", flush=True)
        jax.clear_caches()
    print("All 24 checks recorded; discrepancies are annotations, not blockers.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["gate", "diagnosis", "final_diagnosis"])
    args = parser.parse_args()
    {"diagnosis": diagnosis, "final_diagnosis": final_diagnosis, "gate": gate}[args.mode]()
