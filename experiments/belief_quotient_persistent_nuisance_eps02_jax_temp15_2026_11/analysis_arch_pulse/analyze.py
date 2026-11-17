"""Seed-42 pulse comparison; no training, uploads, or paid infrastructure."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .reference import REVISION, SOURCE, STUDY, core, pulse
from . import recurrent

import jax
import numpy as np

from harness.storage.b2 import B2StorageConfig
from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_history_jax_2026_11.continuation import load_params
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment

HERE = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
RESULTS_REVISION = "d20d13dc172b4959477c4ea12f919520209144f2"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def frozen_json(revision, path):
    return json.loads(git("show", f"{revision}:{path}"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore(architecture, variant):
    leaf = f"experiments/{STUDY.name}/variant_{variant}_{architecture}"
    paths = git("ls-tree", "-r", "--name-only", RESULTS_REVISION, "--", leaf + "/results").splitlines()
    recipes = [(p, frozen_json(RESULTS_REVISION, p)) for p in paths if p.endswith("/resolved_recipe.json")]
    (path, recipe), = [(p, r) for p, r in recipes if r["seed"] == 42 and not r["smoke"]]
    directory = str(Path(path).parent)
    manifest = frozen_json(RESULTS_REVISION, directory + "/run_manifest.json")
    assert manifest["status"] == "completed" and recipe["continued_from"] is None
    remote = manifest["remote_artifacts"]
    client = B2StorageConfig.from_env().s3_client()
    raw = client.get_object(Bucket=remote["bucket"], Key=remote["canonical_manifest_key"])["Body"].read()
    relative = "checkpoints/step_2012217344.npz"
    row, = [r for r in json.loads(raw)["files"] if r["relative_path"] == relative]
    checkpoint = STUDY / "artifacts" / "arch_coarsening" / manifest["run_id"] / Path(relative).name
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        client.download_file(remote["bucket"], row["key"], str(checkpoint))
    if checkpoint.stat().st_size != row.get("size_bytes", row.get("size")) or sha(checkpoint) != row["sha256"]:
        raise ValueError("checkpoint durability verification failed")
    provenance = {"run_id": manifest["run_id"], "results_path": directory,
                  "step": 2012217344, "sha256": sha(checkpoint), "bytes": checkpoint.stat().st_size,
                  "uri": f"s3://{remote['bucket']}/{row['key']}",
                  "manifest_key": remote["canonical_manifest_key"],
                  "manifest_sha256": hashlib.sha256(raw).hexdigest(), "training_git": manifest["git"]}
    return recipe, checkpoint, provenance


def transformer_reference(variant, measured):
    compact_path = SOURCE / "experiments" / STUDY.name / f"variant_{variant}_transformer" / "results" / "belief_steering_v2_seed42_stable.json"
    compact = json.loads(compact_path.read_text())
    reference = compact["full_report"]
    local = STUDY / "artifacts" / "arch_pulse" / f"transformer_v{variant}_original.json"
    if not local.exists():
        client = B2StorageConfig.from_env().s3_client()
        client.download_file(reference["bucket"], reference["key"], str(local))
    if sha(local) != reference["sha256"] or local.stat().st_size != reference["bytes"]:
        raise ValueError("original transformer report hash/size differs")
    original = json.loads(local.read_text())
    comparisons = []
    for name, item in measured["conditions"].items():
        for kind in ("measured", "full_filter"):
            old = np.asarray(original["pulse"]["conditions"][name][kind]["absolute"])
            new = np.asarray(item[kind]["absolute"])
            comparisons.append({"condition": name, "kind": kind,
                "max_abs_difference": float(np.abs(new-old).max()),
                "max_abs_difference_lags_0_1": float(np.abs(new[:2]-old[:2]).max())})
    return {"full_report_uri": reference["uri"], "full_report_sha256": reference["sha256"],
            "compact_report_sha256": sha(compact_path), "comparisons": comparisons,
            "max_abs_difference": max(r["max_abs_difference"] for r in comparisons),
            "original_probes": compact["probes"]["exclusion_8"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("transformer", "gru128", "gru92"), required=True)
    parser.add_argument("--variant", choices=(2, 3), type=int, required=True)
    args = parser.parse_args()
    output = HERE / "results" / f"variant_{args.variant}_{args.architecture}_seed42.json"
    if output.exists():
        raise FileExistsError(output)
    start = time.monotonic()
    recipe, checkpoint, provenance = restore(args.architecture, args.variant)
    (STUDY / "artifacts" / "arch_pulse").mkdir(parents=True, exist_ok=True)
    process = environment.ladder_process(args.variant)
    for key in ("initial", "transitions", "emission"):
        np.testing.assert_array_equal(np.asarray(getattr(process, key)), np.asarray(recipe["process"][key], dtype=np.float32))
    spec = models.ModelSpec(**recipe["model"])
    params = load_params(checkpoint, spec)
    if spec.kind == "transformer":
        maps = core.empty_maps()
        fit = core.rollout(spec, process, params, maps, jax.random.PRNGKey(42), 14, 1024, capture=True)
        maps = core.fit_maps(fit, 12000)
        test = core.rollout(spec, process, params, maps, jax.random.PRNGKey(43), 14, 1024, capture=True)
        probes = core.probe_scores(test, maps, 12000, 8)
        records = core.rollout(spec, process, params, maps, jax.random.PRNGKey(3042), 32, 1024, capture=True)
        result = pulse.run_pulse(spec, params, maps, records, args.variant, environment=environment)
        sanity = transformer_reference(args.variant, result)
    else:
        maps = recurrent.empty_maps(spec)
        fit = recurrent.rollout(spec, process, params, maps, jax.random.PRNGKey(42), 14, 1024)
        maps = recurrent.fit_maps(fit, 12000)
        test = recurrent.rollout(spec, process, params, maps, jax.random.PRNGKey(43), 14, 1024)
        probes = recurrent.probe_scores(test, maps, 12000)
        records = recurrent.rollout(spec, process, params, maps, jax.random.PRNGKey(3042), 32, 1024)
        result = recurrent.run_pulse(spec, params, maps, records, args.variant, environment)
        sanity = None
    report = {"architecture": args.architecture, "variant": args.variant, "seed": 42,
              "checkpoint": provenance, "reference_driver_revision": REVISION,
              "results_revision": RESULTS_REVISION, "analysis_base_revision": git("rev-parse", "HEAD"),
              "origin_main_checked_revision": "4b575146ec27c69d12b09c6fcea9eb00ede5c116",
              "main_reference_status": "steering driver and pulse modules absent on fetched origin/main",
              "source_sha256": {name: sha(HERE / name) for name in ("analyze.py", "reference.py", "recurrent.py")},
              "reference_source_sha256": {Path(p.__file__).name: sha(p.__file__) for p in (core, pulse)},
              "runtime": {"jax": jax.__version__, "numpy": np.__version__, "devices": [str(x) for x in jax.devices()]},
              "protocol": {"fit_seed": 42, "test_seed": 43, "rows_each": 12000, "warmup": 8,
                  "fit_test_episodes": 14, "episode_length": 1024, "affine_rcond": 1e-6,
                  "write": "h + (target(decoded_b)[:2] - decoded_b[:2]) @ encoder; strength 1 once",
                  "encoder": "affine reverse regression b[:2] -> h, not decoder pseudoinverse",
                  "readout": "block4 pre-final-norm" if spec.kind == "transformer" else "top recurrent hidden pre-final-norm",
                  "identical_future_inputs": "saved unsteered observation and previous executed action for both branches",
                  "coordinate_order": ["rho", "delta"], "policy": "greedy"},
              "probes": probes, "pulse": result, "transformer_sanity": sanity,
              "elapsed_seconds": time.monotonic() - start}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {output}; elapsed {report['elapsed_seconds']:.1f}s", flush=True)
    if sanity:
        print("Transformer sanity maximum absolute curve difference:", sanity["max_abs_difference"], flush=True)


if __name__ == "__main__":
    main()
