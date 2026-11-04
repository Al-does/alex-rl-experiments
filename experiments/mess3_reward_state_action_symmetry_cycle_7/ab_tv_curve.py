"""Evaluate A/B sensitivity at initialization and every saved seed-42 checkpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import inspect
import io
import json
import multiprocessing
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from analysis.checkpoints import load_module_only
from experiments.mess3_reward_state_action_symmetry_cycle_7.ab_on_policy_tv import (
    BOOTSTRAP_SEED,
    EVALUATION_SEED,
    WARMUP,
    collect,
    summary,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.ab_pattern_checkpoint import (
    sha256,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.control_campaign import (
    MODULE_FILES,
    WorkItem,
    _module_records,
    _s3_client,
)


STUDY = Path(__file__).resolve().parent
ARM = STUDY / "variant_2_ctx32_ent"
RUN = ARM / "results/mess3_reward_state_action_symmetry_cycle_6-variant_2_ctx32_ent-seed42-10m"


def checkpoint_entries(manifest: dict, progress: list[dict]) -> list[dict]:
    names = {
        Path(record["relative_path"]).parts[-6]
        for record in manifest["files"]
        if record["relative_path"].endswith("/default_policy/module_state.pkl")
    }
    expected = {"initial_checkpoint"} | {f"checkpoint_{i:06d}" for i in range(26)}
    if names != expected:
        raise ValueError(f"expected initialization plus checkpoints 0-25; got {sorted(names)}")
    by_iteration = {int(row["training_iteration"]): row for row in progress}
    entries = []
    for name in sorted(names, key=lambda value: -1 if value == "initial_checkpoint" else int(value[-6:])):
        iteration = 0 if name == "initial_checkpoint" else int(name[-6:]) + 1
        steps = 0 if iteration == 0 else int(float(
            by_iteration[iteration]["env_runners/num_env_steps_sampled_lifetime"]
        ))
        entries.append({"name": name, "training_iteration": iteration, "agent_steps": steps})
    if any(a["agent_steps"] >= b["agent_steps"] for a, b in zip(entries, entries[1:])):
        raise ValueError("checkpoint training-step counts are not strictly increasing")
    return entries


def prepare(scratch: Path) -> Path:
    index_path = scratch / "checkpoint_index.json"
    if index_path.exists():
        raise FileExistsError(f"refusing to overwrite {index_path}")
    run = json.loads((RUN / "run_manifest.json").read_text())
    remote = run["remote_artifacts"]
    client = _s3_client()
    body = client.get_object(
        Bucket=remote["bucket"], Key=remote["canonical_manifest_key"],
    )["Body"].read()
    manifest = json.loads(body)
    progress_record = next(
        row for row in manifest["files"] if row["relative_path"].endswith("/progress.csv")
    )
    progress_body = client.get_object(
        Bucket=remote["bucket"], Key=progress_record["key"],
    )["Body"].read()
    if hashlib.sha256(progress_body).hexdigest() != progress_record["sha256"]:
        raise ValueError("progress.csv hash mismatch")
    entries = checkpoint_entries(manifest, list(csv.DictReader(io.StringIO(progress_body.decode()))))
    existing = {
        "checkpoint_000023": ARM / "artifacts/ab_checkpoint23_tv_seed42/module",
        "checkpoint_000024": ARM / "artifacts/ab_pattern_checkpoint_seed42/previous_module",
        "checkpoint_000025": ARM / "artifacts/ab_logit_evidence_seed42/module",
    }
    for entry in entries:
        name = entry["name"]
        path = existing.get(name, scratch / "modules" / name)
        item = WorkItem(
            "variant_2_ctx32_ent", 2, 42, name, entry["training_iteration"],
            entry["training_iteration"], entry["agent_steps"], RUN.name,
            None if name == "initial_checkpoint" else name,
            RUN / "checkpoint_probe_curve.json", RUN / "run_manifest.json",
            path, scratch / f"{name}.json",
        )
        root, records = _module_records(manifest, item)
        path.mkdir(parents=True, exist_ok=True)
        for filename in MODULE_FILES:
            target = path / filename
            expected = records[filename]["sha256"]
            if target.exists():
                if sha256(target) != expected:
                    raise ValueError(f"existing module file has wrong hash: {target}")
            else:
                client.download_file(remote["bucket"], records[filename]["key"], str(target))
                if sha256(target) != expected:
                    raise ValueError(f"download hash mismatch: {target}")
        entry.update(
            module_path=str(path), checkpoint_relative_path=root,
            module_files={filename: records[filename]["sha256"] for filename in MODULE_FILES},
        )
        print(f"Verified {name}: {entry['agent_steps']} steps", flush=True)
    index = {
        "run_id": run["run_id"], "training_revisions": run["git"],
        "manifest_key": remote["canonical_manifest_key"],
        "manifest_sha256": hashlib.sha256(body).hexdigest(),
        "progress_sha256": hashlib.sha256(progress_body).hexdigest(),
        "checkpoints": entries,
    }
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    return index_path


def evaluation_spec(smoke: bool) -> dict:
    return {
        "evaluation_seed": EVALUATION_SEED,
        "episodes": 4 if smoke else 64, "horizon": 256 if smoke else 1024,
        "warmup": WARMUP, "resamples": 100 if smoke else 2000,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "collector_sha256": sha256(Path(inspect.getfile(inspect.unwrap(collect)))),
        "driver_sha256": sha256(Path(__file__)),
    }


def evaluate_checkpoint(entry: dict, scratch: str, spec: dict, reuse_report: str | None) -> dict:
    torch.set_num_threads(2)
    name = entry["name"]
    point_dir = Path(scratch) / "points"
    point_dir.mkdir(parents=True, exist_ok=True)
    point_path = point_dir / f"{name}.json"
    if point_path.exists():
        cached = json.loads(point_path.read_text())
        if cached["spec"] != spec or cached["checkpoint"] != entry:
            raise ValueError(f"cached point has different protocol or provenance: {point_path}")
        for record in cached["raw"].values():
            if sha256(Path(record["path"])) != record["sha256"]:
                raise ValueError(f"cached raw-data hash mismatch: {record['path']}")
        return cached
    module_path = Path(entry["module_path"])
    for filename, digest in entry["module_files"].items():
        if sha256(module_path / filename) != digest:
            raise ValueError(f"module hash mismatch: {module_path / filename}")
    reuse = None
    if reuse_report and name in {f"checkpoint_{i:06d}" for i in (23, 24, 25)}:
        reuse = json.loads(Path(reuse_report).read_text())
        if (
            reuse["provenance"]["source_sha256"] != spec["collector_sha256"]
            or reuse["protocol"]["evaluation_seed"] != spec["evaluation_seed"]
            or reuse["protocol"]["episodes_per_checkpoint_and_mode"] != spec["episodes"]
            or reuse["protocol"]["horizon"] != spec["horizon"]
            or reuse["protocol"]["warmup"] != spec["warmup"]
        ):
            raise ValueError("existing three-checkpoint evaluation does not match the curve protocol")
        if reuse["provenance"]["checkpoints"][name]["module_state_sha256"] != entry["module_files"]["module_state.pkl"]:
            raise ValueError("reused checkpoint has different weights")
    module = None if reuse else load_module_only(module_path)
    if module is not None and module.sequence_lookback != spec["warmup"]:
        raise ValueError("unexpected checkpoint receptive field")
    results, raw = {}, {}
    for mode in ("greedy", "sampled"):
        if reuse:
            record = reuse["provenance"]["raw"][f"{mode}_{name}"]
            raw_path = Path(record["path"])
            if sha256(raw_path) != record["sha256"]:
                raise ValueError(f"reused raw-data hash mismatch: {raw_path}")
            data = dict(np.load(raw_path))
        else:
            raw_path = Path(scratch) / f"{mode}_{name}.npz"
            if raw_path.exists():
                raise FileExistsError(f"incomplete point already has raw data: {raw_path}")
            print(f"Evaluate {name}: {mode}", flush=True)
            data = collect(
                module, mode=mode, episodes=spec["episodes"],
                horizon=spec["horizon"], seed=spec["evaluation_seed"],
            )
            np.savez_compressed(raw_path, **data)
        results[mode] = summary(data, spec["resamples"])
        if reuse:
            np.testing.assert_allclose(
                results[mode]["total_variation"]["mean"],
                reuse["results"][mode][name]["total_variation"]["mean"], atol=1e-15, rtol=0,
            )
        raw[mode] = {"path": str(raw_path), "sha256": sha256(raw_path)}
    point = {
        "checkpoint": entry, "spec": spec, "results": results,
        "raw": raw, "reused": reuse is not None,
    }
    point_path.write_text(json.dumps(point, indent=2, allow_nan=False) + "\n")
    print(f"Completed {name}: " + ", ".join(
        f"{mode} TV={results[mode]['total_variation']['mean']:.6g}"
        for mode in results
    ), flush=True)
    return point


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--scratch", type=Path, required=True)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--index", type=Path, required=True)
    evaluate_parser.add_argument("--scratch", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--workers", type=int, default=3)
    evaluate_parser.add_argument(
        "--reuse-report", type=Path,
        default=ARM / "results/ab_on_policy_tv_seed42_20261105.json",
    )
    evaluate_parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.scratch = args.scratch.resolve()
    args.scratch.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        print(prepare(args.scratch))
        return
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if args.workers < 1:
        parser.error("workers must be positive")
    index = json.loads(args.index.read_text())
    entries = index["checkpoints"]
    if len(entries) != 27:
        raise ValueError("a complete curve requires initialization plus all 26 checkpoints")
    spec = evaluation_spec(args.smoke)
    reuse = None if args.smoke else str(args.reuse_report.resolve())
    points = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = [
            executor.submit(evaluate_checkpoint, entry, str(args.scratch), spec, reuse)
            for entry in entries
        ]
        for future in concurrent.futures.as_completed(futures):
            points.append(future.result())
            print(f"Curve progress: {len(points)}/{len(entries)} checkpoints", flush=True)
    points.sort(key=lambda point: point["checkpoint"]["agent_steps"])
    library = Path(inspect.getfile(load_module_only)).resolve().parents[1]
    report = {
        "protocol": {
            **spec, "training_seed": 42,
            "action_modes": ["greedy", "sampled"],
            "history": "each checkpoint controls its own environment; shadow receives randomized A/B inputs with C and executed original actions preserved",
            "sampling": "unmodified softmax; common inverse-CDF action uniforms across checkpoints",
            "uncertainty": "pointwise 95% paired-episode bootstrap intervals; not training-seed variation or simultaneous curve bands",
            "coverage": "initialization at step zero and all 26 durable checkpoints; actual progress.csv steps",
        },
        "provenance": {
            "checkpoint_index_path": str(args.index.resolve()),
            "checkpoint_index_sha256": sha256(args.index),
            "training_run_id": index["run_id"],
            "training_revisions": index["training_revisions"],
            "source_sha256": sha256(Path(__file__)),
            "metric_helper_sha256": sha256(Path(inspect.getfile(summary))),
            "reuse_report_path": reuse,
            "reuse_report_sha256": None if reuse is None else sha256(Path(reuse)),
            "experiment_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "experiment_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
            "library_revision": subprocess.check_output(["git", "-C", str(library), "rev-parse", "HEAD"], text=True).strip(),
            "library_dirty": bool(subprocess.check_output(["git", "-C", str(library), "status", "--porcelain"], text=True).strip()),
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        },
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Complete curve saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
