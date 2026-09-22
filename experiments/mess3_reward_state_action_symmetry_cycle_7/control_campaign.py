"""Download cycle-7 modules, run frozen controls, and compact the results."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import boto3

from experiments.mess3_reward_state_action_symmetry_cycle_7.control_analysis import (
    FULL_STEPS,
)


STUDY = Path("experiments/mess3_reward_state_action_symmetry_cycle_7")
CONDITIONS = (
    "variant_1_ctx32_ent",
    "variant_2_ctx32_ent",
    "variant_3_ctx32_ent",
)
SEEDS = tuple(range(42, 57))
ITERATION_STAGES = (1, 2, 4, 8, 16)
MODULE_FILES = (
    "class_and_ctor_args.pkl",
    "metadata.json",
    "module_state.pkl",
)


@dataclass(frozen=True, slots=True)
class WorkItem:
    condition: str
    variant: int
    seed: int
    stage: str
    stage_index: int
    training_iteration: int | None
    agent_steps: int
    run_id: str
    checkpoint_name: str | None
    curve_path: Path
    run_manifest_path: Path
    module_path: Path
    output_path: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _select_records(curve_path: Path) -> list[tuple[str, dict[str, object]]]:
    checkpoints = json.loads(curve_path.read_text())["checkpoints"]
    if not checkpoints or int(checkpoints[0]["agent_steps"]) != 0:
        raise ValueError(f"{curve_path} lacks the initialization checkpoint")
    by_iteration = {
        int(record["training_iteration"]): record
        for record in checkpoints[1:]
    }
    missing = set(ITERATION_STAGES) - set(by_iteration)
    if missing:
        raise ValueError(
            f"{curve_path} lacks training iterations {sorted(missing)}"
        )
    selected = [("init", checkpoints[0])]
    selected.extend(
        (f"iteration_{iteration}", by_iteration[iteration])
        for iteration in ITERATION_STAGES
    )
    selected.append(("final", checkpoints[-1]))
    if len({int(record["agent_steps"]) for _, record in selected}) != 7:
        raise ValueError(f"{curve_path} selected duplicate checkpoints")
    return selected


def _work_items(repository: Path, scratch: Path) -> list[WorkItem]:
    items: list[WorkItem] = []
    for condition_index, condition in enumerate(CONDITIONS, start=1):
        results = repository / STUDY / condition / "results"
        for seed in SEEDS:
            matches = sorted(results.glob(f"*-seed{seed}-10m"))
            if len(matches) != 1:
                raise ValueError(
                    f"{condition} seed {seed} expected one run, got {matches}"
                )
            run_dir = matches[0]
            curve_path = run_dir / "checkpoint_probe_curve.json"
            for stage_index, (stage, record) in enumerate(
                _select_records(curve_path)
            ):
                checkpoint_name = record.get("checkpoint_name")
                items.append(
                    WorkItem(
                        condition=condition,
                        variant=condition_index,
                        seed=seed,
                        stage=stage,
                        stage_index=stage_index,
                        training_iteration=record.get("training_iteration"),
                        agent_steps=int(record["agent_steps"]),
                        run_id=run_dir.name,
                        checkpoint_name=(
                            None
                            if checkpoint_name is None
                            else str(checkpoint_name)
                        ),
                        curve_path=curve_path,
                        run_manifest_path=run_dir / "run_manifest.json",
                        module_path=(
                            scratch
                            / "modules"
                            / condition
                            / f"seed_{seed}"
                            / f"stage_{stage_index}"
                        ),
                        output_path=(
                            scratch
                            / "reports"
                            / condition
                            / f"seed_{seed}"
                            / f"stage_{stage_index}.json"
                        ),
                    )
                )
    return items


def _s3_client() -> object:
    required = (
        "B2_APPLICATION_KEY_ID",
        "B2_APPLICATION_KEY",
        "B2_ENDPOINT",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing B2 credentials: {', '.join(missing)}")
    return boto3.client(
        "s3",
        endpoint_url=os.environ["B2_ENDPOINT"],
        aws_access_key_id=os.environ["B2_APPLICATION_KEY_ID"],
        aws_secret_access_key=os.environ["B2_APPLICATION_KEY"],
    )


def _module_records(
    manifest: dict[str, object],
    item: WorkItem,
) -> tuple[str, dict[str, dict[str, object]]]:
    if item.checkpoint_name is None:
        marker = (
            "initial_checkpoint/learner_group/learner/rl_module/"
            "default_policy/"
        )
    else:
        marker = (
            f"{item.checkpoint_name}/learner_group/learner/rl_module/"
            "default_policy/"
        )
    records = [
        record
        for record in manifest["files"]
        if marker in str(record["relative_path"])
        and Path(str(record["relative_path"])).name in MODULE_FILES
    ]
    roots = {
        str(Path(str(record["relative_path"])).parent) for record in records
    }
    if len(roots) != 1:
        raise ValueError(
            f"{item.run_id} {item.stage} has module roots {sorted(roots)}"
        )
    by_name = {
        Path(str(record["relative_path"])).name: record for record in records
    }
    if set(by_name) != set(MODULE_FILES):
        raise ValueError(
            f"{item.run_id} {item.stage} has module files {sorted(by_name)}"
        )
    return roots.pop(), by_name


def _download_module(item: WorkItem) -> dict[str, object]:
    run_manifest = json.loads(item.run_manifest_path.read_text())
    remote = run_manifest["remote_artifacts"]
    bucket = str(remote["bucket"])
    manifest_key = str(remote["canonical_manifest_key"])
    client = _s3_client()
    manifest = json.loads(
        client.get_object(Bucket=bucket, Key=manifest_key)["Body"].read()
    )
    checkpoint_root, records = _module_records(manifest, item)
    item.module_path.mkdir(parents=True, exist_ok=True)
    for name in MODULE_FILES:
        destination = item.module_path / name
        expected = str(records[name]["sha256"])
        if destination.is_file() and _sha256(destination) == expected:
            continue
        client.download_file(bucket, str(records[name]["key"]), str(destination))
        actual = _sha256(destination)
        if actual != expected:
            destination.unlink()
            raise ValueError(
                f"hash mismatch for {item.run_id} {item.stage}/{name}"
            )
    return {
        "manifest_key": manifest_key,
        "checkpoint_relative_path": checkpoint_root,
        "module_files": {
            name: {
                "sha256": records[name]["sha256"],
                "size_bytes": records[name]["size_bytes"],
            }
            for name in MODULE_FILES
        },
    }


def _run_item(
    item: WorkItem,
    *,
    n_steps: int,
    device: str,
    threads: int,
) -> str:
    if item.output_path.is_file():
        return (
            f"cached {item.condition} seed={item.seed} "
            f"stage={item.stage_index}"
        )
    download = _download_module(item)
    item.output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        (
            "experiments.mess3_reward_state_action_symmetry_cycle_7."
            "control_analysis"
        ),
        "--variant",
        str(item.variant),
        "--checkpoint",
        str(item.module_path),
        "--output",
        str(item.output_path),
        "--seed",
        str(item.seed),
        "--steps",
        str(n_steps),
        "--device",
        device,
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["MKL_NUM_THREADS"] = str(threads)
    subprocess.run(command, check=True, env=environment)
    report = json.loads(item.output_path.read_text())
    repository = Path(__file__).resolve().parents[2]
    report["campaign_checkpoint"] = {
        "condition": item.condition,
        "variant": item.variant,
        "seed": item.seed,
        "stage": item.stage,
        "stage_index": item.stage_index,
        "training_iteration": item.training_iteration,
        "agent_steps": item.agent_steps,
        "run_id": item.run_id,
        "checkpoint_name": item.checkpoint_name,
        "checkpoint_probe_curve": item.curve_path.relative_to(
            repository
        ).as_posix(),
        "checkpoint_probe_curve_sha256": _sha256(item.curve_path),
        **download,
    }
    item.output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return (
        f"finished {item.condition} seed={item.seed} "
        f"stage={item.stage_index}"
    )


def _metric_projection(score: dict[str, object]) -> dict[str, object]:
    return {
        key: score[key]
        for key in (
            "mse",
            "target_variance",
            "global_mse_ratio",
            "r_squared",
            "n_evaluated",
        )
        if key in score
    }


def _compact_report(
    repository: Path,
    items: list[WorkItem],
    *,
    n_steps: int,
) -> dict[str, object]:
    conditions: dict[str, object] = {
        condition: {"seeds": {}} for condition in CONDITIONS
    }
    source_files: dict[str, dict[str, object]] = {}
    for item in items:
        report = json.loads(item.output_path.read_text())
        checkpoint = report["campaign_checkpoint"]
        source_files[checkpoint["checkpoint_probe_curve"]] = {
            "sha256": checkpoint["checkpoint_probe_curve_sha256"],
            "run_id": checkpoint["run_id"],
            "durability_manifest_key": checkpoint["manifest_key"],
        }
        projected = {
            control: {
                target_mode: _metric_projection(scores)
                for target_mode, scores in control_report.items()
            }
            for control, control_report in report["controls"].items()
        }
        seed_report = conditions[item.condition]["seeds"].setdefault(
            str(item.seed),
            [],
        )
        seed_report.append(
            {
                "stage": item.stage,
                "stage_index": item.stage_index,
                "training_iteration": item.training_iteration,
                "agent_steps": item.agent_steps,
                "checkpoint": {
                    "run_id": item.run_id,
                    "checkpoint_name": item.checkpoint_name,
                    "relative_path": checkpoint["checkpoint_relative_path"],
                    "module_files": checkpoint["module_files"],
                },
                "belief_replay_max_abs_error": report[
                    "belief_replay_max_abs_error"
                ],
                "history_control_seeds": report["metadata"][
                    "history_control_seeds"
                ],
                "controls": projected,
            }
        )
    for condition in CONDITIONS:
        for records in conditions[condition]["seeds"].values():
            records.sort(key=lambda record: int(record["stage_index"]))
    return {
        "schema_version": 1,
        "study": "Belief-Quotient Ladder",
        "conditions": conditions,
        "checkpoint_selection": {
            "stages": [
                "init",
                "iteration_1",
                "iteration_2",
                "iteration_4",
                "iteration_8",
                "iteration_16",
                "final",
            ],
            "description": (
                "Initialization, shared log-spaced training iterations "
                "1/2/4/8/16, and each run's final checkpoint. Curves are "
                "aligned by stage without interpolation; plotted x positions "
                "are the mean realized environment steps across seeds."
            ),
            "exception": (
                "Trivial Quotient seed 55 also retained iteration 32; it is "
                "omitted because that stage is unavailable for the other seeds."
            ),
        },
        "analysis": {
            "n_fit": n_steps,
            "n_test": n_steps,
            "fit_test_streams": "independent",
            "warmup_steps_per_episode": 128,
            "transformer_receptive_field": 128,
            "policy_mode": "greedy_argmax",
            "representation": "post_final_layer_norm",
            "metric": (
                "held-out affine-probe MSE divided by held-out target "
                "variance, equal to 1 - R^2"
            ),
            "plotted_control_target": (
                "original time-indexed exact belief from the unmodified "
                "on-policy rollout"
            ),
            "aggregation_unit": (
                "This file contains seed-level results; paper uncertainty is "
                "computed by bootstrapping complete training seeds."
            ),
            "interpretation": (
                "Frozen replay controls are descriptive tests of linear "
                "accessibility, not closed-loop causal interventions."
            ),
        },
        "source": {
            "repository": "https://github.com/Al-does/alex-rl-experiments",
            "git_commit": _git(repository, "rev-parse", "HEAD"),
            "harness_git_commit": _git(
                Path(__import__("analysis").__file__).resolve().parents[1],
                "rev-parse",
                "HEAD",
            ),
            "files": [
                {"path": path, **metadata}
                for path, metadata in sorted(source_files.items())
            ],
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scratch",
        type=Path,
        default=STUDY / "artifacts" / "frozen_probe_controls",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=STUDY / "results" / "frozen_probe_controls.json",
    )
    parser.add_argument("--steps", type=int, default=FULL_STEPS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--overwrite-summary", action="store_true")
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[2]
    scratch = (
        args.scratch
        if args.scratch.is_absolute()
        else repository / args.scratch
    )
    output = (
        args.output if args.output.is_absolute() else repository / args.output
    )
    items = _work_items(repository, scratch)
    run_items = items if args.limit is None else items[: args.limit]
    if not args.aggregate_only:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = [
                executor.submit(
                    _run_item,
                    item,
                    n_steps=args.steps,
                    device=args.device,
                    threads=args.threads_per_worker,
                )
                for item in run_items
            ]
            for future in concurrent.futures.as_completed(futures):
                print(future.result(), flush=True)
    missing = [item.output_path for item in items if not item.output_path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"campaign is incomplete; {len(missing)} reports are missing"
        )
    if output.exists() and not args.overwrite_summary:
        raise FileExistsError(f"refusing to overwrite {output}")
    compact = _compact_report(repository, items, n_steps=args.steps)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
