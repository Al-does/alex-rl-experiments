"""Download archived modules, run frozen controls, and compact the results."""

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

from experiments.two_factor_reward_state_REINFORCE_cycle_4.control_analysis import (
    FULL_STEPS,
    N_NULL_REPEATS,
    TARGETS,
)


STUDY = Path("experiments/two_factor_reward_state_REINFORCE_cycle_4")
CONDITIONS = ("reward_both", "reward_factor_1")
SEEDS = tuple(range(42, 57))
CHECKPOINT_STEPS = (0, 10_000_000, 20_000_000, 30_000_000)
MODULE_FILES = (
    "class_and_ctor_args.pkl",
    "metadata.json",
    "module_state.pkl",
)


@dataclass(frozen=True, slots=True)
class WorkItem:
    condition: str
    seed: int
    agent_steps: int
    run_id: str
    checkpoint: str
    summary_path: Path
    run_manifest_path: Path
    module_path: Path
    output_path: Path


def _run_ids(condition: str, seed: int) -> tuple[str, ...]:
    prefix = f"two_factor_reward_state_REINFORCE_cycle_4-{condition}-seed{seed}"
    if condition == "reward_both":
        return (f"{prefix}-30m",)
    if seed == 42:
        return (
            "20260901T212917Z-347a5b4d",
            f"{prefix}-continued-30m",
        )
    if seed <= 46:
        return (
            f"{prefix}-10m",
            f"{prefix}-continued-30m",
        )
    return (f"{prefix}-30m",)


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


def _work_items(repository: Path, scratch: Path) -> list[WorkItem]:
    items: list[WorkItem] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            reports: dict[int, tuple[dict[str, object], str, Path]] = {}
            previous_max = -1
            for run_id in _run_ids(condition, seed):
                run_dir = repository / STUDY / condition / "results" / run_id
                summary_path = run_dir / "condition_summary.json"
                payload = json.loads(summary_path.read_text())
                segment = sorted(
                    payload["checkpoint_reports"],
                    key=lambda report: int(report["agent_steps"]),
                )
                for report in segment:
                    step = int(report["agent_steps"])
                    if step > previous_max:
                        reports[step] = (report, run_id, summary_path)
                previous_max = max(
                    previous_max,
                    int(segment[-1]["agent_steps"]),
                )
            missing = set(CHECKPOINT_STEPS) - set(reports)
            if missing:
                raise ValueError(
                    f"{condition} seed {seed} lacks checkpoints {sorted(missing)}"
                )
            for step in CHECKPOINT_STEPS:
                report, run_id, summary_path = reports[step]
                checkpoint = str(report["checkpoint"])
                if step == 0:
                    relative_checkpoint = Path("initial_checkpoint")
                else:
                    relative_checkpoint = Path("step_checkpoints") / checkpoint
                run_dir = summary_path.parent
                module_path = (
                    scratch
                    / "modules"
                    / condition
                    / f"seed_{seed}"
                    / f"steps_{step:09d}"
                )
                output_path = (
                    scratch
                    / "reports"
                    / condition
                    / f"seed_{seed}"
                    / f"steps_{step:09d}.json"
                )
                item = WorkItem(
                    condition=condition,
                    seed=seed,
                    agent_steps=step,
                    run_id=run_id,
                    checkpoint=relative_checkpoint.as_posix(),
                    summary_path=summary_path,
                    run_manifest_path=run_dir / "run_manifest.json",
                    module_path=module_path,
                    output_path=output_path,
                )
                items.append(item)
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


def _download_module(item: WorkItem) -> dict[str, object]:
    run_manifest = json.loads(item.run_manifest_path.read_text())
    remote = run_manifest["remote_artifacts"]
    bucket = str(remote["bucket"])
    manifest_key = str(remote["canonical_manifest_key"])
    client = _s3_client()
    manifest = json.loads(
        client.get_object(Bucket=bucket, Key=manifest_key)["Body"].read()
    )
    root = (
        f"{item.checkpoint}/learner_group/learner/rl_module/default_policy"
    )
    records = {
        Path(record["relative_path"]).name: record
        for record in manifest["files"]
        if str(record["relative_path"]).startswith(root + "/")
    }
    if set(records) != set(MODULE_FILES):
        raise ValueError(
            f"{item.run_id} {item.checkpoint} has module files {sorted(records)}"
        )
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
                f"hash mismatch for {item.run_id} {item.checkpoint}/{name}"
            )
    return {
        "manifest_key": manifest_key,
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
    null_repeats: int,
    device: str,
    threads: int,
) -> str:
    if item.output_path.is_file():
        return f"cached {item.condition} seed={item.seed} steps={item.agent_steps}"
    download = _download_module(item)
    item.output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        (
            "experiments.two_factor_reward_state_REINFORCE_cycle_4."
            "control_analysis"
        ),
        "--condition",
        item.condition,
        "--checkpoint",
        str(item.module_path),
        "--output",
        str(item.output_path),
        "--seed",
        str(item.seed),
        "--steps",
        str(n_steps),
        "--null-repeats",
        str(null_repeats),
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
        "seed": item.seed,
        "agent_steps": item.agent_steps,
        "run_id": item.run_id,
        "checkpoint_relative_path": item.checkpoint,
        "condition_summary": item.summary_path.relative_to(
            repository
        ).as_posix(),
        "condition_summary_sha256": _sha256(item.summary_path),
        **download,
    }
    item.output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return f"finished {item.condition} seed={item.seed} steps={item.agent_steps}"


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
    }


def _compact_report(
    repository: Path,
    items: list[WorkItem],
    *,
    n_steps: int,
    null_repeats: int,
) -> dict[str, object]:
    conditions: dict[str, object] = {
        condition: {"seeds": {}} for condition in CONDITIONS
    }
    source_files: dict[str, dict[str, object]] = {}
    for item in items:
        report = json.loads(item.output_path.read_text())
        checkpoint = report["campaign_checkpoint"]
        source_files[checkpoint["condition_summary"]] = {
            "sha256": checkpoint["condition_summary_sha256"],
            "run_id": checkpoint["run_id"],
            "durability_manifest_key": checkpoint["manifest_key"],
        }
        projected_targets = {}
        for target in TARGETS:
            target_report = report["targets"][target]
            projected_targets[target] = {
                variant: {
                    mode: _metric_projection(scores)
                    for mode, scores in target_report[variant].items()
                }
                for variant in (
                    "intact",
                    "shuffled_history",
                    "empirical_random_history",
                    "uniform_random_history",
                )
            }
            projected_targets[target]["nulls"] = {
                name: _metric_projection(null["mean"])
                for name, null in target_report["nulls"].items()
            }
        conditions[item.condition]["seeds"].setdefault(
            str(item.seed),
            {},
        )[str(item.agent_steps)] = {
            "checkpoint": {
                "run_id": item.run_id,
                "relative_path": item.checkpoint,
                "module_files": checkpoint["module_files"],
            },
            "belief_replay_max_abs_error": report[
                "belief_replay_max_abs_error"
            ],
            "history_control_seeds": report["metadata"][
                "history_control_seeds"
            ],
            "targets": projected_targets,
        }
    return {
        "schema_version": 1,
        "study": "Reward Relevance",
        "conditions": conditions,
        "checkpoint_selection": {
            "agent_steps": list(CHECKPOINT_STEPS),
            "description": (
                "Initialization and exact 10M-step intervals shared by all "
                "15 seeds in both conditions; no interpolation."
            ),
        },
        "analysis": {
            "n_fit": n_steps,
            "n_test": n_steps,
            "n_null_repeats": null_repeats,
            "fit_test_streams": "independent",
            "warmup_steps_per_episode": 8,
            "policy_mode": "greedy_argmax",
            "metric": (
                "held-out affine-probe MSE divided by held-out target "
                "variance, equal to 1 - R^2"
            ),
            "aggregation_unit": (
                "This file contains seed-level results; paper uncertainty is "
                "computed by bootstrapping complete training seeds."
            ),
            "interpretation": (
                "Frozen replay and probe nulls are descriptive tests of "
                "linear accessibility, not closed-loop causal interventions."
            ),
        },
        "source": {
            "repository": "https://github.com/Al-does/alex-rl-experiments",
            "git_commit": _git(repository, "rev-parse", "HEAD"),
            "harness_git_commit": _git(
                repository.parent / "rl-harness",
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
    parser.add_argument("--null-repeats", type=int, default=N_NULL_REPEATS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=1)
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
    if not args.aggregate_only:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = [
                executor.submit(
                    _run_item,
                    item,
                    n_steps=args.steps,
                    null_repeats=args.null_repeats,
                    device=args.device,
                    threads=args.threads_per_worker,
                )
                for item in items
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
    compact = _compact_report(
        repository,
        items,
        n_steps=args.steps,
        null_repeats=args.null_repeats,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
