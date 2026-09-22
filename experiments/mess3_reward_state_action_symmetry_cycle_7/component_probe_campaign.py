"""Run rho/delta probes for every saved cycle-7 quotient checkpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

from experiments.mess3_reward_state_action_symmetry_cycle_7.component_probe_analysis import (
    FIT_STEPS,
    TEST_STEPS,
    WARMUP,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.control_campaign import (
    CONDITIONS,
    SEEDS,
    STUDY,
    WorkItem,
    _download_module,
    _git,
    _sha256,
)


def all_checkpoint_records(
    curve_path: Path,
) -> list[tuple[str, dict[str, object]]]:
    """Return initialization plus every saved checkpoint in trajectory order."""

    checkpoints = json.loads(curve_path.read_text())["checkpoints"]
    if not checkpoints or int(checkpoints[0]["agent_steps"]) != 0:
        raise ValueError(f"{curve_path} lacks the initialization checkpoint")
    steps = [int(record["agent_steps"]) for record in checkpoints]
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError(f"{curve_path} checkpoint steps are not increasing")
    names = [
        str(record["checkpoint_name"])
        for record in checkpoints[1:]
        if record.get("checkpoint_name") is not None
    ]
    if len(names) != len(checkpoints) - 1 or len(set(names)) != len(names):
        raise ValueError(f"{curve_path} has invalid checkpoint names")

    records = []
    for index, record in enumerate(checkpoints):
        if index == 0:
            stage = "init"
        elif index == len(checkpoints) - 1:
            stage = "final"
        else:
            stage = f"iteration_{int(record['training_iteration'])}"
        records.append((stage, record))
    return records


def work_items(repository: Path, scratch: Path) -> list[WorkItem]:
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
                all_checkpoint_records(curve_path)
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
                            / f"checkpoint_{stage_index}"
                        ),
                        output_path=(
                            scratch
                            / "reports"
                            / condition
                            / f"seed_{seed}"
                            / f"checkpoint_{stage_index}.json"
                        ),
                    )
                )
    return items


def _run_item(
    item: WorkItem,
    *,
    fit_steps: int,
    test_steps: int,
    warmup: int,
    device: str,
    threads: int,
) -> str:
    if item.output_path.is_file():
        report = json.loads(item.output_path.read_text())
        metadata = report["metadata"]
        expected = (fit_steps, test_steps, warmup)
        actual = (
            int(metadata["n_fit"]),
            int(metadata["n_test"]),
            int(metadata["warmup"]),
        )
        if actual != expected:
            raise ValueError(
                f"{item.output_path} protocol {actual} != requested {expected}"
            )
        return (
            f"cached {item.condition} seed={item.seed} "
            f"checkpoint={item.stage_index}"
        )

    download = _download_module(item)
    item.output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        (
            "experiments.mess3_reward_state_action_symmetry_cycle_7."
            "component_probe_analysis"
        ),
        "--variant",
        str(item.variant),
        "--checkpoint",
        str(item.module_path),
        "--output",
        str(item.output_path),
        "--seed",
        str(item.seed),
        "--fit-steps",
        str(fit_steps),
        "--test-steps",
        str(test_steps),
        "--warmup",
        str(warmup),
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
        f"checkpoint={item.stage_index}"
    )


def _metric_projection(target: dict[str, object]) -> dict[str, object]:
    return {
        key: target[key]
        for key in (
            "symbol",
            "definition",
            "description",
            "mse",
            "target_variance",
            "global_mse_ratio",
            "r_squared",
            "n_evaluated",
        )
    }


def compact_report(
    repository: Path,
    items: list[WorkItem],
    *,
    fit_steps: int,
    test_steps: int,
    warmup: int,
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
                "targets": {
                    name: _metric_projection(target)
                    for name, target in report["targets"].items()
                },
            }
        )
    for condition in CONDITIONS:
        for records in conditions[condition]["seeds"].values():
            records.sort(key=lambda record: int(record["stage_index"]))

    return {
        "schema_version": 1,
        "study": "Belief-Quotient Ladder scalar components",
        "conditions": conditions,
        "checkpoint_selection": {
            "analysis_scope": (
                "Every saved checkpoint in each run, including initialization "
                "and the final checkpoint."
            ),
            "shared_figure_stages": [
                "init",
                "iteration_1",
                "iteration_2",
                "iteration_4",
                "iteration_8",
                "iteration_16",
                "final",
            ],
            "alignment": (
                "Paper curves align the stages shared by all 15 seeds within "
                "each condition without interpolation; x positions are mean "
                "realized environment steps."
            ),
            "exception": (
                "Trivial Quotient seed 55 also contains iteration 32. It is "
                "analyzed and retained here, but omitted from the 15-seed "
                "figure because no other seed has that saved stage."
            ),
        },
        "analysis": {
            "n_fit": fit_steps,
            "n_test": test_steps,
            "fit_test_streams": "independent",
            "warmup_steps_per_episode": warmup,
            "policy_mode": "greedy_argmax",
            "sampling_distribution": "process_weighted_rollout",
            "representation": "post_final_layer_norm",
            "probe": "fixed-ridge affine least squares",
            "ridge": 1e-6,
            "targets": {
                "rho": "rho_t = b_3,t",
                "delta": "delta_t = b_1,t - b_2,t",
            },
            "metric": (
                "held-out affine-probe MSE divided by held-out target "
                "variance, equal to 1 - R^2"
            ),
            "aggregation_unit": (
                "This file contains seed-level results; paper uncertainty is "
                "computed by bootstrapping complete training seeds."
            ),
            "interpretation": (
                "Scalar-probe fit measures linear accessibility, not causal "
                "policy use."
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
        default=STUDY / "artifacts" / "component_probes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=STUDY / "results" / "component_probe_trajectories.json",
    )
    parser.add_argument("--fit-steps", type=int, default=FIT_STEPS)
    parser.add_argument("--test-steps", type=int, default=TEST_STEPS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
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
    items = work_items(repository, scratch)
    run_items = items if args.limit is None else items[: args.limit]
    if not args.aggregate_only:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = [
                executor.submit(
                    _run_item,
                    item,
                    fit_steps=args.fit_steps,
                    test_steps=args.test_steps,
                    warmup=args.warmup,
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
    compact = compact_report(
        repository,
        items,
        fit_steps=args.fit_steps,
        test_steps=args.test_steps,
        warmup=args.warmup,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
