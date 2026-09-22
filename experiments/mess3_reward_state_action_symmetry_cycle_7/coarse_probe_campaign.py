"""Run the exact coarse-filter probe over all cycle-7 variant-2 checkpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

from experiments.mess3_reward_state_action_symmetry_cycle_7.coarse_probe_analysis import (
    CONDITION,
    FIT_STEPS,
    TEST_STEPS,
    VARIANT,
    WARMUP,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.component_probe_campaign import (
    all_checkpoint_records,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.control_campaign import (
    SEEDS,
    STUDY,
    WorkItem,
    _download_module,
    _git,
    _sha256,
)


def work_items(repository: Path, scratch: Path) -> list[WorkItem]:
    """Return all 105 saved Reward-State Quotient checkpoint probes."""

    items: list[WorkItem] = []
    results = repository / STUDY / CONDITION / "results"
    for seed in SEEDS:
        matches = sorted(results.glob(f"*-seed{seed}-10m"))
        if len(matches) != 1:
            raise ValueError(
                f"{CONDITION} seed {seed} expected one run, got {matches}"
            )
        run_dir = matches[0]
        curve_path = run_dir / "checkpoint_probe_curve.json"
        for stage_index, (stage, record) in enumerate(
            all_checkpoint_records(curve_path)
        ):
            checkpoint_name = record.get("checkpoint_name")
            items.append(
                WorkItem(
                    condition=CONDITION,
                    variant=VARIANT,
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
                        / f"seed_{seed}"
                        / f"checkpoint_{stage_index}"
                    ),
                    output_path=(
                        scratch
                        / "reports"
                        / f"seed_{seed}"
                        / f"checkpoint_{stage_index}.json"
                    ),
                )
            )
    if len(items) != 105:
        raise ValueError(f"expected 105 checkpoint probes, got {len(items)}")
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
        actual = (
            int(metadata["n_fit"]),
            int(metadata["n_test"]),
            int(metadata["warmup"]),
        )
        expected = (fit_steps, test_steps, warmup)
        if actual != expected:
            raise ValueError(
                f"{item.output_path} protocol {actual} != requested {expected}"
            )
        if int(metadata["variant"]) != VARIANT:
            raise ValueError(f"{item.output_path} is not a variant-2 report")
        return f"cached seed={item.seed} checkpoint={item.stage_index}"

    download = _download_module(item)
    item.output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        (
            "experiments.mess3_reward_state_action_symmetry_cycle_7."
            "coarse_probe_analysis"
        ),
        "--variant",
        str(VARIANT),
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
    return f"finished seed={item.seed} checkpoint={item.stage_index}"


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
    seeds: dict[str, list[dict[str, object]]] = {}
    source_files: dict[str, dict[str, object]] = {}
    coarse_model: dict[str, object] | None = None
    source_commits: set[str] = set()
    harness_commits: set[str] = set()
    for item in items:
        report = json.loads(item.output_path.read_text())
        checkpoint = report["campaign_checkpoint"]
        source_files[checkpoint["checkpoint_probe_curve"]] = {
            "sha256": checkpoint["checkpoint_probe_curve_sha256"],
            "run_id": checkpoint["run_id"],
            "durability_manifest_key": checkpoint["manifest_key"],
        }
        if coarse_model is None:
            coarse_model = report["coarse_model"]
        elif coarse_model != report["coarse_model"]:
            raise ValueError("coarse HMM specification changed across reports")
        source_commits.add(str(report["source"]["experiment_git_commit"]))
        harness_commits.add(str(report["source"]["harness_git_commit"]))
        if report["source"]["experiment_git_dirty"]:
            raise ValueError(f"{item.output_path} came from a dirty checkout")
        seeds.setdefault(str(item.seed), []).append(
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
                "full_belief_replay_max_abs_error": report[
                    "full_belief_replay_max_abs_error"
                ],
                "target_difference": report["target_difference"],
                "targets": {
                    name: _metric_projection(target)
                    for name, target in report["targets"].items()
                },
            }
        )
    for records in seeds.values():
        records.sort(key=lambda record: int(record["stage_index"]))
    if set(map(int, seeds)) != set(SEEDS):
        raise ValueError("compact report does not contain seeds 42 through 56")
    if any(len(records) != 7 for records in seeds.values()):
        raise ValueError("each variant-2 seed must contain seven checkpoints")
    if len(source_commits) != 1 or len(harness_commits) != 1:
        raise ValueError("campaign reports do not share source revisions")
    if coarse_model is None:
        raise ValueError("campaign has no coarse HMM specification")

    return {
        "schema_version": 1,
        "study": "Belief-Quotient Ladder exact coarse-filter probe",
        "condition": "Reward-State Quotient",
        "implementation_condition": CONDITION,
        "seeds": seeds,
        "checkpoint_selection": {
            "analysis_scope": (
                "Every saved checkpoint in each variant-2 run, including "
                "initialization and the final checkpoint."
            ),
            "stages": [
                "init",
                "iteration_1",
                "iteration_2",
                "iteration_4",
                "iteration_8",
                "iteration_16",
                "final",
            ],
            "record_count": len(items),
        },
        "analysis": {
            "n_fit": fit_steps,
            "n_test": test_steps,
            "fit_test_streams": "independent",
            "seed_streams": {
                "probe_train": [510],
                "probe_test": [511],
            },
            "warmup_steps_per_episode": warmup,
            "policy_mode": "greedy_argmax",
            "sampling_distribution": "process_weighted_rollout",
            "representation": "post_final_layer_norm",
            "probe": "fixed-ridge affine least squares",
            "ridge": 1e-6,
            "targets": {
                "coarse_b2": (
                    "c_t, the B={state 3} component of a separately updated "
                    "exact two-state filter over A={states 1,2}, B={state 3}"
                ),
                "full_rho": (
                    "rho_t=b_3,t, the state-3 component of the exact full "
                    "three-state filter"
                ),
            },
            "target_timing": (
                "The reset prior is conditioned on the first token. Later "
                "updates use the previous executed action followed by the "
                "current token likelihood."
            ),
            "history_protocol": (
                "Complete episode prefixes are filtered before the per-episode "
                "warmup mask is applied."
            ),
            "metric": (
                "held-out affine-probe MSE divided by held-out target "
                "variance, equal to 1 - R^2"
            ),
            "aggregation_unit": (
                "This file contains seed-level results; uncertainty over the "
                "campaign should bootstrap complete training seeds."
            ),
            "interpretation": (
                "The targets are correlated but non-identical. Probe fit "
                "measures linear accessibility, not causal policy use or "
                "unique identification of the network's filtering algorithm."
            ),
        },
        "coarse_model": coarse_model,
        "source": {
            "repository": "https://github.com/Al-does/alex-rl-experiments",
            "git_commit": source_commits.pop(),
            "harness_git_commit": harness_commits.pop(),
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
        default=STUDY / "artifacts" / "coarse_filter_probes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=STUDY / "results" / "coarse_probe_trajectories.json",
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
    missing = [
        item.output_path for item in items if not item.output_path.is_file()
    ]
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
