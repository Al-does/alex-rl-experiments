"""Run matched coarse-observation probes over the cycle-7 quotient ladder."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from experiments.mess3_reward_state_action_symmetry_cycle_7.coarse_probe_analysis import (
    CONDITION_LABELS,
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
    _select_records,
    _sha256,
)


STAGES = (
    "init",
    "iteration_1",
    "iteration_2",
    "iteration_4",
    "iteration_8",
    "iteration_16",
    "final",
)
COMPONENT_RESULT = STUDY / "results" / "component_probe_trajectories.json"


def work_items(repository: Path, scratch: Path) -> list[WorkItem]:
    """Return the seven shared checkpoints for all 45 trained agents."""

    items: list[WorkItem] = []
    for variant, condition in enumerate(CONDITIONS, start=1):
        results = repository / STUDY / condition / "results"
        for seed in SEEDS:
            matches = sorted(results.glob(f"*-seed{seed}-10m"))
            if len(matches) != 1:
                raise ValueError(
                    f"{condition} seed {seed} expected one run, got {matches}"
                )
            run_dir = matches[0]
            curve_path = run_dir / "checkpoint_probe_curve.json"
            records = _select_records(curve_path)
            if tuple(stage for stage, _ in records) != STAGES:
                raise ValueError(
                    f"{condition} seed {seed} has unmatched stages"
                )
            for stage_index, (stage, record) in enumerate(records):
                checkpoint_name = record.get("checkpoint_name")
                items.append(
                    WorkItem(
                        condition=condition,
                        variant=variant,
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
    if len(items) != 315:
        raise ValueError(f"expected 315 checkpoint probes, got {len(items)}")
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
            int(metadata["variant"]),
            str(metadata["condition"]),
        )
        expected = (
            fit_steps,
            test_steps,
            warmup,
            item.variant,
            item.condition,
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
            "coarse_probe_analysis"
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


def _normalized_error(target: dict[str, object]) -> float:
    ratio = float(target["global_mse_ratio"])
    complement = 1.0 - float(target["r_squared"])
    if abs(ratio - complement) > 1e-10:
        raise ValueError("normalized error is inconsistent with 1 - R^2")
    return ratio


def _preference_summary(
    records_by_seed: dict[str, list[dict[str, object]]],
    stage: str,
) -> dict[str, object]:
    per_seed = []
    for seed in SEEDS:
        record = next(
            record
            for record in records_by_seed[str(seed)]
            if record["stage"] == stage
        )
        coarse = _normalized_error(record["targets"]["coarse_b2"])
        full = _normalized_error(record["targets"]["full_rho"])
        per_seed.append(
            {
                "seed": seed,
                "coarse_observation_error": coarse,
                "full_rho_error": full,
                "full_minus_coarse": full - coarse,
            }
        )
    differences = np.asarray(
        [record["full_minus_coarse"] for record in per_seed],
        dtype=np.float64,
    )
    tolerance = 1e-12
    return {
        "per_seed": per_seed,
        "mean_full_minus_coarse": float(np.mean(differences)),
        "median_full_minus_coarse": float(np.median(differences)),
        "min_full_minus_coarse": float(np.min(differences)),
        "max_full_minus_coarse": float(np.max(differences)),
        "coarse_favored_seed_count": int(np.sum(differences > tolerance)),
        "full_rho_favored_seed_count": int(
            np.sum(differences < -tolerance)
        ),
        "tie_seed_count": int(np.sum(np.abs(differences) <= tolerance)),
        "n_seeds": len(per_seed),
    }


def matched_seed_comparison(
    conditions: dict[str, dict[str, object]],
) -> dict[str, object]:
    stages: dict[str, object] = {}
    for stage in STAGES:
        summaries = {
            condition: _preference_summary(
                conditions[condition]["seeds"],
                stage,
            )
            for condition in CONDITIONS
        }
        contrasts = []
        for seed_index, seed in enumerate(SEEDS):
            trivial = summaries[CONDITIONS[0]]["per_seed"][seed_index][
                "full_minus_coarse"
            ]
            reward = summaries[CONDITIONS[1]]["per_seed"][seed_index][
                "full_minus_coarse"
            ]
            full_state = summaries[CONDITIONS[2]]["per_seed"][seed_index][
                "full_minus_coarse"
            ]
            contrasts.append(
                {
                    "seed": seed,
                    "reward_state_minus_trivial_preference": reward
                    - trivial,
                    "reward_state_minus_full_state_preference": reward
                    - full_state,
                }
            )
        stages[stage] = {
            "conditions": summaries,
            "matched_condition_contrasts": contrasts,
        }

    final = stages["final"]["conditions"]
    coarse_mean_favored = [
        condition
        for condition in CONDITIONS
        if final[condition]["mean_full_minus_coarse"] > 0.0
    ]
    return {
        "metric": {
            "name": "target_specific_normalized_error",
            "definition": (
                "Each target's held-out affine-probe MSE divided by its own "
                "held-out variance, equal to 1 - R^2."
            ),
            "preference": (
                "full_minus_coarse > 0 favors the coarse-observation target; "
                "full_minus_coarse < 0 favors full rho."
            ),
        },
        "stages": stages,
        "final_descriptive_readout": {
            "conditions_with_lower_mean_coarse_error": coarse_mean_favored,
            "reward_state_uniquely_favors_coarse_by_mean": (
                coarse_mean_favored == [CONDITIONS[1]]
            ),
            "full_state_favors_full_rho_by_mean": (
                final[CONDITIONS[2]]["mean_full_minus_coarse"] < 0.0
            ),
            "scope": (
                "These are matched-seed differences in linear probe "
                "accessibility. They are not environmental state, token, "
                "reward, or next-observation prediction accuracies."
            ),
        },
    }


def compact_report(
    repository: Path,
    items: list[WorkItem],
    *,
    fit_steps: int,
    test_steps: int,
    warmup: int,
) -> dict[str, object]:
    conditions: dict[str, dict[str, object]] = {
        condition: {
            "variant": variant,
            "label": CONDITION_LABELS[variant],
            "seeds": {},
        }
        for variant, condition in enumerate(CONDITIONS, start=1)
    }
    source_files: dict[str, dict[str, object]] = {}
    coarse_models: dict[str, dict[str, object]] = {}
    source_commits: set[str] = set()
    harness_commits: set[str] = set()
    component_path = repository / COMPONENT_RESULT
    component = json.loads(component_path.read_text())
    component_conditions = component["conditions"]

    for item in items:
        report = json.loads(item.output_path.read_text())
        checkpoint = report["campaign_checkpoint"]
        source_files[checkpoint["checkpoint_probe_curve"]] = {
            "sha256": checkpoint["checkpoint_probe_curve_sha256"],
            "run_id": checkpoint["run_id"],
            "durability_manifest_key": checkpoint["manifest_key"],
        }
        previous_model = coarse_models.setdefault(
            item.condition,
            report["coarse_model"],
        )
        if previous_model != report["coarse_model"]:
            raise ValueError(
                f"{item.condition} filter specification changed across reports"
            )
        source_commits.add(str(report["source"]["experiment_git_commit"]))
        harness_commits.add(str(report["source"]["harness_git_commit"]))
        if report["source"]["experiment_git_dirty"]:
            raise ValueError(f"{item.output_path} came from a dirty checkout")

        component_record = next(
            record
            for record in component_conditions[item.condition]["seeds"][
                str(item.seed)
            ]
            if record["stage"] == item.stage
        )
        report_full = _normalized_error(report["targets"]["full_rho"])
        component_full = _normalized_error(
            component_record["targets"]["rho"]
        )
        if abs(report_full - component_full) > 1e-12:
            raise ValueError(
                f"{item.condition} seed {item.seed} stage {item.stage} "
                "does not reproduce the scalar-component rho score"
            )

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

    for condition in CONDITIONS:
        seeds = conditions[condition]["seeds"]
        if set(map(int, seeds)) != set(SEEDS):
            raise ValueError(f"{condition} does not contain seeds 42 through 56")
        for records in seeds.values():
            records.sort(key=lambda record: int(record["stage_index"]))
            if tuple(record["stage"] for record in records) != STAGES:
                raise ValueError(f"{condition} has unmatched checkpoint stages")
        conditions[condition]["coarse_model"] = coarse_models[condition]
    if len(source_commits) != 1 or len(harness_commits) != 1:
        raise ValueError("campaign reports do not share source revisions")

    return {
        "schema_version": 2,
        "study": "Belief-Quotient Ladder exact coarse-observation probes",
        "conditions": conditions,
        "checkpoint_selection": {
            "analysis_scope": (
                "The seven semantic stages shared by every run: "
                "initialization, iterations 1/2/4/8/16, and final."
            ),
            "stages": list(STAGES),
            "record_count": len(items),
            "exception": (
                "Trivial Quotient seed 55's unique iteration-32 checkpoint is "
                "excluded because it is not shared across matched seeds."
            ),
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
                    "state-3 posterior from exact recursive filtering after "
                    "merging observation tokens A/B into not-C"
                ),
                "full_rho": (
                    "rho_t=b_3,t from the exact full-observation three-state "
                    "filter"
                ),
            },
            "filter_construction": (
                "Variants 1 and 2 use separately verified exact two-state "
                "quotient filters over {states 1,2}/{state 3}. Variant 3 is "
                "not lumpable under its actions and therefore uses the exact "
                "full three-state transition model with coarsened emissions."
            ),
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
                "target-specific held-out affine-probe MSE divided by that "
                "target's held-out variance, equal to 1 - R^2"
            ),
            "aggregation_unit": (
                "Seed-level trajectories are retained; comparisons match the "
                "same training seed and semantic checkpoint across conditions."
            ),
            "interpretation": (
                "Probe scores measure linear accessibility in network "
                "representations. No environmental prediction accuracy is "
                "estimated, and the results do not establish causal policy use."
            ),
        },
        "matched_seed_comparison": matched_seed_comparison(conditions),
        "source": {
            "repository": "https://github.com/Al-does/alex-rl-experiments",
            "git_commit": source_commits.pop(),
            "harness_git_commit": harness_commits.pop(),
            "component_result": {
                "path": COMPONENT_RESULT.as_posix(),
                "sha256": _sha256(component_path),
                "validation": (
                    "Every full-rho score exactly reproduces the matched "
                    "scalar-component campaign score."
                ),
            },
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
        default=STUDY / "artifacts" / "coarse_observation_probes",
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
