"""Recover and analyze final Variant-2 checkpoints by seed."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from harness.storage.b2 import B2StorageConfig

MODULE = (
    "experiments.mess3_reward_state_action_symmetry_cycle_6."
    "independent_flip_diagnostic.experiment"
)
EXPERIMENT_DIR = Path(__file__).resolve().parent
BATTERY_RESULTS = EXPERIMENT_DIR.parent / "battery" / "results"
CYCLE_5_RESULTS = (
    EXPERIMENT_DIR.parents[1]
    / "mess3_reward_state_action_symmetry_cycle_5"
    / "variant_2"
    / "results"
)
SOURCE_BUNDLES = EXPERIMENT_DIR / "artifacts" / "source_bundles"
SEED_QUEUE_RESULTS = EXPERIMENT_DIR / "results"
ESSENTIAL_CHECKPOINT_FILES = (
    Path("algorithm_state.pkl"),
    Path("class_and_ctor_args.pkl"),
    Path(
        "learner_group/learner/rl_module/default_policy/"
        "class_and_ctor_args.pkl"
    ),
    Path("learner_group/learner/rl_module/default_policy/module_state.pkl"),
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _source_run(
    seed: int,
    *,
    source_cycle: int = 6,
) -> tuple[Path, dict[str, object]]:
    if source_cycle == 5:
        source_dir = CYCLE_5_RESULTS / f"mess3-rsa-c5-v2-seed{seed}"
        manifest_path = source_dir / "run_manifest.json"
        if (
            not manifest_path.is_file()
            or not (source_dir / "tune_summary.json").is_file()
        ):
            raise FileNotFoundError(
                f"no completed Cycle-5 Variant-2 result found for seed {seed}"
            )
        manifest = _load_json(manifest_path)
        if manifest.get("status") != "completed":
            raise ValueError(f"Cycle-5 seed {seed} did not complete")
        return source_dir, manifest
    if source_cycle != 6:
        raise ValueError(f"unsupported source cycle: {source_cycle}")

    candidates = []
    for manifest_path in BATTERY_RESULTS.glob("*/run_manifest.json"):
        manifest = _load_json(manifest_path)
        if (
            manifest.get("status") == "completed"
            and int(manifest["runtime"]["seed"]) == seed
            and (manifest_path.parent / "variant_2/tune_summary.json").is_file()
        ):
            candidates.append((manifest_path.parent, manifest))
    if not candidates:
        raise FileNotFoundError(
            f"no completed Cycle-6 battery result found for seed {seed}"
        )
    candidates.sort(key=lambda pair: pair[1].get("started_at", ""))
    return candidates[-1]


def _checkpoint_valid(checkpoint: Path) -> bool:
    return all(
        (checkpoint / relative).is_file()
        for relative in ESSENTIAL_CHECKPOINT_FILES
    )


def _storage_config(remote: dict[str, object]) -> B2StorageConfig:
    access_key_id = os.environ.get("B2_APPLICATION_KEY_ID")
    secret_access_key = os.environ.get("B2_APPLICATION_KEY")
    if not access_key_id or not secret_access_key:
        raise RuntimeError(
            "B2_APPLICATION_KEY_ID and B2_APPLICATION_KEY are required"
        )
    return B2StorageConfig(
        bucket=str(remote["bucket"]),
        endpoint=str(remote["endpoint"]),
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        prefix="",
    )


def _cycle_5_remote() -> dict[str, object]:
    _, cycle_6_manifest = _source_run(42)
    cycle_6_remote = cycle_6_manifest["remote_artifacts"]
    return {
        "bucket": cycle_6_remote["bucket"],
        "endpoint": cycle_6_remote["endpoint"],
    }


def _source_details(
    *,
    source_cycle: int,
    seed: int,
    source_dir: Path,
    manifest: dict[str, object],
    checkpoint_name: str | None = None,
) -> tuple[str, str, dict[str, object]]:
    tune_relative = (
        "variant_2/tune_summary.json"
        if source_cycle == 6
        else "tune_summary.json"
    )
    tune_summary = _load_json(source_dir / tune_relative)
    trials = tune_summary.get("trials", [])
    if len(trials) != 1:
        raise ValueError("expected exactly one Variant-2 Tune trial")
    final_checkpoint_name = Path(trials[0]["checkpoint"]).name
    selected_checkpoint_name = checkpoint_name or final_checkpoint_name
    if (
        Path(selected_checkpoint_name).name != selected_checkpoint_name
        or not selected_checkpoint_name.startswith("checkpoint_")
    ):
        raise ValueError(
            f"invalid checkpoint name: {selected_checkpoint_name!r}"
        )
    if source_cycle == 6:
        remote = manifest["remote_artifacts"]
        prefix = f"{str(remote['prefix']).rstrip('/')}/variant_2"
    else:
        remote = _cycle_5_remote()
        prefix = (
            "experiments/mess3_reward_state_action_asymmetry_cycle_5/"
            f"variant_2/mess3-rsa-c5-v2-seed{seed}"
        )
    return selected_checkpoint_name, prefix, remote


def _recover_checkpoint(
    seed: int,
    *,
    source_cycle: int = 6,
    checkpoint_name: str | None = None,
) -> tuple[Path, dict[str, object]]:
    source_dir, manifest = _source_run(seed, source_cycle=source_cycle)
    checkpoint_name, prefix, remote = _source_details(
        source_cycle=source_cycle,
        seed=seed,
        source_dir=source_dir,
        manifest=manifest,
        checkpoint_name=checkpoint_name,
    )
    bundle = (
        SOURCE_BUNDLES
        / f"cycle_{source_cycle}"
        / f"seed_{seed}"
        / checkpoint_name
    )
    if _checkpoint_valid(bundle):
        return bundle, {
            "cycle": source_cycle,
            "run_id": manifest["run_id"],
            "remote_prefix": prefix,
        }

    marker = f"/{checkpoint_name}/"
    storage = _storage_config(remote)
    client = storage.s3_client()
    objects = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=storage.bucket,
        Prefix=f"{prefix.rstrip('/')}/",
    ):
        objects.extend(
            item
            for item in page.get("Contents", [])
            if marker in str(item["Key"])
        )
    if not objects:
        raise FileNotFoundError(
            f"remote final checkpoint {checkpoint_name!r} was not found"
        )
    for item in objects:
        key = str(item["Key"])
        relative = Path(key.split(marker, 1)[1])
        destination = bundle / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(storage.bucket, key, str(destination))
    if not _checkpoint_valid(bundle):
        raise RuntimeError(f"recovered checkpoint is incomplete: {bundle}")
    provenance = {
        "source_cycle": source_cycle,
        "seed": seed,
        "source_run_id": manifest["run_id"],
        "source_prefix": prefix,
        "checkpoint_name": checkpoint_name,
        "checkpoint_files": len(objects),
    }
    (bundle.parent / "source.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return bundle, {
        "cycle": source_cycle,
        "run_id": manifest["run_id"],
        "remote_prefix": prefix,
    }


def _run_diagnostic(
    *,
    seed: int,
    checkpoint: Path,
    source: dict[str, object],
    smoke: bool,
    checkpoint_name_override: bool = False,
) -> Path:
    source_cycle = int(source["cycle"])
    run_id = (
        f"mess3-cycle{source_cycle}-variant2-independent-flip-seed{seed}"
    )
    if checkpoint_name_override:
        run_id += f"-{checkpoint.name.replace('_', '')}"
    if smoke:
        run_id += "-smoke"
    command = [
        sys.executable,
        "-m",
        "harness.cli",
        MODULE,
        "--run-id",
        run_id,
        "--seed",
        str(seed),
        "--resume-from",
        str(checkpoint),
        "--hardware-profile",
        "cpu",
    ]
    if smoke:
        command.append("--smoke")
    subprocess.run(command, check=True)
    result = (
        EXPERIMENT_DIR
        / ".smoke"
        / run_id
        / "results"
        / "independent_flip_diagnostic.json"
        if smoke
        else SEED_QUEUE_RESULTS / run_id / "independent_flip_diagnostic.json"
    )
    if not result.is_file():
        raise FileNotFoundError(f"diagnostic result was not written: {result}")
    record = _load_json(result)
    record["implementation_cycle"] = record["cycle"]
    record["cycle"] = source_cycle
    record["study"] = (
        f"cycle_{source_cycle}_variant_2_checkpoint_"
        "independent_token_flip"
    )
    record["source_cycle"] = source_cycle
    record["checkpoint"] = checkpoint.name
    record["source"] = {
        "cycle": source_cycle,
        "run_id": source["run_id"],
        "remote_prefix": source["remote_prefix"],
    }
    result.write_text(json.dumps(record, indent=2) + "\n")
    return result


def _aggregate(
    results: list[Path],
    *,
    source_cycle: int,
    smoke: bool,
) -> Path:
    records = [_load_json(path) for path in results]
    summary = {
        "schema_version": 1,
        "study": (
            f"cycle_{source_cycle}_variant_2_checkpoint_"
            "independent_token_flip"
        ),
        "source_cycle": source_cycle,
        "smoke": smoke,
        "seeds": [record["seed"] for record in records],
        "metrics": {
            "factual_fine_mse": [
                record["paired_fixed_action_replay"]["factual"]["fine_probe_s"][
                    "mse"
                ]
                for record in records
            ],
            "randomized_fine_mse": [
                record["paired_fixed_action_replay"]["randomized"]["fine_probe_s"][
                    "mse"
                ]
                for record in records
            ],
            "factual_coarse_mse": [
                record["paired_fixed_action_replay"]["factual"]["coarse_probe_c"][
                    "mse"
                ]
                for record in records
            ],
            "randomized_coarse_mse": [
                record["paired_fixed_action_replay"]["randomized"][
                    "coarse_probe_c"
                ]["mse"]
                for record in records
            ],
            "fine_counterfactual_over_factual_mse": [
                record["paired_fixed_action_replay"][
                    "fine_probe_counterfactual_over_factual_mse"
                ]
                for record in records
            ],
            "coarse_counterfactual_over_factual_mse": [
                record["paired_fixed_action_replay"][
                    "coarse_probe_counterfactual_over_factual_mse"
                ]
                for record in records
            ],
            "decoded_s_shift_r_squared": [
                record["paired_fixed_action_replay"]["randomized"][
                    "decoded_s_shift_r_squared"
                ]
                for record in records
            ],
            "exact_s_shift_rms": [
                record["paired_fixed_action_replay"]["randomized"][
                    "exact_s_shift_rms"
                ]
                for record in records
            ],
            "decoded_s_shift_rms": [
                record["paired_fixed_action_replay"]["randomized"][
                    "decoded_s_shift_rms"
                ]
                for record in records
            ],
            "decoded_coarse_invariance_rmse": [
                record["paired_fixed_action_replay"]["randomized"][
                    "decoded_coarse_invariance_rmse"
                ]
                for record in records
            ],
            "activation_shift_rms": [
                record["paired_fixed_action_replay"]["randomized"][
                    "activation_shift_rms"
                ]
                for record in records
            ],
            "policy_probability_total_variation_mean": [
                record["paired_fixed_action_replay"]["randomized"]["policy"][
                    "probability_total_variation_mean"
                ]
                for record in records
            ],
            "policy_greedy_action_agreement": [
                record["paired_fixed_action_replay"]["randomized"]["policy"][
                    "greedy_action_agreement"
                ]
                for record in records
            ],
            "closed_loop_reward_delta": [
                record["closed_loop_policy"]["randomized_reward_delta_mean"]
                for record in records
            ],
        },
        "result_files": [
            str(path.relative_to(EXPERIMENT_DIR)) for path in results
        ],
    }
    for name, values in summary["metrics"].items():
        summary["metrics"][name] = {
            "by_seed": dict(zip(map(str, summary["seeds"]), values)),
            "mean": float(sum(values) / len(values)),
        }
    output = (
        EXPERIMENT_DIR / ".smoke" / "independent_flip_summary.json"
        if smoke
        else SEED_QUEUE_RESULTS / "independent_flip_summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--source-cycle", type=int, choices=(5, 6), default=6)
    parser.add_argument("--checkpoint-name")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    results = []
    for seed in arguments.seeds:
        checkpoint, source = _recover_checkpoint(
            seed,
            source_cycle=arguments.source_cycle,
            checkpoint_name=arguments.checkpoint_name,
        )
        results.append(
            _run_diagnostic(
                seed=seed,
                checkpoint=checkpoint,
                source=source,
                smoke=arguments.smoke,
                checkpoint_name_override=arguments.checkpoint_name is not None,
            )
        )
    print(
        _aggregate(
            results,
            source_cycle=arguments.source_cycle,
            smoke=arguments.smoke,
        )
    )


if __name__ == "__main__":
    main()
