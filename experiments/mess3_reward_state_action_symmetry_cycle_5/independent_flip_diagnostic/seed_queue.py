"""Recover and analyze Cycle-5 Variant-2 checkpoints by seed."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from harness.storage.b2 import B2StorageConfig

MODULE = (
    "experiments.mess3_reward_state_action_symmetry_cycle_5."
    "independent_flip_diagnostic.experiment"
)
EXPERIMENT_DIR = Path(__file__).resolve().parent
SOURCE_RESULTS = EXPERIMENT_DIR.parent / "variant_2" / "results"
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


def _source_run(seed: int) -> tuple[Path, dict[str, object]]:
    source_dir = SOURCE_RESULTS / f"mess3-rsa-c5-v2-seed{seed}"
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


def _checkpoint_valid(checkpoint: Path) -> bool:
    return all(
        (checkpoint / relative).is_file()
        for relative in ESSENTIAL_CHECKPOINT_FILES
    )


def _storage_config() -> B2StorageConfig:
    bucket = os.environ.get("B2_BUCKET")
    endpoint = os.environ.get("B2_ENDPOINT")
    access_key_id = os.environ.get("B2_APPLICATION_KEY_ID")
    secret_access_key = os.environ.get("B2_APPLICATION_KEY")
    if not bucket or not endpoint or not access_key_id or not secret_access_key:
        raise RuntimeError(
            "B2_BUCKET, B2_ENDPOINT, B2_APPLICATION_KEY_ID, and "
            "B2_APPLICATION_KEY are required"
        )
    normalized_endpoint = endpoint.replace("\\", "/")
    if (
        normalized_endpoint.startswith("https:/")
        and not normalized_endpoint.startswith("https://")
    ):
        normalized_endpoint = (
            "https://" + normalized_endpoint.removeprefix("https:/")
        )
    return B2StorageConfig(
        bucket=bucket,
        endpoint=normalized_endpoint,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        prefix="",
    )


def _source_details(
    *,
    seed: int,
    source_dir: Path,
    checkpoint_name: str | None = None,
) -> tuple[str, str]:
    tune_summary = _load_json(source_dir / "tune_summary.json")
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
    prefix = (
        "experiments/mess3_reward_state_action_asymmetry_cycle_5/"
        f"variant_2/mess3-rsa-c5-v2-seed{seed}"
    )
    return selected_checkpoint_name, prefix


def _recover_checkpoint(
    seed: int,
    *,
    checkpoint_name: str | None = None,
) -> tuple[Path, dict[str, object]]:
    source_dir, manifest = _source_run(seed)
    checkpoint_name, prefix = _source_details(
        seed=seed,
        source_dir=source_dir,
        checkpoint_name=checkpoint_name,
    )
    bundle = (
        SOURCE_BUNDLES
        / f"seed_{seed}"
        / checkpoint_name
    )
    if _checkpoint_valid(bundle):
        return bundle, {
            "run_id": manifest["run_id"],
            "remote_prefix": prefix,
        }

    marker = f"/{checkpoint_name}/"
    storage = _storage_config()
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
    run_id = f"mess3-cycle5-variant2-independent-flip-seed{seed}"
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
    record["study"] = "cycle_5_variant_2_checkpoint_independent_token_flip"
    record["checkpoint"] = checkpoint.name
    record["source"] = {
        "cycle": 5,
        "run_id": source["run_id"],
        "remote_prefix": source["remote_prefix"],
    }
    result.write_text(json.dumps(record, indent=2) + "\n")
    return result


def _aggregate(
    results: list[Path],
    *,
    smoke: bool,
) -> Path:
    records = [_load_json(path) for path in results]
    summary = {
        "schema_version": 1,
        "study": "cycle_5_variant_2_checkpoint_independent_token_flip",
        "cycle": 5,
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
    parser.add_argument("--checkpoint-name")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    results = []
    for seed in arguments.seeds:
        checkpoint, source = _recover_checkpoint(
            seed,
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
            smoke=arguments.smoke,
        )
    )


if __name__ == "__main__":
    main()
