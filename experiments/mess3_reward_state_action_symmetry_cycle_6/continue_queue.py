"""Resume a completed Cycle-6 battery variant and continue REINFORCE training.

Downloads the variant's final Tune checkpoint for a completed seed run from
B2 (located via the committed battery results), then executes the variant
leaf with ``resume_from`` set so training continues to the continuation
budget.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path

from experiments.mess3_reward_state_action_symmetry_cycle_6.shared import (
    write_budget_spec,
)
from harness.cli import execute_experiment, load_experiment, make_run_context
from harness.storage.b2 import B2StorageConfig

STUDY = "mess3_reward_state_action_symmetry_cycle_6"
STUDY_DIR = Path(__file__).resolve().parent
BATTERY_RESULTS = STUDY_DIR / "battery" / "results"
ESSENTIAL_CHECKPOINT_FILES = (
    Path("algorithm_state.pkl"),
    Path("class_and_ctor_args.pkl"),
    Path(
        "learner_group/learner/rl_module/default_policy/"
        "class_and_ctor_args.pkl"
    ),
    Path("learner_group/learner/rl_module/default_policy/module_state.pkl"),
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _instance_id() -> str | None:
    path = Path("/root/vast_instance_id")
    if path.is_file():
        return path.read_text().strip() or None
    return os.environ.get("VAST_INSTANCE_ID")


def _push_results(run_name: str) -> bool:
    try:
        from devops.vast.self_destruct import push_results
    except ImportError as error:
        print(f"[continue_queue] push unavailable: {error}", flush=True)
        return False
    return bool(
        push_results(
            branch=os.environ.get("VAST_RESULTS_BRANCH", "results"),
            run_name=run_name,
            instance_id=_instance_id(),
        )
    )


def _source_run(seed: int, variant: int) -> tuple[Path, dict]:
    """Latest completed battery result dir that trained ``variant`` on ``seed``."""
    candidates = []
    for manifest_path in BATTERY_RESULTS.glob("*/run_manifest.json"):
        manifest = _load_json(manifest_path)
        if (
            manifest.get("status") == "completed"
            and int(manifest["runtime"]["seed"]) == seed
            and (
                manifest_path.parent / f"variant_{variant}/tune_summary.json"
            ).is_file()
        ):
            candidates.append((manifest_path.parent, manifest))
    if not candidates:
        raise FileNotFoundError(
            f"no completed Cycle-6 battery result for seed {seed} "
            f"variant {variant}"
        )
    candidates.sort(key=lambda pair: pair[1].get("started_at", ""))
    return candidates[-1]


def _final_checkpoint_name(source_dir: Path, variant: int) -> str:
    tune_summary = _load_json(
        source_dir / f"variant_{variant}/tune_summary.json"
    )
    trials = tune_summary.get("trials", [])
    if len(trials) != 1:
        raise ValueError(
            f"expected exactly one Variant-{variant} Tune trial"
        )
    return Path(trials[0]["checkpoint"]).name


def _checkpoint_valid(checkpoint: Path) -> bool:
    return all(
        (checkpoint / relative).is_file()
        for relative in ESSENTIAL_CHECKPOINT_FILES
    )


def _storage_config(remote: dict) -> B2StorageConfig:
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


def _recover_checkpoint(
    seed: int,
    variant: int,
    *,
    checkpoint_name: str | None = None,
) -> tuple[Path, dict]:
    """Download the variant checkpoint for ``seed`` from B2 (cached on disk)."""
    source_dir, manifest = _source_run(seed, variant)
    checkpoint_name = checkpoint_name or _final_checkpoint_name(
        source_dir, variant
    )
    if (
        Path(checkpoint_name).name != checkpoint_name
        or not checkpoint_name.startswith("checkpoint_")
    ):
        raise ValueError(f"invalid checkpoint name: {checkpoint_name!r}")
    remote = manifest["remote_artifacts"]
    prefix = f"{str(remote['prefix']).rstrip('/')}/variant_{variant}"
    bundle = (
        STUDY_DIR
        / f"variant_{variant}"
        / "artifacts"
        / "source_bundles"
        / f"seed_{seed}"
        / checkpoint_name
    )
    if _checkpoint_valid(bundle):
        return bundle, {"run_id": manifest["run_id"], "remote_prefix": prefix}

    marker = f"/{checkpoint_name}/"
    storage = _storage_config(remote)
    client = storage.s3_client()
    objects = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=storage.bucket,
        Prefix=f"{prefix}/",
    ):
        objects.extend(
            item
            for item in page.get("Contents", [])
            if marker in str(item["Key"])
        )
    if not objects:
        raise FileNotFoundError(
            f"remote checkpoint {checkpoint_name!r} not found under {prefix}"
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
        "variant": variant,
        "source_run_id": manifest["run_id"],
        "source_prefix": prefix,
        "checkpoint_name": checkpoint_name,
        "checkpoint_files": len(objects),
    }
    (bundle.parent / "source.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return bundle, {"run_id": manifest["run_id"], "remote_prefix": prefix}


def _run_one(
    *,
    variant: int,
    seed: int,
    target_agent_steps: int,
    hardware_profile: str,
    smoke: bool,
    publish_smoke: bool,
    upload_artifacts: bool,
    push_each: bool,
    checkpoint_name: str | None,
) -> int:
    bundle, source = _recover_checkpoint(
        seed, variant, checkpoint_name=checkpoint_name
    )
    run_id = (
        f"{STUDY}-variant{variant}-continued-seed{seed}"
        f"-{target_agent_steps // 1_000_000}m"
    )
    if smoke:
        run_id += "-smoke"
    experiment = load_experiment(
        f"experiments.{STUDY}.variant_{variant}.experiment"
    )
    context = make_run_context(
        experiment,
        seed=seed,
        run_id=run_id,
        smoke=smoke,
        publish_smoke=publish_smoke,
        resume_from=bundle,
        hardware_profile=hardware_profile,
    )
    write_budget_spec(context, target_agent_steps)
    print(
        f"[continue_queue] start variant={variant} seed={seed} "
        f"target_steps={target_agent_steps} run_id={run_id} "
        f"resume_from={bundle} (source run {source['run_id']})",
        flush=True,
    )
    started = time.time()
    try:
        execute_experiment(
            experiment,
            context,
            command=[
                "python",
                "-m",
                f"experiments.{STUDY}.continue_queue",
                "--variant",
                str(variant),
                "--seed",
                str(seed),
                "--target-agent-steps",
                str(target_agent_steps),
            ],
            runtime_overrides={
                "seed": seed,
                "variant": variant,
                "target_agent_steps": target_agent_steps,
                "hardware_profile": hardware_profile,
                "resume_from": str(bundle),
                "source_run_id": source["run_id"],
                "source_remote_prefix": source["remote_prefix"],
                "upload_artifacts": upload_artifacts,
            },
            upload_artifacts=upload_artifacts,
        )
    except Exception as error:  # noqa: BLE001 - report then exit nonzero
        print(f"[continue_queue] FAILED {run_id}: {error}", flush=True)
        if push_each:
            _push_results(f"{run_id}-failed")
        return 1

    print(
        f"[continue_queue] done {run_id} "
        f"elapsed_s={time.time() - started:.1f}",
        flush=True,
    )
    if push_each and not _push_results(run_id):
        print(f"[continue_queue] push failed for {run_id}", flush=True)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-agent-steps", type=int, default=300_000_000)
    parser.add_argument("--hardware-profile", default="cuda4090")
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--publish-smoke", action="store_true")
    parser.add_argument(
        "--upload-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--push-each",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.publish_smoke and not args.smoke:
        raise SystemExit("--publish-smoke requires --smoke")
    return _run_one(
        variant=args.variant,
        seed=args.seed,
        target_agent_steps=args.target_agent_steps,
        hardware_profile=args.hardware_profile,
        smoke=args.smoke,
        publish_smoke=args.publish_smoke,
        upload_artifacts=args.upload_artifacts
        and (not args.smoke or args.publish_smoke),
        push_each=args.push_each and not args.smoke,
        checkpoint_name=args.checkpoint_name,
    )


if __name__ == "__main__":
    raise SystemExit(main())
