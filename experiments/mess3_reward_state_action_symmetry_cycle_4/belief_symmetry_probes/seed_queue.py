"""Recover training checkpoints and run arbitrary belief-probe jobs.

Example:
  python -m experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.seed_queue \
    v1:42 v2:43 v3:44
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any

from harness.cli import execute_experiment, load_experiment, make_run_context
from harness.storage.b2 import B2StorageConfig

HISTORICAL_STUDIES = {
    4: "mess3_reward_state_action_symmetry_cycle_4",
    5: "mess3_reward_state_action_asymmetry_cycle_5",
}
SOURCE_RUN_PATTERNS = {
    4: "mess3-rsa-c4-v{variant}-seed{seed}",
    5: "mess3-rsa-c5-v{variant}-seed{seed}",
}
CHECKPOINT_NAME = re.compile(r"checkpoint_\d+")
CAMPAIGN_SUFFIX = "0035"
TRAJECTORY_SUFFIX = "0036"
SEEDS = (42, 43, 44, 45, 46)
TARGET_VARIANTS = {
    "symmetric_b2": (1, 2, 3),
    "antisymmetric_b0_minus_b1": (1, 2, 3),
    # This campaign asks whether variant 2 learns the explicitly coarsened
    # two-state HMM filter. Variant 1 is lumpable too, but is not part of this
    # graph's preregistered comparison.
    "coarse_b2": (2,),
    "full_belief": (2,),
}


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)


def _final_checkpoint_name(tune_summary: dict[str, Any]) -> str:
    names = {
        match.group(0)
        for value in _walk_strings(tune_summary)
        for match in CHECKPOINT_NAME.finditer(value)
    }
    if not names:
        raise ValueError("tune_summary.json contains no checkpoint name")
    return max(names, key=lambda name: int(name.rsplit("_", 1)[1]))


def _objects(client: Any, bucket: str, prefix: str) -> list[str]:
    paginator = client.get_paginator("list_objects_v2")
    return [
        item["Key"]
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
        for item in page.get("Contents", [])
    ]


def _candidate_bases(
    *,
    configured_prefix: str,
    study: str,
    variant: int,
    source_run_id: str,
) -> list[str]:
    suffix = (
        f"experiments/{study}/variant_{variant}/{source_run_id}"
    )
    candidates = [
        "/".join(
            segment
            for segment in (configured_prefix.strip("/"), suffix)
            if segment
        ),
        suffix,
    ]
    return list(dict.fromkeys(candidates))


def _select_source_base(
    client: Any,
    bucket: str,
    bases: Sequence[str],
) -> tuple[str, str]:
    """Select the first candidate containing the compact Tune summary."""
    failures: list[Exception] = []
    for base in bases:
        tune_key = f"{base}/compact-results/tune_summary.json"
        try:
            client.head_object(Bucket=bucket, Key=tune_key)
        except Exception as error:
            failures.append(error)
            response = client.list_objects_v2(
                Bucket=bucket, Prefix=tune_key, MaxKeys=1
            )
            keys = {item["Key"] for item in response.get("Contents", [])}
            if tune_key not in keys:
                continue
        return base, tune_key
    raise FileNotFoundError(
        "tune_summary.json was not found at any historical B2 base"
    ) from (failures[-1] if failures else None)


def _download_atomic(
    client: Any,
    bucket: str,
    key: str,
    destination: Path,
    *,
    max_attempts: int = 5,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        temporary = Path(
            tempfile.mkstemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
            )[1]
        )
        try:
            response = client.get_object(Bucket=bucket, Key=key)
            body = response["Body"]
            try:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(body, output, length=8 * 1024 * 1024)
            finally:
                body.close()
            os.replace(temporary, destination)
            return
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt + 1 >= max_attempts:
                break
            delay = min(60.0, 4.0 * (2**attempt))
            print(
                f"[seed_queue] retrying B2 download for {key} "
                f"after {error!r} (attempt {attempt + 2}/{max_attempts})",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(f"B2 download failed for {key}") from last_error


def _validate_checkpoint(path: Path) -> None:
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    if not files:
        raise ValueError(f"empty recovered checkpoint: {path}")
    essential_names = {
        "algorithm_state.pkl",
        "rllib_checkpoint.json",
        "metadata.json",
        "class_and_ctor_args.pkl",
    }
    if not any(candidate.name in essential_names for candidate in files):
        raise ValueError(f"checkpoint essentials not found under {path}")


def recover_bundle(
    *,
    cycle: int,
    variant: int,
    seed: int,
    destination: Path,
    requested_target: str | None = None,
) -> Path:
    """Download the requested source checkpoint sequence from B2."""
    config = B2StorageConfig.from_env()
    if config is None:
        raise RuntimeError("B2 credentials are required to recover source checkpoints")
    client = config.s3_client()
    study = HISTORICAL_STUDIES[cycle]
    source_run_id = SOURCE_RUN_PATTERNS[cycle].format(variant=variant, seed=seed)
    base, tune_key = _select_source_base(
        client,
        config.bucket,
        _candidate_bases(
            configured_prefix=config.prefix,
            study=study,
            variant=variant,
            source_run_id=source_run_id,
        ),
    )
    tune_response = client.get_object(Bucket=config.bucket, Key=tune_key)
    tune_body = tune_response["Body"]
    try:
        tune_summary = json.loads(tune_body.read())
    finally:
        tune_body.close()
    final_name = _final_checkpoint_name(tune_summary)
    all_keys = _objects(client, config.bucket, f"{base}/")
    selected: list[tuple[str, str, str]] = []
    checkpoint_names = sorted(
        {
            part
            for key in all_keys
            for part in key.split("/")
            if CHECKPOINT_NAME.fullmatch(part)
        },
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    for key in all_keys:
        if "/initial_checkpoint/" in key:
            relative = key.split("/initial_checkpoint/", 1)[1]
            selected.append((key, "initial_checkpoint", relative))
        elif requested_target is None:
            if f"/{final_name}/" in key and "/compact-results/" not in key:
                relative = key.split(f"/{final_name}/", 1)[1]
                selected.append((key, "final_checkpoint", relative))
        else:
            for checkpoint_name in checkpoint_names:
                marker = f"/{checkpoint_name}/"
                if marker in key and "/compact-results/" not in key:
                    relative = key.split(marker, 1)[1]
                    selected.append(
                        (
                            key,
                            f"checkpoints/{checkpoint_name}",
                            relative,
                        )
                    )
                    break
    if not selected:
        raise FileNotFoundError(f"no checkpoint objects found under s3://{config.bucket}/{base}")
    if requested_target is not None and not checkpoint_names:
        raise FileNotFoundError(f"no training checkpoints found under s3://{config.bucket}/{base}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        for key, member, relative in selected:
            _download_atomic(client, config.bucket, key, staging / member / relative)
        _validate_checkpoint(staging / "initial_checkpoint")
        if requested_target is None:
            _validate_checkpoint(staging / "final_checkpoint")
        else:
            for checkpoint_name in checkpoint_names:
                _validate_checkpoint(staging / "checkpoints" / checkpoint_name)
            (staging / "checkpoint_manifest.json").write_text(
                json.dumps(
                    {
                        "checkpoints": [
                            {
                                "label": "initial",
                                "training_iteration": 0,
                                "path": "initial_checkpoint",
                            },
                            *[
                                {
                                    "label": checkpoint_name,
                                    # RLlib's zero-based checkpoint suffix is
                                    # one behind training_iteration.
                                    "training_iteration": (
                                        int(checkpoint_name.rsplit("_", 1)[1]) + 1
                                    ),
                                    "path": f"checkpoints/{checkpoint_name}",
                                }
                                for checkpoint_name in checkpoint_names
                            ],
                        ]
                    },
                    indent=2,
                )
                + "\n"
            )
        (staging / "source_provenance.json").write_text(
            json.dumps(
                {
                    "source_run_id": source_run_id,
                    "historical_study": study,
                    "cycle": cycle,
                    "variant": variant,
                    "seed": seed,
                    "b2_base": base,
                    "final_checkpoint_name": final_name,
                    "requested_target": requested_target,
                    "checkpoint_count": (
                        len(checkpoint_names) + 1
                        if requested_target is not None
                        else 2
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _instance_id() -> str | None:
    path = Path("/root/vast_instance_id")
    return path.read_text().strip() if path.is_file() else os.environ.get("VAST_INSTANCE_ID")


def _push_results(run_name: str) -> bool:
    from devops.vast.self_destruct import push_results

    return bool(
        push_results(
            run_name=run_name,
            instance_id=_instance_id(),
        )
    )


def _parse_job(job: str) -> tuple[int, int]:
    match = re.fullmatch(r"v([123]):(\d+)", job)
    if match is None:
        raise argparse.ArgumentTypeError("jobs must have form v1:42")
    return int(match.group(1)), int(match.group(2))


def run_job(
    *,
    cycle: int,
    variant: int,
    seed: int,
    target: str | None,
    push_each: bool,
) -> int:
    package = f"mess3_reward_state_action_symmetry_cycle_{cycle}"
    module = f"experiments.{package}.belief_symmetry_probes.variant_{variant}.experiment"
    if target is None:
        run_id = (
            f"mess3-rsa-c{cycle}-belief-symmetry-probe-{CAMPAIGN_SUFFIX}-"
            f"v{variant}-seed{seed}"
        )
    else:
        target_slug = target.replace("_", "-")
        run_id = (
            f"mess3-rsa-c{cycle}-belief-trajectory-{TRAJECTORY_SUFFIX}-"
            f"{target_slug}-v{variant}-seed{seed}"
        )
    experiment = load_experiment(module)
    context = make_run_context(
        experiment,
        seed=seed,
        run_id=run_id,
        hardware_profile="cuda4090",
    )
    bundle = context.artifacts_dir / "source_checkpoint_bundle"
    try:
        recover_bundle(
            cycle=cycle,
            variant=variant,
            seed=seed,
            destination=bundle,
            requested_target=target,
        )
        context = make_run_context(
            experiment,
            seed=seed,
            run_id=run_id,
            resume_from=bundle,
            results_dir=context.results_dir,
            artifacts_dir=context.artifacts_dir,
            hardware_profile="cuda4090",
        )
        started = time.time()
        execute_experiment(
            experiment,
            context,
            command=["seed_queue", f"v{variant}:{seed}"],
            runtime_overrides={
                "hardware_profile": "cuda4090",
                "seed": seed,
                "resume_from": str(bundle),
                "probe_target": target,
                "upload_artifacts": False,
            },
            upload_artifacts=False,
        )
        print(f"[seed_queue] completed {run_id} in {time.time() - started:.1f}s", flush=True)
    except Exception as error:  # keep independent assignments running
        print(f"[seed_queue] FAILED {run_id}: {error}", flush=True)
        if push_each:
            _push_results(f"{run_id}-failed")
        return 1
    finally:
        shutil.rmtree(bundle, ignore_errors=True)
    if push_each and not _push_results(run_id):
        return 2
    return 0


def main(argv: Sequence[str] | None = None, *, cycle: int = 4) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", nargs="*", type=_parse_job)
    parser.add_argument("--target", choices=tuple(TARGET_VARIANTS))
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Run the target's complete variant x seeds 42-46 graph campaign.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--push-each", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)
    if args.all_seeds:
        if args.target is None:
            parser.error("--all-seeds requires --target")
        if args.jobs:
            parser.error("explicit jobs cannot be combined with --all-seeds")
        jobs = [
            (variant, seed)
            for variant in TARGET_VARIANTS[args.target]
            for seed in SEEDS
        ]
    else:
        jobs = list(args.jobs)
    if not jobs:
        parser.error("provide jobs or use --target TARGET --all-seeds")
    if args.target == "coarse_b2" and any(variant != 2 for variant, _ in jobs):
        parser.error("the coarse_b2 graph campaign is restricted to variant 2")
    codes: dict[str, int] = {}
    for variant, seed in jobs:
        label = f"v{variant}:{seed}"
        codes[label] = run_job(
            cycle=cycle,
            variant=variant,
            seed=seed,
            target=args.target,
            push_each=args.push_each,
        )
        if codes[label] and args.fail_fast:
            break
    if args.target is not None and not any(codes.values()):
        from experiments.mess3_reward_state_action_symmetry_cycle_4.belief_symmetry_probes.trajectory_campaign import (
            write_campaign,
        )

        package = Path(__file__).resolve().parents[1].name.replace(
            "cycle_4", f"cycle_{cycle}"
        )
        root = (
            Path(__file__).resolve().parents[1].parent
            / package
            / "belief_symmetry_probes"
        )
        output = write_campaign(root, cycle=cycle, target=args.target)
        print(f"[seed_queue] wrote campaign graph to {output}", flush=True)
        if args.push_each and not _push_results(
            f"mess3-rsa-c{cycle}-belief-trajectory-{TRAJECTORY_SUFFIX}-{args.target}"
        ):
            return 2
    print({"job_exit_codes": codes}, flush=True)
    return 1 if any(codes.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
