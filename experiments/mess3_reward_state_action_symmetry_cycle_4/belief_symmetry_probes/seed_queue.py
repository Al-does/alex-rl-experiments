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
            # B2 often rejects HeadObject while ListObjectsV2 still works.
            response = client.list_objects_v2(
                Bucket=bucket, Prefix=tune_key, MaxKeys=1
            )
            if response.get("Contents"):
                return base, tune_key
        except Exception as error:  # provider clients use generated exception types
            failures.append(error)
            continue
    raise FileNotFoundError(
        "tune_summary.json was not found at any historical B2 base"
    ) from (failures[-1] if failures else None)


def _download_atomic(client: Any, bucket: str, key: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        client.download_file(bucket, key, str(temporary))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


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
) -> Path:
    """Download only initial and actual final checkpoints from B2."""
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
    with tempfile.NamedTemporaryFile() as handle:
        client.download_file(config.bucket, tune_key, handle.name)
        tune_summary = json.loads(Path(handle.name).read_text())
    final_name = _final_checkpoint_name(tune_summary)
    all_keys = _objects(client, config.bucket, f"{base}/")
    selected: list[tuple[str, str, str]] = []
    for key in all_keys:
        if "/initial_checkpoint/" in key:
            relative = key.split("/initial_checkpoint/", 1)[1]
            selected.append((key, "initial_checkpoint", relative))
        elif f"/{final_name}/" in key and "/compact-results/" not in key:
            relative = key.split(f"/{final_name}/", 1)[1]
            selected.append((key, "final_checkpoint", relative))
    if not selected:
        raise FileNotFoundError(f"no checkpoint objects found under s3://{config.bucket}/{base}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        for key, member, relative in selected:
            _download_atomic(client, config.bucket, key, staging / member / relative)
        _validate_checkpoint(staging / "initial_checkpoint")
        _validate_checkpoint(staging / "final_checkpoint")
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
            branch=os.environ.get(
                "VAST_RESULTS_BRANCH", "results-belief-symmetry-0035"
            ),
            run_name=run_name,
            instance_id=_instance_id(),
        )
    )


def _parse_job(job: str) -> tuple[int, int]:
    match = re.fullmatch(r"v([123]):(\d+)", job)
    if match is None:
        raise argparse.ArgumentTypeError("jobs must have form v1:42")
    return int(match.group(1)), int(match.group(2))


def run_job(*, cycle: int, variant: int, seed: int, push_each: bool) -> int:
    package = f"mess3_reward_state_action_symmetry_cycle_{cycle}"
    module = f"experiments.{package}.belief_symmetry_probes.variant_{variant}.experiment"
    run_id = (
        f"mess3-rsa-c{cycle}-belief-symmetry-probe-{CAMPAIGN_SUFFIX}-"
        f"v{variant}-seed{seed}"
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
    if push_each and not _push_results(run_id):
        return 2
    return 0


def main(argv: Sequence[str] | None = None, *, cycle: int = 4) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", nargs="+", type=_parse_job)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--push-each", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)
    codes: dict[str, int] = {}
    for variant, seed in args.jobs:
        label = f"v{variant}:{seed}"
        codes[label] = run_job(
            cycle=cycle, variant=variant, seed=seed, push_each=args.push_each
        )
        if codes[label] and args.fail_fast:
            break
    print({"job_exit_codes": codes}, flush=True)
    return 1 if any(codes.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
