"""Recover Tune training state from B2 for Cassandra best-critic continuations."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from harness.storage.b2 import B2StorageConfig

SOURCE_LEAF = (
    "experiments/cassandra_belief_factoring_2026_08/"
    "best_critic_bptt64_250m/targeted"
)
SOURCE_RUNS: dict[int, str] = {
    42: "20260825T062526Z-0452c263",
    43: "20260825T070342Z-fa83b069",
}
REQUIRED_ARTIFACT_PREFIXES = ("tune/", "metrics.jsonl")


def source_run_id(seed: int) -> str:
    """Return the completed 250M targeted run id for one seed."""

    try:
        return SOURCE_RUNS[seed]
    except KeyError as error:
        raise ValueError(
            f"no completed targeted 250M source run registered for seed {seed}"
        ) from error


def _candidate_bases(*, configured_prefix: str, source_run_id: str) -> list[str]:
    suffix = f"{SOURCE_LEAF}/{source_run_id}"
    candidates = [
        "/".join(
            segment
            for segment in (configured_prefix.strip("/"), suffix)
            if segment
        ),
        suffix,
    ]
    return list(dict.fromkeys(candidates))


def _list_objects(client: Any, bucket: str, prefix: str) -> list[str]:
    paginator = client.get_paginator("list_objects_v2")
    return [
        item["Key"]
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
        for item in page.get("Contents", [])
    ]


def _download_atomic(
    client: Any,
    *,
    bucket: str,
    key: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        try:
            with temporary.open("wb") as output:
                shutil.copyfileobj(body, output, length=8 * 1024 * 1024)
        finally:
            body.close()
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _select_source_base(
    client: Any,
    bucket: str,
    bases: list[str],
) -> tuple[str, list[str]]:
    failures: list[Exception] = []
    for base in bases:
        keys = _list_objects(client, bucket, f"{base}/")
        tune_keys = [key for key in keys if "/tune/" in f"/{key}/"]
        if tune_keys:
            return base, keys
        failures.append(FileNotFoundError(f"no tune objects under s3://{bucket}/{base}/"))
    raise FileNotFoundError(
        "Tune artifacts were not found at any candidate B2 base"
    ) from (failures[-1] if failures else None)


def _validate_recovered_tree(artifacts_dir: Path) -> Path:
    tune_dir = artifacts_dir / "tune"
    experiment_states = list(tune_dir.glob("experiment_state-*.json"))
    if not experiment_states:
        raise FileNotFoundError(
            f"recovered Tune tree is missing experiment_state under {tune_dir}"
        )
    checkpoints = list(tune_dir.rglob("checkpoint_*"))
    if not checkpoints:
        raise FileNotFoundError(
            f"recovered Tune tree is missing checkpoints under {tune_dir}"
        )
    return tune_dir


def recover_tune_artifacts(
    *,
    destination: Path,
    source_run_id: str,
) -> Path:
    """Download Tune state for one completed targeted run into ``destination``."""

    config = B2StorageConfig.from_env()
    if config is None:
        raise RuntimeError(
            "B2 credentials are required to recover source Tune checkpoints"
        )
    client = config.s3_client()
    base, keys = _select_source_base(
        client,
        config.bucket,
        _candidate_bases(
            configured_prefix=config.prefix,
            source_run_id=source_run_id,
        ),
    )
    selected: list[tuple[str, str]] = []
    base_prefix = f"{base}/"
    for key in keys:
        if not key.startswith(base_prefix):
            continue
        relative = key[len(base_prefix) :]
        if relative.startswith("tune/") or relative == "metrics.jsonl":
            selected.append((key, relative))
    if not selected:
        raise FileNotFoundError(
            f"no recoverable Tune objects found under s3://{config.bucket}/{base}/"
        )

    destination.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        for key, relative in selected:
            _download_atomic(
                client,
                bucket=config.bucket,
                key=key,
                destination=staging / relative,
            )
        tune_dir = _validate_recovered_tree(staging)
        provenance = {
            "source_run_id": source_run_id,
            "source_leaf": SOURCE_LEAF,
            "b2_base": base,
            "object_count": len(selected),
            "tune_dir": str(tune_dir),
        }
        (staging / "source_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "tune"


__all__ = [
    "SOURCE_LEAF",
    "SOURCE_RUNS",
    "recover_tune_artifacts",
    "source_run_id",
]
