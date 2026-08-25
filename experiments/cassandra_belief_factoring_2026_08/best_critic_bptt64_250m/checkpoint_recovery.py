"""Recover Algorithm checkpoints from B2 for Cassandra best-critic continuations."""

from __future__ import annotations

import json
import os
import re
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
CHECKPOINT_NAME = re.compile(r"checkpoint_\d+")


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


def _select_source_base(
    client: Any,
    bucket: str,
    bases: list[str],
) -> tuple[str, dict[str, Any]]:
    failures: list[Exception] = []
    for base in bases:
        tune_key = f"{base}/compact-results/tune_summary.json"
        try:
            response = client.get_object(Bucket=bucket, Key=tune_key)
            body = response["Body"]
            try:
                tune_summary = json.loads(body.read())
            finally:
                body.close()
            return base, tune_summary
        except Exception as error:
            failures.append(error)
    raise FileNotFoundError(
        "tune_summary.json was not found at any candidate B2 base"
    ) from (failures[-1] if failures else None)


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


def recover_source_checkpoint(
    *,
    destination: Path,
    source_run_id: str,
) -> Path:
    """Download the final Algorithm checkpoint for one completed targeted run."""

    config = B2StorageConfig.from_env()
    if config is None:
        raise RuntimeError(
            "B2 credentials are required to recover source checkpoints"
        )
    client = config.s3_client()
    base, tune_summary = _select_source_base(
        client,
        config.bucket,
        _candidate_bases(
            configured_prefix=config.prefix,
            source_run_id=source_run_id,
        ),
    )
    final_name = _final_checkpoint_name(tune_summary)
    all_keys = _list_objects(client, config.bucket, f"{base}/")
    selected: list[tuple[str, str]] = []
    marker = f"/{final_name}/"
    for key in all_keys:
        if marker in key and "/compact-results/" not in key:
            relative = key.split(marker, 1)[1]
            selected.append((key, relative))
    if not selected:
        raise FileNotFoundError(
            f"no checkpoint objects found for {final_name} under "
            f"s3://{config.bucket}/{base}/"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
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
        _validate_checkpoint(staging)
        (staging / "source_provenance.json").write_text(
            json.dumps(
                {
                    "source_run_id": source_run_id,
                    "source_leaf": SOURCE_LEAF,
                    "b2_base": base,
                    "final_checkpoint_name": final_name,
                },
                indent=2,
                sort_keys=True,
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


__all__ = [
    "SOURCE_LEAF",
    "SOURCE_RUNS",
    "recover_source_checkpoint",
    "source_run_id",
]
