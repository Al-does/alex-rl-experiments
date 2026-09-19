"""Incremental B2 artifact upload for checkpoint durability during long runs.

The harness's ``upload_run_artifacts`` ships ``artifacts/`` only at run end,
which leaves multi-hour remote runs one crash away from losing everything
since the last iteration. This experiment-repo helper uploads a file or
directory tree to the same
``{prefix}/{repo-relative experiment dir}/{run_id}/{relative}`` keys the
run-end upload uses, so the final sweep overwrites rather than duplicates.

Planned to become the harness's default behavior; until then it lives here so
experiment branches do not depend on an unmerged harness feature.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.context import RunContext
from harness.storage.b2 import B2StorageConfig


def _repository_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _object_prefix(
    context: RunContext,
    *,
    base_prefix: str,
    experiment_module: str | None,
) -> str:
    """Match the key layout used by ``harness.storage.b2.upload_run_artifacts``."""
    segments: list[str] = []
    if base_prefix:
        segments.append(base_prefix.strip("/"))
    repository = _repository_root(context.experiment_dir)
    if repository is not None:
        try:
            relative = context.experiment_dir.resolve().relative_to(
                repository.resolve()
            )
            segments.append(relative.as_posix())
        except ValueError:
            segments.append(context.experiment_dir.name)
    elif experiment_module:
        segments.append(experiment_module.replace(".", "/"))
    else:
        segments.append(context.experiment_dir.name)
    segments.append(context.run_id)
    return "/".join(segment for segment in segments if segment)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload_artifact_path(
    context: RunContext,
    path: Path,
    *,
    config: B2StorageConfig | None = None,
    experiment_module: str | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Upload one artifact file or directory tree to B2 immediately.

    ``path`` must live under ``context.artifacts_dir``. Returns the upload
    summary; no manifest is written — the caller records what it needs.
    """
    resolved = config or B2StorageConfig.from_env()
    if resolved is None:
        raise RuntimeError(
            "B2 artifact upload is not configured. Set B2_BUCKET, B2_ENDPOINT, "
            "B2_APPLICATION_KEY_ID, and B2_APPLICATION_KEY."
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"artifact path does not exist: {path}")
    files = [path] if path.is_file() else sorted(
        f for f in path.rglob("*") if f.is_file()
    )

    prefix = _object_prefix(
        context,
        base_prefix=resolved.prefix,
        experiment_module=experiment_module,
    )
    s3 = client or resolved.s3_client()
    uploaded: list[dict[str, Any]] = []
    total_bytes = 0
    for file in files:
        relative_path = file.relative_to(context.artifacts_dir).as_posix()
        key = f"{prefix}/{relative_path}"
        size_bytes = file.stat().st_size
        s3.upload_file(str(file), resolved.bucket, key)
        uploaded.append(
            {
                "kind": "artifact",
                "relative_path": relative_path,
                "key": key,
                "uri": f"s3://{resolved.bucket}/{key}",
                "sha256": _file_sha256(file),
                "size_bytes": size_bytes,
            }
        )
        total_bytes += size_bytes
    return {
        "backend": "b2-s3",
        "bucket": resolved.bucket,
        "endpoint": resolved.endpoint,
        "prefix": prefix,
        "status": "completed",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(uploaded),
        "total_bytes": total_bytes,
        "files": uploaded,
    }
