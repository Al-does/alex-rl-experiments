"""Check compact run manifests against B2 indexes and hash final checkpoints."""
from hashlib import sha256
import json

from harness.storage.b2 import B2StorageConfig

from .report import FINAL, SEEDS, STUDY_DIR, TOTAL_CHECKPOINTS, load_runs


def verify_run(client, manifest, summary_dir):
    remote = manifest["remote_artifacts"]
    bucket, prefix = remote["bucket"], remote["prefix"]
    manifest_key = remote["canonical_manifest_key"]
    with client.get_object(Bucket=bucket, Key=manifest_key)["Body"] as body:
        canonical = json.load(body)
    for key in ("status", "bucket", "prefix", "file_count", "total_bytes", "canonical_manifest_key"):
        if canonical[key] != remote[key]:
            raise ValueError(f"compact/canonical manifest mismatch: {manifest['run_id']}/{key}")
    expected = {f["key"]: f["size_bytes"] for f in canonical["files"]}
    if (remote["status"] != "completed" or len(expected) != remote["file_count"] or
            sum(expected.values()) != remote["total_bytes"]):
        raise ValueError(f"inconsistent B2 index: {manifest['run_id']}")
    listing = {
        obj["Key"]: obj["Size"]
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/")
        for obj in page.get("Contents", ())
    }
    if not set(expected).issubset(listing) or any(listing[k] != n for k, n in expected.items()):
        raise ValueError(f"B2 file missing or wrong size: {manifest['run_id']}")
    files = {f["relative_path"]: f for f in canonical["files"]}
    checkpoints = {name for name in files if name.startswith("checkpoints/step_")}
    batch = FINAL // (TOTAL_CHECKPOINTS - 1)
    if checkpoints != {f"checkpoints/step_{i * batch:010d}.npz" for i in range(TOTAL_CHECKPOINTS)}:
        raise ValueError(f"incorrect checkpoint set: {manifest['run_id']}")
    for name in ("condition_summary.json", "training_curves.jsonl", "resolved_recipe.json"):
        if sha256((summary_dir / name).read_bytes()).hexdigest() != files[f"results/{name}"]["sha256"]:
            raise ValueError(f"compact results differ from B2 index: {manifest['run_id']}/{name}")
    final = files[f"checkpoints/step_{FINAL:010d}.npz"]
    with client.get_object(Bucket=bucket, Key=final["key"])["Body"] as body:
        digest = sha256(body.read()).hexdigest()
    if digest != final["sha256"]:
        raise ValueError(f"final checkpoint SHA-256 mismatch: {manifest['run_id']}")
    return len(expected), sum(expected.values()), digest


def main():
    config = B2StorageConfig.from_env()
    if config is None:
        raise RuntimeError("B2 is not configured")
    client = config.s3_client()
    runs = load_runs(STUDY_DIR)
    for leaf, by_seed in runs.items():
        for seed in SEEDS:
            manifest, _ = by_seed[seed]
            path = STUDY_DIR / leaf / "results" / manifest["run_id"]
            n, size, digest = verify_run(client, manifest, path)
            print(f"{leaf}: seed={seed} run={manifest['run_id']} objects={n} bytes={size} "
                  f"final_sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
