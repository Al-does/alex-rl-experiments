"""Verify B2 archives and check or materialize their report projections."""
import argparse
from hashlib import sha256
import json

from harness.storage.b2 import B2StorageConfig

from .projection import PROJECTION, project
from .report import FINAL, SEEDS, STUDY_DIR, TOTAL_CHECKPOINTS, load_runs

ARCHIVE = "b2_full_results.jsonl"
FULL_RESULTS = ("condition_summary.json", "training_curves.jsonl")


def read_verified_object(client, bucket, file):
    with client.get_object(Bucket=bucket, Key=file["key"])["Body"] as body:
        contents = body.read()
    if len(contents) != file["size_bytes"] or sha256(contents).hexdigest() != file["sha256"]:
        raise ValueError(f"B2 object size or SHA-256 mismatch: {file['key']}")
    return contents


def verify_run(client, manifest, summary_dir, *, write_projections=False):
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
    recipe = "resolved_recipe.json"
    if sha256((summary_dir / recipe).read_bytes()).hexdigest() != files[f"results/{recipe}"]["sha256"]:
        raise ValueError(f"resolved recipe differs from B2 index: {manifest['run_id']}")
    archive, full = [], {}
    for name in FULL_RESULTS:
        file = files[f"results/{name}"]
        full[name] = read_verified_object(client, bucket, file)
        local = summary_dir / name
        if write_projections and local.read_bytes() != full[name]:
            raise ValueError(f"tracked full result differs from B2: {local}")
        archive.append({
            "path": local.relative_to(STUDY_DIR).as_posix(),
            "b2_uri": f"s3://{bucket}/{file['key']}",
            "bytes": file["size_bytes"],
            "sha256": file["sha256"],
        })
    summary = json.loads(full["condition_summary.json"])
    curves = [json.loads(line) for line in full["training_curves.jsonl"].splitlines()]
    projected = project(summary, curves)
    if not write_projections and (summary_dir / PROJECTION).read_bytes() != projected.encode():
        raise ValueError(f"projection differs from B2: {summary_dir / PROJECTION}")
    read_verified_object(client, bucket, files[f"checkpoints/step_{FINAL:010d}.npz"])
    return len(expected), sum(expected.values()), archive, projected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-projections", action="store_true",
                        help="materialize projections only after matching tracked full results against B2")
    args = parser.parse_args()
    config = B2StorageConfig.from_env()
    if config is None:
        raise RuntimeError("B2 is not configured")
    client = config.s3_client()
    runs = load_runs(STUDY_DIR)
    archive, projected = [], {}
    total_objects = total_bytes = 0
    for leaf, by_seed in runs.items():
        for seed in SEEDS:
            manifest, _ = by_seed[seed]
            path = STUDY_DIR / leaf / "results" / manifest["run_id"]
            n, size, files, projection = verify_run(
                client, manifest, path, write_projections=args.write_projections,
            )
            total_objects += n
            total_bytes += size
            archive.extend(files)
            projected[path / PROJECTION] = projection
            print(f"{leaf}: seed={seed} run={manifest['run_id']} objects={n} bytes={size} "
                  "full_results=2 final_checkpoint_sha256=verified", flush=True)
    archive_text = "".join(json.dumps(row, separators=(",", ":")) + "\n"
                           for row in sorted(archive, key=lambda row: row["path"]))
    archive_path = STUDY_DIR / "results" / ARCHIVE
    if args.write_projections:
        for path, projection in projected.items():
            path.write_text(projection)
        archive_path.write_text(archive_text)
    elif archive_path.read_bytes() != archive_text.encode():
        raise ValueError(f"full-results manifest differs from B2: {archive_path}")
    print(f"{len(projected)} projections; {len(archive)} full files SHA-256 verified "
          f"({sum(file['bytes'] for file in archive)} bytes); "
          f"{total_objects} indexed objects ({total_bytes} bytes); "
          f"{len(projected)} final checkpoint hashes verified")


if __name__ == "__main__":
    main()
