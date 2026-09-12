"""Download a completed run's final Tune checkpoint from B2 for warm starts.

Locates the newest ``tune/PPO_*/checkpoint_000000/`` subtree under a run's
B2 prefix and restores it to a local directory suitable for
``rl-harness ... --resume-from`` warm-start continuations.
"""

import argparse
from pathlib import Path

from harness.storage.b2 import B2StorageConfig

FINAL_CHECKPOINT_DIR = "checkpoint_000000"
CHECKPOINT_SENTINEL = "algorithm_state.pkl"


def _final_checkpoint_prefixes(client, bucket: str, run_prefix: str) -> dict:
    paginator = client.get_paginator("list_objects_v2")
    candidates = {}
    marker = f"/{FINAL_CHECKPOINT_DIR}/"
    for page in paginator.paginate(
        Bucket=bucket, Prefix=f"{run_prefix}tune/"
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(CHECKPOINT_SENTINEL) and marker in key:
                prefix = key[: key.index(marker) + len(marker)]
                previous = candidates.get(prefix)
                if previous is None or obj["LastModified"] > previous:
                    candidates[prefix] = obj["LastModified"]
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-prefix",
        required=True,
        help="B2 key prefix of the finished run (the directory containing tune/)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = B2StorageConfig.from_env()
    if config is None:
        raise SystemExit("B2 artifact storage is not configured (B2_* env)")
    client = config.s3_client()
    run_prefix = args.run_prefix.strip("/") + "/"

    candidates = _final_checkpoint_prefixes(client, config.bucket, run_prefix)
    if not candidates:
        raise SystemExit(f"no final checkpoint found under {run_prefix}tune/")
    checkpoint_prefix = max(candidates, key=candidates.get)
    print(f"[download_warm_start] source {checkpoint_prefix}", flush=True)

    count = 0
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=config.bucket, Prefix=checkpoint_prefix
    ):
        for obj in page.get("Contents", []):
            relative = obj["Key"][len(checkpoint_prefix) :]
            destination = args.out / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(config.bucket, obj["Key"], str(destination))
            count += 1
    if not (args.out / CHECKPOINT_SENTINEL).exists():
        raise SystemExit(
            f"downloaded checkpoint at {args.out} is missing "
            f"{CHECKPOINT_SENTINEL}"
        )
    print(
        f"[download_warm_start] {count} files -> {args.out}", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
