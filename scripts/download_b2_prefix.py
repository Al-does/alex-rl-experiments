"""Download an S3/B2 prefix to a local directory using B2_* env vars.

Intended for remote GPU boxes: fetch a checkpoint tree from the bucket before
launching a resumed run, e.g.

    python scripts/download_b2_prefix.py \
        --prefix experiments/study/cond/<run-id>/log_spaced_checkpoints/<ckpt>/ \
        --dest /root/resume_ckpt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="S3 key prefix to sync")
    parser.add_argument("--dest", required=True, help="local destination dir")
    args = parser.parse_args()

    bucket = os.environ["B2_BUCKET"]
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["B2_ENDPOINT"],
        aws_access_key_id=os.environ["B2_APPLICATION_KEY_ID"],
        aws_secret_access_key=os.environ["B2_APPLICATION_KEY"],
    )

    dest = Path(args.dest)
    count = 0
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=args.prefix
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(args.prefix) :]
            if not rel:
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(target))
            count += 1
    if count == 0:
        raise SystemExit(f"no objects under s3://{bucket}/{args.prefix}")
    print(f"downloaded {count} files to {dest}")


if __name__ == "__main__":
    main()
