import os
import pathlib
import sys

import boto3
from botocore.config import Config


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: download_b2_checkpoint.py <s3-key-prefix> <local-dir>")
        sys.exit(1)
    prefix = sys.argv[1].rstrip("/") + "/"
    local_dir = pathlib.Path(sys.argv[2])
    local_dir.mkdir(parents=True, exist_ok=True)

    endpoint = os.environ["B2_ENDPOINT"]
    key_id = os.environ["B2_APPLICATION_KEY_ID"]
    secret = os.environ["B2_APPLICATION_KEY"]
    bucket = os.environ.get("B2_BUCKET", "slop-bucket")
    region = os.environ.get("B2_REGION", "us-west-004")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        region_name=region,
        config=Config(signature_version="s3v4"),
    )

    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = pathlib.Path(key).relative_to(prefix)
            target = local_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(target))
            count += 1

    print(f"Downloaded {count} files to {local_dir}")
    if count == 0:
        print("No objects found at prefix", prefix)
        sys.exit(1)


if __name__ == "__main__":
    main()
