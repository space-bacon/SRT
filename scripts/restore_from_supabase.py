#!/usr/bin/env python
"""Restore files backed up by scripts/backup_to_supabase.py.

Plain objects download directly; chunked files (.manifest.json + .partNNN)
are reassembled and sha256-verified.

Usage:
    .venv-tools/bin/python scripts/restore_from_supabase.py backups/artifacts_local/gallery_full.npz /tmp/gallery_full.npz
    .venv-tools/bin/python scripts/restore_from_supabase.py --list [prefix]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUCKET = os.environ.get("SUNSTONE_BACKUP_BUCKET", "SRT")


def load_env() -> None:
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def client():
    import boto3
    from botocore.config import Config

    ref = os.environ["SUPABASE_URL"].split("//")[1].split(".")[0]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{ref}.storage.supabase.co/storage/v1/s3",
        region_name="us-east-1",
        aws_access_key_id=os.environ["SUPABASE_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["SUPABASE_S3_SECRET_ACCESS_KEY"],
        config=Config(retries={"max_attempts": 3}),
    )


def main() -> int:
    load_env()
    s3 = client()
    if len(sys.argv) >= 2 and sys.argv[1] == "--list":
        prefix = sys.argv[2] if len(sys.argv) > 2 else "backups/"
        pager = s3.get_paginator("list_objects_v2")
        for page in pager.paginate(Bucket=BUCKET, Prefix=prefix):
            for o in page.get("Contents", []):
                print(f"{o['Size']/1e6:9.1f} MB  {o['Key']}")
        return 0

    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    key, dest = sys.argv[1], Path(sys.argv[2])
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Chunked?
    try:
        manifest = json.loads(
            s3.get_object(Bucket=BUCKET, Key=key + ".manifest.json")["Body"].read())
    except Exception:
        manifest = None

    if manifest is None:
        s3.download_file(BUCKET, key, str(dest))
        print(f"restored {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return 0

    sha = hashlib.sha256()
    with open(dest, "wb") as f:
        for part in manifest["parts"]:
            blob = s3.get_object(Bucket=BUCKET, Key=part["key"])["Body"].read()
            sha.update(blob)
            f.write(blob)
    if sha.hexdigest() != manifest["sha256"]:
        print("CHECKSUM MISMATCH — restored file is corrupt", file=sys.stderr)
        return 1
    print(f"restored {dest} ({dest.stat().st_size/1e6:.1f} MB, sha256 verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
