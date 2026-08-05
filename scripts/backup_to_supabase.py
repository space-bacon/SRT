#!/usr/bin/env python
"""Incremental backup of the Mac-only irreplaceables to Supabase Storage.

Syncs (only new/changed files, compared by size + mtime metadata) every
irreplaceable data/checkpoint tree under ~/development — corpora, raw
crawls, checkpoints, run artifacts — to s3://SRT/backups/<set>/...

Credentials come from .env (SUPABASE_URL + SUPABASE_S3_ACCESS_KEY_ID/SECRET).
Run manually or via the com.sunstonenorth.backup LaunchAgent (daily 03:30):
    .venv-tools/bin/python scripts/backup_to_supabase.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUCKET = os.environ.get("SUNSTONE_BACKUP_BUCKET", "SRT")
PREFIX = "backups"
# Supabase enforces a per-request upload cap (50MB observed even on Pro
# until the global limit propagates); anything bigger is stored as .partNNN
# objects plus a .manifest.json. scripts/restore_from_supabase.py reassembles.
CHUNK = 45 * 1024 * 1024
DEV = ROOT.parent
# name -> (base dir, excluded top-level subdirs). Everything irreplaceable
# under ~/development: corpora, crawls, checkpoints, run artifacts.
SYNC_SETS: dict[str, tuple[Path, set[str]]] = {
    "artifacts_local": (ROOT / "artifacts" / "local", set()),
    "artifacts": (ROOT / "artifacts", {"local"}),
    "checkpoints": (ROOT / "checkpoints", set()),
    "srt_adapter_data": (ROOT / "data", set()),
    "srt_adapter_private": (ROOT / "private", set()),
    "srt_data": (DEV / "SRT" / "data", set()),
    "srt_checkpoints": (DEV / "SRT" / "checkpoints", set()),
}


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


def needs_upload(s3, key: str, path: Path) -> bool:
    probe = key if path.stat().st_size <= CHUNK else key + ".manifest.json"
    try:
        head = s3.head_object(Bucket=BUCKET, Key=probe)
    except Exception:
        return True
    meta = head.get("Metadata", {})
    if path.stat().st_size > CHUNK:
        return (meta.get("mtime") != str(int(path.stat().st_mtime))
                or meta.get("size") != str(path.stat().st_size))
    return (head["ContentLength"] != path.stat().st_size
            or meta.get("mtime") != str(int(path.stat().st_mtime)))


def upload_chunked(s3, key: str, path: Path) -> None:
    """Store a large file as 45MB part objects + a manifest."""
    import hashlib
    import json

    size = path.stat().st_size
    parts = []
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        i = 0
        while True:
            blob = f.read(CHUNK)
            if not blob:
                break
            sha.update(blob)
            part_key = f"{key}.part{i:03d}"
            s3.put_object(Bucket=BUCKET, Key=part_key, Body=blob)
            parts.append({"key": part_key, "size": len(blob)})
            i += 1
    manifest = {"file": path.name, "size": size, "sha256": sha.hexdigest(),
                "parts": parts}
    s3.put_object(Bucket=BUCKET, Key=key + ".manifest.json",
                  Body=json.dumps(manifest).encode(),
                  Metadata={"mtime": str(int(path.stat().st_mtime)),
                            "size": str(size)})


def main() -> int:
    load_env()
    s3 = client()
    uploaded = skipped = failed = 0
    bytes_up = 0
    for name, (base, excludes) in SYNC_SETS.items():
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(base)
            if rel.parts and rel.parts[0] in excludes:
                continue
            if any(part.startswith(".") for part in rel.parts):
                continue
            key = f"{PREFIX}/{name}/{rel}"
            if not needs_upload(s3, key, path):
                skipped += 1
                continue
            try:
                if path.stat().st_size > CHUNK:
                    upload_chunked(s3, key, path)
                else:
                    s3.upload_file(
                        str(path), BUCKET, key,
                        ExtraArgs={"Metadata": {"mtime": str(int(path.stat().st_mtime))}},
                    )
                uploaded += 1
                bytes_up += path.stat().st_size
                print(f"  up  {key}  ({path.stat().st_size/1e6:.1f} MB)")
            except Exception as e:  # noqa: BLE001 - keep syncing the rest
                failed += 1
                print(f"  FAIL {key}: {e}", file=sys.stderr)
    print(f"{time.strftime('%F %T')} backup: {uploaded} uploaded "
          f"({bytes_up/1e6:.0f} MB), {skipped} unchanged, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
