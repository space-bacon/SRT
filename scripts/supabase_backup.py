#!/usr/bin/env python3
"""Back up large artifacts to the Supabase 'SRT' bucket.

Covers the gap git can't: the .npz state dumps and .pt checkpoints are
gitignored or too big for the repo, and vast.ai boxes die without warning, so
anything living only on a remote box is one host-reclaim away from gone.

Credentials come from .env (gitignored) and are never printed.

  python scripts/supabase_backup.py --prefix omni artifacts/nla/omni/*.npz
  python scripts/supabase_backup.py --list
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import sys

BUCKET = os.environ.get("SUPABASE_BUCKET", "SRT")
REGION = "us-east-1"
# Supabase's S3 endpoint rejects UploadPart, so keep every object single-part.
MULTIPART_CEILING = 5 * 1024 ** 3
PART_SIZE = 45 * 1024 ** 2


def load_env(path=".env"):
    env = {}
    p = pathlib.Path(path)
    if not p.exists():
        sys.exit(f"no {path}; cannot authenticate")
    for line in p.read_text().splitlines():
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return env


def client(env):
    import boto3
    from botocore.config import Config

    missing = [k for k in ("SUPABASE_URL", "SUPABASE_S3_ACCESS_KEY_ID",
                           "SUPABASE_S3_SECRET_ACCESS_KEY") if k not in env]
    if missing:
        sys.exit(f"missing keys in .env: {missing}")
    return boto3.client(
        "s3",
        endpoint_url=f"{env['SUPABASE_URL']}/storage/v1/s3",
        aws_access_key_id=env["SUPABASE_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=env["SUPABASE_S3_SECRET_ACCESS_KEY"],
        region_name=REGION,
        config=Config(signature_version="s3v4"),
    )


def transfer_config():
    from boto3.s3.transfer import TransferConfig
    return TransferConfig(multipart_threshold=MULTIPART_CEILING,
                          multipart_chunksize=MULTIPART_CEILING,
                          use_threads=False)


def sha(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--prefix", default="misc")
    ap.add_argument("--flatten", action="store_true",
                    help="Key by basename only. Unsafe when names repeat.")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    s3 = client(load_env())
    TRANSFER = transfer_config()

    if args.list:
        resp = s3.list_objects_v2(Bucket=BUCKET)
        objs = resp.get("Contents", [])
        total = sum(o["Size"] for o in objs)
        for o in sorted(objs, key=lambda x: x["Key"]):
            print(f"  {o['Size']/1e6:9.1f} MB  {o['Key']}")
        print(f"{len(objs)} objects, {total/1e9:.2f} GB in bucket {BUCKET}")
        return

    if not args.files:
        sys.exit("no files given")

    # Skip re-uploads by comparing the local digest to the stored metadata.
    done = 0
    seen = {}
    for f in args.files:
        p = pathlib.Path(f)
        if not p.is_file():
            print(f"  SKIP (not a file) {f}")
            continue
        # Basenames repeat across sibling result dirs (summary.json), so key by
        # path unless explicitly asked to flatten.
        rel = p.name if args.flatten else str(p).lstrip("./").replace("/", "__")
        key = f"{args.prefix}/{rel}"
        if key in seen:
            sys.exit(f"key collision: {seen[key]} and {p} both map to {key}")
        seen[key] = p
        digest = sha(p)
        try:
            head = s3.head_object(Bucket=BUCKET, Key=key)
            if head.get("Metadata", {}).get("sha256") == digest:
                print(f"  = {key} ({p.stat().st_size/1e6:.1f} MB) already current")
                continue
        except Exception:
            pass
        size = p.stat().st_size
        if size <= PART_SIZE:
            s3.upload_file(str(p), BUCKET, key,
                           ExtraArgs={"Metadata": {"sha256": digest}},
                           Config=TRANSFER)
            print(f"  ^ {key} ({size/1e6:.1f} MB)")
        else:
            # The bucket enforces a per-object size cap, so large artifacts go
            # up as numbered parts with a manifest, matching the existing
            # backups/ convention. Reassemble with `cat *.partNNN > file`.
            n_parts = 0
            with open(p, "rb") as fh:
                while True:
                    blk = fh.read(PART_SIZE)
                    if not blk:
                        break
                    s3.put_object(Bucket=BUCKET,
                                  Key=f"{key}.part{n_parts:03d}", Body=blk)
                    n_parts += 1
            s3.put_object(
                Bucket=BUCKET, Key=f"{key}.manifest.json",
                Body=json.dumps({"parts": n_parts, "bytes": size,
                                 "sha256": digest,
                                 "part_size": PART_SIZE}).encode())
            print(f"  ^ {key} ({size/1e6:.1f} MB in {n_parts} parts)")
        done += 1
    print(f"uploaded {done} file(s) to {BUCKET}/{args.prefix}/")


if __name__ == "__main__":
    main()
