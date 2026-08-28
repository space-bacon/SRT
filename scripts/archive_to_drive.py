#!/usr/bin/env python3
"""Mirror everything that matters onto an offline drive.

Four sources, because losing any one of them has already cost us work:
the working tree, the Hugging Face cache that holds the host models, our own
published Hub repos, and the Supabase bucket.

Every stage is resumable and skips files that already match, so re-running
after a disconnect costs a scan rather than a re-download.

    python scripts/archive_to_drive.py --dest /Volumes/LaCie/SRT-ARCHIVE
    python scripts/archive_to_drive.py --dest ... --only hub
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

HOME = pathlib.Path.home()
# Virtualenvs hardcode absolute paths, so a restored copy is broken anyway.
RSYNC_EXCLUDE = [".venv", ".venv-*", "venv", "node_modules", "__pycache__",
                 "*.pyc", ".DS_Store", ".pytest_cache", ".mypy_cache",
                 ".ipynb_checkpoints"]
# The drive is unencrypted, so live credentials stay off it. Templates are kept
# because a rebuild needs to know which keys to supply.
SECRET_KEEP = [".env.example", ".env.sample", ".env.template"]
SECRET_EXCLUDE = [".env", ".env.*", "*.key", "*.pem", "*.p12", "*.pfx",
                  "id_rsa*", "id_ed25519*", "id_ecdsa*", ".netrc",
                  ".git-credentials", ".npmrc", ".pypirc",
                  "credentials*.json", "service-account*.json",
                  "token", "stored_tokens", "*.keychain*", "secrets*.yaml",
                  "secrets*.yml"]
LOCAL = [(HOME / "development", "development"),
         (HOME / ".cache/huggingface", "hf-cache"),
         (HOME / "Downloads", "Downloads"),
         (HOME / "Desktop", "Desktop")]


def sh(cmd):
    print("  $ " + " ".join(str(c) for c in cmd[:6]) + " ...", flush=True)
    return subprocess.call(cmd)


def stage_local(dest):
    for src, name in LOCAL:
        if not src.exists():
            print(f"skip {name}, not present", flush=True)
            continue
        out = dest / name
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {name}  <- {src}", flush=True)
        # -l keeps symlinks as symlinks, which the HF cache depends on. Owner
        # and group are dropped because the volume has ownership disabled.
        cmd = ["rsync", "-rlptD", "--delete", "--partial", "--human-readable",
               "--stats"]
        # rsync takes the first matching rule, so the keeps must precede the
        # secret excludes that would otherwise swallow them.
        for k in SECRET_KEEP:
            cmd += ["--include", k]
        for e in RSYNC_EXCLUDE + SECRET_EXCLUDE:
            cmd += ["--exclude", e]
        cmd += [str(src) + "/", str(out) + "/"]
        sh(cmd)


def stage_hub(dest):
    from huggingface_hub import HfApi, get_token, snapshot_download
    api = HfApi(token=get_token())
    out = dest / "hf-repos"
    listers = [("model", api.list_models), ("dataset", api.list_datasets),
               ("space", api.list_spaces)]
    repos = [(k, r.id) for k, l in listers for r in l(author="RiverRider")]
    print(f"\n=== hf-repos: {len(repos)} repos", flush=True)
    for i, (kind, rid) in enumerate(sorted(repos, key=lambda x: x[1]), 1):
        target = out / kind / rid.split("/", 1)[1]
        print(f"  [{i}/{len(repos)}] {kind}/{rid}", flush=True)
        try:
            snapshot_download(rid, repo_type=kind, local_dir=str(target),
                              token=get_token(), max_workers=4)
        except Exception as e:
            print(f"      FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)


def stage_supabase(dest, repo_root):
    sys.path.insert(0, str(repo_root / "scripts"))
    import supabase_backup as sb
    env = sb.load_env(str(repo_root / ".env"))
    s3 = sb.client(env)
    out = dest / "supabase"
    out.mkdir(parents=True, exist_ok=True)
    token, n, got = None, 0, 0
    print("\n=== supabase bucket", sb.BUCKET, flush=True)
    while True:
        kw = {"Bucket": sb.BUCKET, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            n += 1
            p = out / o["Key"]
            if p.exists() and p.stat().st_size == o["Size"]:
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(sb.BUCKET, o["Key"], str(p))
            got += 1
            if got % 25 == 0:
                print(f"  {got} downloaded of {n} seen", flush=True)
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    print(f"  {n} objects listed, {got} newly downloaded", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--only", choices=["local", "hub", "supabase"])
    a = ap.parse_args()
    dest = pathlib.Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    t0 = time.time()

    if a.only in (None, "local"):
        stage_local(dest)
    if a.only in (None, "hub"):
        stage_hub(dest)
    if a.only in (None, "supabase"):
        stage_supabase(dest, repo_root)

    manifest = {
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_hours": round((time.time() - t0) / 3600, 2),
        "sources": {name: str(src) for src, name in LOCAL},
        "excluded": RSYNC_EXCLUDE,
        "secrets_excluded": SECRET_EXCLUDE,
        "secrets_note": "No live credentials on this drive by design. "
                        ".env.example templates are kept; supply real values "
                        "from the password manager. See REBUILD.md.",
    }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"\ndone in {manifest['elapsed_hours']} h -> {dest}", flush=True)


if __name__ == "__main__":
    main()
