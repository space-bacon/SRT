#!/usr/bin/env bash
# Get everything irreplaceable off this box before it is destroyed.
#
# Ordered by cost to regenerate, so a failure partway through loses the least:
#   raw118k     ~2h of GPU. The input to any future head refit, and the thing
#               whose absence forced a full re-encode last time.
#   images.pt   val2017 states at eight layers, base and abliterated. Every
#               evaluation in this line scores against it.
#   captions.pt 25K captions through the 27B tower, twice. Large but expensive.
#
# capcache is deliberately not here: 7.1 GB that regenerates in ~15 minutes from
# a model that is itself on the Hub.
set -uo pipefail
export HF_HOME=/root/.hf_home
REPO=RiverRider/srt-qwen38-coco-states

/venv/main/bin/python - <<'PY'
import os, sys
from huggingface_hub import HfApi
api = HfApi()
repo = "RiverRider/srt-qwen38-coco-states"
api.create_repo(repo, repo_type="dataset", exist_ok=True)

def push_file(path, dest):
    if not os.path.exists(path):
        print("missing, skipping", path, flush=True); return
    gb = os.path.getsize(path) / 1e9
    print(f"--> {dest} ({gb:.1f} GB)", flush=True)
    api.upload_file(path_or_fileobj=path, path_in_repo=dest,
                    repo_id=repo, repo_type="dataset")
    print("    done", flush=True)

print("=== raw118k: image states for head refits ===", flush=True)
api.upload_folder(folder_path="/root/raw118k", path_in_repo="raw118k",
                  repo_id=repo, repo_type="dataset",
                  allow_patterns=["*.npz", "*.done"])
print("    done", flush=True)

print("=== val states ===", flush=True)
push_file("/root/qwen38_coco_ls/images.pt", "qwen38_coco_ls/images.pt")

print("=== caption states (large) ===", flush=True)
push_file("/root/qwen38_coco_ls/captions.pt", "qwen38_coco_ls/captions.pt")

print("=== arm results ===", flush=True)
import glob
for f in sorted(glob.glob("/root/v2_*.json")) + sorted(glob.glob("/root/browser_head_118k_report.json")):
    push_file(f, "results/" + os.path.basename(f))
PY

echo BACKUP_DONE
