#!/usr/bin/env bash
# Pack thumbnails and publish. Written as a file rather than an inline ssh
# heredoc because the nested quoting silently produced nothing, and because a
# pgrep check for an inline command matches the ssh invocation itself.
set -uo pipefail
PY=/venv/main/bin/python
export HF_HOME=/root/.hf_home

echo "[pack $(date -u +%H:%M:%SZ)] packing"
$PY -u /root/pack_thumbnails.py \
  --thumbs /root/thumbs \
  --index /root/gallery_123k_v2.srtidx \
  --out /root/packed || { echo "PACK FAILED"; exit 1; }

echo "[pack $(date -u +%H:%M:%SZ)] uploading"
$PY - <<'PYEOF'
from huggingface_hub import HfApi
api = HfApi()
repo = "RiverRider/srt-coco-thumbs"
api.create_repo(repo, repo_type="dataset", exist_ok=True)
for f in ["thumbs.bin", "thumbs_offsets.bin"]:
    api.upload_file(path_or_fileobj=f"/root/packed/{f}", path_in_repo=f,
                    repo_id=repo, repo_type="dataset")
    print("uploaded", f, flush=True)
PYEOF

echo "[pack $(date -u +%H:%M:%SZ)] PACK_DONE"
