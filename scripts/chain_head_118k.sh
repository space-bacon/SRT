#!/usr/bin/env bash
# Overnight chain: wait for the re-encode, refit the head, rebuild the gallery,
# and get everything off this box.
#
# Gated on flag files rather than pgrep. A pgrep wait-gate matches the launcher
# whose cmdline contains the pattern, and separately matches any later ssh
# command quoting it, which has stalled chains here before.
#
# Every step is skipped if its output already exists, so re-running after a
# crash costs only the step that failed.
#
#   nohup bash /root/chain_head.sh > /root/chain_head.log 2>&1 &
set -uo pipefail

PY=/venv/main/bin/python
export HF_HOME=/root/.hf_home
RAW=/root/raw118k
HEAD=/root/browser_head_118k.pt
GAL=/root/gallery_123k_v2.npz
IDX=/root/gallery_123k_v2.srtidx
REPORT=/root/browser_head_118k_report.json
REPO=RiverRider/srt-browser-head-118k

say() { echo "[chain $(date -u +%H:%M:%SZ)] $*"; }

say "waiting for both encode shards"
while [ ! -f "$RAW/shard0.done" ] || [ ! -f "$RAW/shard1.done" ]; do
  sleep 60
done
say "encode complete: $(ls $RAW/*.npz | wc -l) chunks"

if [ ! -f "$HEAD" ]; then
  say "fitting head on 118K pairs"
  $PY -u /root/browser_head_118k.py \
    --raw-dir "$RAW" \
    --val-states /root/qwen38_coco_ls/images.pt \
    --ann /root/annotations \
    --out "$HEAD" --gallery-out "$GAL" --report "$REPORT" \
    || { say "FIT FAILED"; exit 1; }
else
  say "head already present, skipping fit"
fi

if [ ! -f "$IDX" ] && [ -f "$GAL" ]; then
  say "building int8 index"
  $PY -u /root/merge_gallery_shards.py \
    --chunks /root/nonexistent --also "$GAL" --also-prefix "" \
    --dtype int8 --out "$IDX" || say "INDEX FAILED"
fi

# This box has died mid-run before and taken everything with it, so the small
# artifacts leave as soon as they exist rather than when the chain ends.
say "publishing artifacts"
$PY - <<'PYEOF' || say "UPLOAD FAILED (artifacts still on disk)"
import os
from huggingface_hub import HfApi
api = HfApi()
repo = "RiverRider/srt-browser-head-118k"
api.create_repo(repo, repo_type="model", exist_ok=True)
for p in ["/root/browser_head_118k.pt",
          "/root/browser_head_118k_report.json",
          "/root/gallery_123k_v2.srtidx"]:
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo=os.path.basename(p),
                        repo_id=repo, repo_type="model")
        print("uploaded", p, flush=True)
PYEOF

say "resuming thumbnail upload if it stopped"
if ! grep -q THUMBS_UPLOADED /root/thumbs_up.log 2>/dev/null; then
  /venv/main/bin/hf upload-large-folder RiverRider/srt-coco-thumbs /root/thumbs \
    --repo-type=dataset --num-workers=8 >> /root/thumbs_up.log 2>&1 \
    && echo THUMBS_UPLOADED >> /root/thumbs_up.log
fi

say "CHAIN_DONE"
[ -f "$REPORT" ] && cat "$REPORT"
