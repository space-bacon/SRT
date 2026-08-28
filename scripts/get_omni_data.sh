#!/bin/bash
# Stage every modality: image captions, audio+caption, video+caption.
# Sourced from HF because cocodataset.org stalled at ~250MB while HF sustains
# ~370MB/s on this box.
set -u
export HF_HOME=/root/.hf_home
cd /root

echo "=== $(date -u +%H:%M:%S) coco captions"
/venv/main/bin/python - <<'PY'
import os, zipfile
from huggingface_hub import hf_hub_download
for repo, fn in [("merve/coco", "annotations_trainval2017.zip"),
                 ("ydshieh/coco_dataset_script", "annotations_trainval2017.zip")]:
    try:
        p = hf_hub_download(repo, fn, repo_type="dataset")
        zipfile.ZipFile(p).extractall("/root")
        print("  ok from", repo, sorted(os.listdir("/root/annotations"))[:3])
        break
    except Exception as e:
        print("  miss", repo, type(e).__name__)
PY

echo "=== $(date -u +%H:%M:%S) audiocaps"
/venv/main/bin/hf download confit/audiocaps --repo-type dataset 2>&1 | tail -1

echo "=== $(date -u +%H:%M:%S) msr-vtt"
/venv/main/bin/hf download friedrichor/MSR-VTT --repo-type dataset 2>&1 | tail -1

echo "=== $(date -u +%H:%M:%S) DATA DONE  cache $(du -sh /root/.hf_home | cut -f1)"
