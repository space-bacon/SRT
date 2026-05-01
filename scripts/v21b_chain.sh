#!/usr/bin/env bash
# v21b: model souping experiment.
# Eval two souped checkpoints (alpha=0.7 v18-favored, alpha=0.5 even mix)
# of v18 (English-STS SOTA) and v20 (cross-lingual sibling).
# No training. ~16 min eval each.
set -euo pipefail
REPO=/root/srt-adapter
PY=/root/srt_venv/bin/python
cd "${REPO}"
export PYTHONPATH=${REPO}
export HF_HOME=/root/hf_cache
export CUDA_VISIBLE_DEVICES=0

VLOG=${REPO}/artifacts/v21b_chain.log
echo "[$(date)] === v21b: eval souped (alpha=0.70 v18, 0.30 v20) ===" >> "${VLOG}"
${PY} scripts/mteb_eval.py \
  --backbone Qwen/Qwen2.5-7B \
  --adapter ${REPO}/checkpoints/adapter_v21b_a070/best_adapter.pt \
  --output-dir ${REPO}/artifacts/mteb/v21b_a070 \
  --task-types STS --task-langs eng \
  --batch-size 16 --max-seq-len 256 --dtype bfloat16 >> "${VLOG}" 2>&1

echo "[$(date)] === v21b: eval souped (alpha=0.50 v18, 0.50 v20) ===" >> "${VLOG}"
${PY} scripts/mteb_eval.py \
  --backbone Qwen/Qwen2.5-7B \
  --adapter ${REPO}/checkpoints/adapter_v21b_a050/best_adapter.pt \
  --output-dir ${REPO}/artifacts/mteb/v21b_a050 \
  --task-types STS --task-langs eng \
  --batch-size 16 --max-seq-len 256 --dtype bfloat16 >> "${VLOG}" 2>&1

echo "[$(date)] === v21b DONE ===" >> "${VLOG}"
