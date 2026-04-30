#!/usr/bin/env bash
# Reference launcher for v12 contrastive fine-tune (MTEB-targeted).
#
# Assumes:
#   - venv at /root/srt_venv with mteb + sentence-transformers installed
#   - data prepared under /root/srt-adapter/data/contrastive/
#       train.jsonl  (columns: query, positive, negatives[])
#       val.jsonl    (columns: query, positive)
#   - warm-start checkpoint chosen from week-1 MTEB baseline run

set -euo pipefail

REPO=${REPO:-/root/srt-adapter}
PY=${PY:-/root/srt_venv/bin/python}
WARM=${WARM:-${REPO}/checkpoints/adapter_v8a/best_adapter.pt}
OUT=${OUT:-${REPO}/checkpoints/adapter_v12}

export PYTHONPATH=${REPO}:${PYTHONPATH:-}
export HF_HOME=${HF_HOME:-/root/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

cd "${REPO}"

echo "=== v12 contrastive fine-tune ==="
echo "warm-start: ${WARM}"
echo "output:     ${OUT}"

${PY} scripts/train_contrastive.py \
  --backbone Qwen/Qwen2.5-7B \
  --warm-start "${WARM}" \
  --train-data data/contrastive/train.jsonl \
  --val-data   data/contrastive/val.jsonl \
  --output-dir "${OUT}" \
  --batch-size 32 \
  --negatives-per-row 7 \
  --max-seq-len 256 \
  --temperature 0.02 \
  --lr 1e-4 \
  --warmup-steps 200 \
  --epochs 1 \
  --max-val-samples 2000 \
  --val-every 500 \
  --log-every 50 \
  --grad-clip 1.0 \
  --dtype bfloat16

echo "=== v12 MTEB evaluation (subset) ==="
${PY} scripts/mteb_eval.py \
  --backbone Qwen/Qwen2.5-7B \
  --adapter "${OUT}/best_adapter.pt" \
  --output-dir "${REPO}/artifacts/mteb/v12" \
  --task-types "Classification,STS,Clustering" \
  --batch-size 16 \
  --max-seq-len 256 \
  --dtype bfloat16
