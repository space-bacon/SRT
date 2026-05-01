#!/usr/bin/env bash
# v22c: soup v18 + v21a (analogous to v21b but with the new SOTA).
# v18 = English-purist SOTA, v21a = new mean-SOTA with cross-lingual gains.
# We expect a soup to keep most of v21a's cross-lingual wins while
# recovering some of v18's English calibration. Two alphas: 0.5 (even) and
# 0.3 (v21a-leaning) to bracket. ~16 min eval each.
set -euo pipefail
REPO=/root/srt-adapter
PY=/root/srt_venv/bin/python
cd "${REPO}"
export PYTHONPATH=${REPO}
export HF_HOME=/root/hf_cache
export CUDA_VISIBLE_DEVICES=0

VLOG=${REPO}/artifacts/v22c_chain.log
echo "[$(date)] === v22c: soup v18 + v21a ===" >> "${VLOG}"

# ─── 1. soup at three alphas (alpha = weight on v18) ───
for ALPHA in 0.50 0.30 0.70; do
  TAG=$(echo "${ALPHA}" | sed 's/0\.//')   # 50, 30, 70
  OUT=${REPO}/checkpoints/adapter_v22c_a0${TAG}/best_adapter.pt
  mkdir -p "$(dirname ${OUT})"
  ${PY} scripts/soup_adapters.py \
    --ckpt-a ${REPO}/checkpoints/adapter_v18/best_adapter.pt \
    --ckpt-b ${REPO}/checkpoints/adapter_v21a/best_adapter.pt \
    --alpha ${ALPHA} \
    --out ${OUT} >> "${VLOG}" 2>&1
done

# ─── 2. eval each soup ───
for TAG in 050 030 070; do
  echo "[$(date)] === v22c: eval v22c_a0${TAG} ===" >> "${VLOG}"
  ${PY} scripts/mteb_eval.py \
    --backbone Qwen/Qwen2.5-7B \
    --adapter ${REPO}/checkpoints/adapter_v22c_a0${TAG}/best_adapter.pt \
    --output-dir ${REPO}/artifacts/mteb/v22c_a0${TAG} \
    --task-types STS --task-langs eng \
    --batch-size 16 --max-seq-len 256 --dtype bfloat16 >> "${VLOG}" 2>&1
done

echo "[$(date)] === v22c DONE ===" >> "${VLOG}"
