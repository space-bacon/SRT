#!/usr/bin/env bash
# Capability eval: run scripts/capability_eval.py on v15a, v18, v22c_a050.
# Sequential on GPU 0 (~5 min each).
set -euo pipefail
REPO=/root/srt-adapter
PY=/root/srt_venv/bin/python
cd "${REPO}"
export PYTHONPATH=${REPO}
export HF_HOME=/root/hf_cache
export CUDA_VISIBLE_DEVICES=0

LOG=${REPO}/artifacts/capability_chain.log
echo "[$(date)] === capability eval: v15a, v18, v22c_a050 ===" > "${LOG}"

declare -A CKPTS=(
  [v15a]=${REPO}/checkpoints/adapter_v15a/best_adapter.pt
  [v18]=${REPO}/checkpoints/adapter_v18/best_adapter.pt
  [v22c_a050]=${REPO}/checkpoints/adapter_v22c_a050/best_adapter.pt
)

for tag in v15a v18 v22c_a050; do
  ckpt=${CKPTS[$tag]}
  if [ ! -f "${ckpt}" ]; then
    echo "[$(date)] MISSING ckpt for ${tag}: ${ckpt}" | tee -a "${LOG}"
    continue
  fi
  echo "[$(date)] --- ${tag} (${ckpt}) ---" >> "${LOG}"
  ${PY} scripts/capability_eval.py \
    --backbone Qwen/Qwen2.5-7B \
    --adapter "${ckpt}" \
    --output-dir ${REPO}/artifacts/capability/${tag} \
    --batch-size 16 --max-seq-len 128 --dtype bfloat16 >> "${LOG}" 2>&1 \
    && echo "[$(date)] ${tag} done" >> "${LOG}" \
    || echo "[$(date)] ${tag} FAILED" >> "${LOG}"
done
echo "[$(date)] === capability eval DONE ===" >> "${LOG}"
