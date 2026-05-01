#!/usr/bin/env bash
# v23 noise floor: re-eval v18 and v22c_a050 3x each on the MTEB STS suite
# to estimate per-task variance from bfloat16 / DataLoader nondeterminism.
# Splits across 2 GPUs for parallelism: 3 rounds x ~16 min = ~50 min total.
#
# Output: artifacts/mteb/{v18,v22c_a050}_seed{0,1,2}/summary.json
set -euo pipefail
REPO=/root/srt-adapter
PY=/root/srt_venv/bin/python
cd "${REPO}"
export PYTHONPATH=${REPO}
export HF_HOME=/root/hf_cache

VLOG=${REPO}/artifacts/v23_noise_chain.log
echo "[$(date)] === v23 noise floor: 3x v18 + 3x v22c_a050 ===" >> "${VLOG}"

CKPT_V18=${REPO}/checkpoints/adapter_v18/best_adapter.pt
CKPT_SOTA=${REPO}/checkpoints/adapter_v22c_a050/best_adapter.pt

run_one () {
  local gpu=$1 ckpt=$2 outdir=$3
  CUDA_VISIBLE_DEVICES=${gpu} ${PY} scripts/mteb_eval.py \
    --backbone Qwen/Qwen2.5-7B \
    --adapter "${ckpt}" \
    --output-dir "${outdir}" \
    --task-types STS --task-langs eng \
    --batch-size 16 --max-seq-len 256 --dtype bfloat16 >> "${VLOG}" 2>&1
}

for SEED in 0 1 2; do
  echo "[$(date)] --- round seed=${SEED}: GPU0=v18, GPU1=v22c_a050 ---" >> "${VLOG}"
  run_one 0 "${CKPT_V18}"  "${REPO}/artifacts/mteb/v18_seed${SEED}"        &
  PID0=$!
  run_one 1 "${CKPT_SOTA}" "${REPO}/artifacts/mteb/v22c_a050_seed${SEED}"  &
  PID1=$!
  wait ${PID0} ${PID1}
  echo "[$(date)] --- round seed=${SEED} done ---" >> "${VLOG}"
done

echo "[$(date)] === v23 noise floor DONE ===" >> "${VLOG}"
