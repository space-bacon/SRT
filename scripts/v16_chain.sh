#!/usr/bin/env bash
# v16: continuation of v15a (SOTA mean STS 0.5250).
# v15a finished with LR fully cosine-decayed to 0 (peak 1e-4 -> 0 over 8628 steps,
# 1 epoch on 552K NLI rows). Loss was still bouncy 0.5-1.2 at the end and val
# r@1 only 0.2585 — the model didn't converge, it ran out of LR budget.
#
# v16 = warm v15a, second epoch, lower peak LR (3e-5 → 0 cosine), same corpus.
# Hypothesis: another epoch with reduced LR pushes mean STS above 0.5250 without
# adding any new corpora or modifying the recipe.
#
# Wall-clock: ~40 min train + ~3 min mteb_eval.

set -euo pipefail

REPO=/root/srt-adapter
PY=/root/srt_venv/bin/python
cd "${REPO}"
export PYTHONPATH=${REPO}
export HF_HOME=/root/hf_cache
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

NAME=v16
DATA_DIR=data/contrastive_v15a
OUT_DIR=${REPO}/checkpoints/adapter_${NAME}
HF_REPO=RiverRider/srt-adapter-${NAME}
WARM_PT=${REPO}/checkpoints/adapter_v15a/best_adapter.pt
VLOG=${REPO}/artifacts/${NAME}_chain.log

mkdir -p "${OUT_DIR}" "${REPO}/artifacts/mteb/${NAME}"

echo "[$(date)] === ${NAME}: starting watcher -> ${HF_REPO} ===" >> "${VLOG}"
nohup bash ${REPO}/scripts/hf_push_watcher.sh "${OUT_DIR}/best_adapter.pt" "${HF_REPO}" \
  >> "${REPO}/artifacts/${NAME}_watcher.log" 2>&1 &
WPID=$!
echo "[$(date)] watcher PID=${WPID}" >> "${VLOG}"

echo "[$(date)] === ${NAME}: DDP train (warm=${WARM_PT}, lr=3e-5, world=2) ===" >> "${VLOG}"
${PY} -m torch.distributed.run --standalone --nproc_per_node=2 \
  ${REPO}/scripts/train_contrastive_ddp.py \
  --backbone Qwen/Qwen2.5-7B \
  --warm-start "${WARM_PT}" \
  --train-data ${DATA_DIR}/train.jsonl \
  --val-data   ${DATA_DIR}/val.jsonl \
  --output-dir "${OUT_DIR}" \
  --batch-size 32 \
  --negatives-per-row 1 \
  --max-seq-len 128 \
  --temperature 0.05 \
  --lr 3e-5 \
  --warmup-steps 200 \
  --epochs 1 \
  --max-val-samples 2000 \
  --val-every 1000 \
  --log-every 100 \
  --grad-clip 1.0 \
  --num-workers 2 \
  --dtype bfloat16 >> "${VLOG}" 2>&1

echo "[$(date)] === ${NAME}: MTEB STS eval ===" >> "${VLOG}"
CUDA_VISIBLE_DEVICES=0 ${PY} scripts/mteb_eval.py \
  --backbone Qwen/Qwen2.5-7B \
  --adapter "${OUT_DIR}/best_adapter.pt" \
  --output-dir "${REPO}/artifacts/mteb/${NAME}" \
  --task-types STS \
  --task-langs eng \
  --batch-size 16 \
  --max-seq-len 256 \
  --dtype bfloat16 >> "${VLOG}" 2>&1

echo "[$(date)] === ${NAME}: push MTEB summary to HF ===" >> "${VLOG}"
${PY} - <<PYEOF >> "${VLOG}" 2>&1
import os
from huggingface_hub import HfApi
token = open("/root/.cache/huggingface/token").read().strip()
api = HfApi(token=token)
for fn in ["summary.json", "results.json", "per_task.csv"]:
    p = f"${REPO}/artifacts/mteb/${NAME}/{fn}"
    if os.path.exists(p):
        url = api.upload_file(
            path_or_fileobj=p,
            path_in_repo=f"mteb/{fn}",
            repo_id="${HF_REPO}",
            repo_type="model",
            commit_message=f"${NAME} MTEB STS results: {fn}",
        )
        print("pushed", url)
PYEOF

echo "[$(date)] === ${NAME}: killing watcher ${WPID} ===" >> "${VLOG}"
kill ${WPID} 2>/dev/null || true
echo "[$(date)] === ${NAME} DONE ===" >> "${VLOG}"
