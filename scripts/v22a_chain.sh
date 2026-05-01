#!/usr/bin/env bash
# v22a: STSB-anchored teacher distillation.
# Same recipe as v21a but with STSB pairs upsampled 4x in the training
# pool to recover the English STS calibration that v21a partially lost.
#
# Hypothesis: 200K NLI-pair-rescored cosines flooded the v21a batches and
# crowded out the 7K graded STSB rows. Upsampling STSB to ~12% of pool
# should recover HUMESTSBenchmark / STSBenchmark / STS13/14 without
# sacrificing the cross-lingual gains (which come from the diverse pool).
#
# Recipe: identical to v21a (lr=1e-5, scale=20, batch=32, max-seq=128, 2 ep).
# Wall-clock: ~2 min corpus build + ~30 min train + ~16 min mteb.
set -euo pipefail

REPO=/root/srt-adapter
PY=/root/srt_venv/bin/python
cd "${REPO}"
export PYTHONPATH=${REPO}
export HF_HOME=/root/hf_cache
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

NAME=v22a
SRC_DIR=${REPO}/data/supervised_sts_v21a
DATA_DIR=${REPO}/data/supervised_sts_${NAME}
OUT_DIR=${REPO}/checkpoints/adapter_${NAME}
HF_REPO=RiverRider/srt-adapter-${NAME}
WARM_PT=${REPO}/checkpoints/adapter_v15a/best_adapter.pt
VLOG=${REPO}/artifacts/${NAME}_chain.log

mkdir -p "${OUT_DIR}" "${REPO}/artifacts/mteb/${NAME}"

# ─── 1. corpus (just upsamples STSB; teacher cosines already in v21a pool) ───
if [ ! -s "${DATA_DIR}/train.jsonl" ]; then
  echo "[$(date)] === ${NAME}: building corpus (4x STSB upsample of v21a pool) ===" >> "${VLOG}"
  ${PY} scripts/build_v22a_corpus.py \
    --in-train ${SRC_DIR}/train.jsonl \
    --in-dev   ${SRC_DIR}/dev.jsonl \
    --output-dir "${DATA_DIR}" \
    --stsb-upsample 4 >> "${VLOG}" 2>&1
fi

# ─── 2. start HF push watcher ───
echo "[$(date)] === ${NAME}: starting watcher -> ${HF_REPO} ===" >> "${VLOG}"
nohup bash ${REPO}/scripts/hf_push_watcher.sh "${OUT_DIR}/best_adapter.pt" "${HF_REPO}" \
  >> "${REPO}/artifacts/${NAME}_watcher.log" 2>&1 &
WPID=$!
echo "[$(date)] watcher PID=${WPID}" >> "${VLOG}"

# ─── 3. DDP CoSENT training ───
echo "[$(date)] === ${NAME}: DDP CoSENT (warm=v15a, lr=1e-5, scale=20, 2 ep) ===" >> "${VLOG}"
${PY} -m torch.distributed.run --standalone --nproc_per_node=2 \
  ${REPO}/scripts/train_supervised_sts_ddp.py \
  --backbone Qwen/Qwen2.5-7B \
  --warm-start "${WARM_PT}" \
  --train-data ${DATA_DIR}/train.jsonl \
  --dev-data   ${DATA_DIR}/dev.jsonl \
  --output-dir "${OUT_DIR}" \
  --batch-size 32 \
  --max-seq-len 128 \
  --lr 1e-5 \
  --warmup-steps 200 \
  --epochs 2 \
  --val-every 500 \
  --log-every 100 \
  --grad-clip 1.0 \
  --num-workers 2 \
  --dtype bfloat16 \
  --loss cosent \
  --cosent-scale 20.0 >> "${VLOG}" 2>&1

# ─── 4. MTEB STS eval ───
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

# ─── 5. push MTEB results to HF Hub ───
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
