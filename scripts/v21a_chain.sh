#!/usr/bin/env bash
# v21a: teacher distillation. Re-score v20 pair pool with mxbai-embed-large-v1
# cosines (continuous STS-aligned labels), then train CoSENT with v18 hyperparams.
#
# v20 lesson: NLI categorical labels {entail=1.0, neutral=0.5, contra=0.0}
# warped English STS. Replace those labels with TEACHER COSINES on the
# same pair pool to get continuous, STS-aligned signal.
#
# Recipe: identical to v18/v19b/v20 (proven hyperparameters)
#   - warm v15a, lr=1e-5, scale=20, batch=32, max-seq=128
#   - 2 epochs, val_every=500, log_every=100, warmup=200
# Wall-clock: ~10 min teacher rescoring + ~25 min train + ~16 min mteb.
#
# Decision rule:
#   - v21a > v18 on mean AND >= v18 on graded English (HUMESTS*, STSB*)
#       -> NEW SOTA, ship as RiverRider/srt-adapter-v21a, update production.
#   - v21a beats v18 on mean but loses some English (small)
#       -> ship as multilingual+graded sibling.
#   - v21a regresses
#       -> teacher distribution mismatch; pivot to a different teacher
#          (multilingual-e5-large-instruct) or different pair pool.

set -euo pipefail

REPO=/root/srt-adapter
PY=/root/srt_venv/bin/python
cd "${REPO}"
export PYTHONPATH=${REPO}
export HF_HOME=/root/hf_cache
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

NAME=v21a
SRC_DIR=${REPO}/data/supervised_sts_v20
DATA_DIR=${REPO}/data/supervised_sts_${NAME}
OUT_DIR=${REPO}/checkpoints/adapter_${NAME}
HF_REPO=RiverRider/srt-adapter-${NAME}
WARM_PT=${REPO}/checkpoints/adapter_v15a/best_adapter.pt
TEACHER=mixedbread-ai/mxbai-embed-large-v1
VLOG=${REPO}/artifacts/${NAME}_chain.log

mkdir -p "${OUT_DIR}" "${REPO}/artifacts/mteb/${NAME}"

# ─── 1. teacher rescoring (single GPU, fast) ───
if [ ! -s "${DATA_DIR}/train.jsonl" ]; then
  echo "[$(date)] === ${NAME}: teacher-rescoring v20 pool with ${TEACHER} ===" >> "${VLOG}"
  CUDA_VISIBLE_DEVICES=0 ${PY} scripts/build_v21a_corpus.py \
    --in-train ${SRC_DIR}/train.jsonl \
    --in-dev   ${SRC_DIR}/dev.jsonl \
    --output-dir "${DATA_DIR}" \
    --teacher "${TEACHER}" \
    --batch-size 128 \
    --dtype bfloat16 >> "${VLOG}" 2>&1
fi

# ─── 2. start HF push watcher ───
echo "[$(date)] === ${NAME}: starting watcher -> ${HF_REPO} ===" >> "${VLOG}"
nohup bash ${REPO}/scripts/hf_push_watcher.sh "${OUT_DIR}/best_adapter.pt" "${HF_REPO}" \
  >> "${REPO}/artifacts/${NAME}_watcher.log" 2>&1 &
WPID=$!
echo "[$(date)] watcher PID=${WPID}" >> "${VLOG}"

# ─── 3. DDP CoSENT training ───
echo "[$(date)] === ${NAME}: DDP CoSENT train (warm=v15a, lr=1e-5, scale=20, 2 ep over teacher-rescored ~207K pairs) ===" >> "${VLOG}"
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
