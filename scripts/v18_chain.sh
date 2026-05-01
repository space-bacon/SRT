#!/usr/bin/env bash
# v18: CoSENT (pairwise rank loss) on labeled STS pairs.
#
# Hypothesis: v17's failure was the loss FUNCTION, not the data size.
#   - MTEB STS metric is Spearman rank correlation.
#   - v17 used MSE on absolute cosine values -- a Pearson-aligned objective.
#   - That's a misalignment: rank-preserving distortions of cos lose 0 Spearman
#     but ANY MSE.  The optimizer therefore burns capacity matching absolute
#     STSB scores (which are noisy human-labeled values in a narrow domain),
#     producing the dev-Spearman 0.6936 / test-Spearman 0.5613 = 0.13 overfit
#     gap we observed.
#
# CoSENT is the de-facto sentence-transformers loss for STS:
#     L = log(1 + sum_{(i,j): s_i > s_j} exp(scale * (cos_j - cos_i)))
# It directly optimizes pairwise ranking -- the thing Spearman measures.
# Same data, same warm-start, same schedule as v17. Pure A/B on loss function.
#
# Decision rule:
#   - If v18 beats v15a   -> ship as new SOTA.
#   - If v18 ~= v17       -> loss wasn't the bottleneck; corpus IS too narrow.
#                            Next step: v19 with bigger labeled corpus or distill.
#   - If v18 << v17       -> CoSENT mis-tuned (scale/LR), retry with smaller scale.
#
# Wall-clock estimate: ~10 min train + ~3 min mteb + push.

set -euo pipefail

REPO=/root/srt-adapter
PY=/root/srt_venv/bin/python
cd "${REPO}"
export PYTHONPATH=${REPO}
export HF_HOME=/root/hf_cache
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

NAME=v18
DATA_DIR=${REPO}/data/supervised_sts_v17  # reuse v17 corpus (same intent)
OUT_DIR=${REPO}/checkpoints/adapter_${NAME}
HF_REPO=RiverRider/srt-adapter-${NAME}
WARM_PT=${REPO}/checkpoints/adapter_v15a/best_adapter.pt
VLOG=${REPO}/artifacts/${NAME}_chain.log

mkdir -p "${OUT_DIR}" "${REPO}/artifacts/mteb/${NAME}"

# ─── 1. corpus must already exist from v17 ───
if [ ! -s "${DATA_DIR}/train.jsonl" ]; then
  echo "[$(date)] === ${NAME}: building supervised STS data -> ${DATA_DIR} ===" >> "${VLOG}"
  ${PY} scripts/build_supervised_sts_data.py --output-dir "${DATA_DIR}" >> "${VLOG}" 2>&1
fi

# ─── 2. start HF push watcher ───
echo "[$(date)] === ${NAME}: starting watcher -> ${HF_REPO} ===" >> "${VLOG}"
nohup bash ${REPO}/scripts/hf_push_watcher.sh "${OUT_DIR}/best_adapter.pt" "${HF_REPO}" \
  >> "${REPO}/artifacts/${NAME}_watcher.log" 2>&1 &
WPID=$!
echo "[$(date)] watcher PID=${WPID}" >> "${VLOG}"

# ─── 3. DDP CoSENT training ───
echo "[$(date)] === ${NAME}: DDP CoSENT train (warm=${WARM_PT}, lr=1e-5, scale=20, world=2) ===" >> "${VLOG}"
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
  --warmup-steps 50 \
  --epochs 5 \
  --val-every 100 \
  --log-every 20 \
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

# ─── 5. push to HF Hub ───
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
