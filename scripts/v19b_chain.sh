#!/usr/bin/env bash
# v19b: CoSENT on enlarged labeled-pair corpus (STSB + 100K all-NLI triplets).
#
# Hypothesis: v19a proved that v18's hyperparameters were correct -- the
# adapter cannot move further without more / more diverse labeled pairs.
# v18's STSB-only corpus (5K) saturated; the 1500-row dev set saturated.
#
# v19b corpus:
#   - STSB train + dev (graded, in-distribution anchor)   ~7K pairs
#   - all-NLI triplets (100K rows -> 200K binary pairs)   ~200K pairs
#   Total: ~207K pairs, varied scores per batch -> CoSENT gradient on
#   every (pos_i, neg_j) pair within a batch.
#
# Recipe: v18 hyperparameters (NOT v19a's). lr=1e-5, scale=20, batch=32,
# max-seq=128, dtype=bf16. Corpus is ~30x larger than v18 so we run
# 2 epochs (~13K steps) instead of 5 epochs of the small set. Wall-clock
# estimate: ~25 min train + ~3 min mteb.
#
# Decision rule:
#   - v19b > v18      -> ship as new SOTA, plan v20 with even more / multilingual
#   - v19b ~= v18     -> NLI pairs add nothing the v15a NLI-InfoNCE warm-start
#                        didn't already encode. Pivot to teacher distillation.
#   - v19b << v18     -> binary 0/1 pairs flatten the score distribution and
#                        hurt graded-STS calibration. Retry with graded pairs
#                        only (e.g. mining cosine sims from a teacher model).

set -euo pipefail

REPO=/root/srt-adapter
PY=/root/srt_venv/bin/python
cd "${REPO}"
export PYTHONPATH=${REPO}
export HF_HOME=/root/hf_cache
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

NAME=v19b
DATA_DIR=${REPO}/data/supervised_sts_v19b
OUT_DIR=${REPO}/checkpoints/adapter_${NAME}
HF_REPO=RiverRider/srt-adapter-${NAME}
WARM_PT=${REPO}/checkpoints/adapter_v15a/best_adapter.pt
VLOG=${REPO}/artifacts/${NAME}_chain.log

mkdir -p "${OUT_DIR}" "${REPO}/artifacts/mteb/${NAME}"

# ─── 1. build corpus if missing ───
if [ ! -s "${DATA_DIR}/train.jsonl" ]; then
  echo "[$(date)] === ${NAME}: building corpus -> ${DATA_DIR} ===" >> "${VLOG}"
  ${PY} scripts/build_v19b_corpus.py \
    --output-dir "${DATA_DIR}" \
    --nli-triplets 100000 >> "${VLOG}" 2>&1
fi

# ─── 2. start HF push watcher ───
echo "[$(date)] === ${NAME}: starting watcher -> ${HF_REPO} ===" >> "${VLOG}"
nohup bash ${REPO}/scripts/hf_push_watcher.sh "${OUT_DIR}/best_adapter.pt" "${HF_REPO}" \
  >> "${REPO}/artifacts/${NAME}_watcher.log" 2>&1 &
WPID=$!
echo "[$(date)] watcher PID=${WPID}" >> "${VLOG}"

# ─── 3. DDP CoSENT training (v18 hyperparameters, larger corpus, 2 epochs) ───
echo "[$(date)] === ${NAME}: DDP CoSENT train (warm=${WARM_PT}, lr=1e-5, scale=20, 2 ep over ~207K pairs) ===" >> "${VLOG}"
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
