#!/usr/bin/env bash
# v20: CoSENT on STSB + GRADED NLI (pair-class).
#
# v19b lesson: 200K binary {1.0, 0.0} NLI pairs flattened the per-batch
# score distribution to bimodal -> graded English STS regressed even
# though cross-lingual got better.
#
# v20 fix: same ~200K NLI pairs but use the `pair-class` config and map
#   entailment    (0) -> 1.0
#   neutral       (1) -> 0.5    <-- the missing intermediate
#   contradiction (2) -> 0.0
# Now every batch carries graded mass at {0, 0.5, 1} (plus STSB's
# continuous spread). CoSENT pairwise margins remain meaningful between
# every pair of distinct scores within a batch.
#
# Recipe: identical to v19b (proven hyperparameters + shape)
#   - warm v15a, lr=1e-5, scale=20, batch=32, max-seq=128, 2 epochs
#   - val_every=500, log_every=100, warmup=200
# Expected wall-clock: ~25 min train + ~3 min mteb.
#
# Decision rule:
#   - v20 > v18 on mean AND >= v18 on graded English (HUMESTS*, STSB*, STS13/14/15)
#       -> NEW SOTA, ship and update production.
#   - v20 ~= v19b on mean but better graded English
#       -> directional win, label as multilingual+graded sibling, keep v18.
#   - v20 << v18 on graded English
#       -> NLI scores too coarse / mismatched with STSB; pivot to teacher
#          distillation (v21 = mxbai-embed-large-v1 cosines as continuous labels).

set -euo pipefail

REPO=/root/srt-adapter
PY=/root/srt_venv/bin/python
cd "${REPO}"
export PYTHONPATH=${REPO}
export HF_HOME=/root/hf_cache
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

NAME=v20
DATA_DIR=${REPO}/data/supervised_sts_v20
OUT_DIR=${REPO}/checkpoints/adapter_${NAME}
HF_REPO=RiverRider/srt-adapter-${NAME}
WARM_PT=${REPO}/checkpoints/adapter_v15a/best_adapter.pt
VLOG=${REPO}/artifacts/${NAME}_chain.log

mkdir -p "${OUT_DIR}" "${REPO}/artifacts/mteb/${NAME}"

# ─── 1. build corpus if missing ───
if [ ! -s "${DATA_DIR}/train.jsonl" ]; then
  echo "[$(date)] === ${NAME}: building corpus -> ${DATA_DIR} ===" >> "${VLOG}"
  ${PY} scripts/build_v20_corpus.py \
    --output-dir "${DATA_DIR}" \
    --nli-rows 200000 >> "${VLOG}" 2>&1
fi

# ─── 2. start HF push watcher ───
echo "[$(date)] === ${NAME}: starting watcher -> ${HF_REPO} ===" >> "${VLOG}"
nohup bash ${REPO}/scripts/hf_push_watcher.sh "${OUT_DIR}/best_adapter.pt" "${HF_REPO}" \
  >> "${REPO}/artifacts/${NAME}_watcher.log" 2>&1 &
WPID=$!
echo "[$(date)] watcher PID=${WPID}" >> "${VLOG}"

# ─── 3. DDP CoSENT training ───
echo "[$(date)] === ${NAME}: DDP CoSENT train (warm=${WARM_PT}, lr=1e-5, scale=20, 2 ep over ~207K graded pairs) ===" >> "${VLOG}"
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
