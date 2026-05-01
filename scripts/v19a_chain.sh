#!/usr/bin/env bash
# v19a: CoSENT-tuned. Pure schedule fix on top of v18.
#
# v18 result: NEW SOTA (mean STS 0.3707, +0.0074 vs v15a) but dev Spearman
# PEAKED at step ~100 (0.6856) and DEGRADED to 0.5626 by step 500. The
# best-checkpoint saver caught the early peak, but the schedule was clearly
# over-long for this corpus + loss combo.
#
# v19a hypothesis: shorten the schedule + val more aggressively to land closer
# to the true peak. Same data, same loss, same warm-start.
#
# Changes vs v18:
#   - lr           1e-5  -> 5e-6   (gentler)
#   - epochs       5     -> 2      (~226 steps total instead of ~565)
#   - val-every    100   -> 25     (4x more checkpoints near peak)
#   - warmup-steps 50    -> 20     (proportional to shorter run)
#   - everything else identical (corpus, scale=20, batch=32, max-seq=128)
#
# Decision rule:
#   - v19a > v18  -> ship as new SOTA, then v19b = bigger corpus
#   - v19a ~ v18  -> schedule wasn't the cap; corpus is. Go straight to v19b.
#   - v19a < v18  -> lr=5e-6 too small to escape v15a basin in 226 steps. Retry
#                   v19a' with lr=1e-5, epochs=2, val_every=25.
#
# Wall-clock: ~5 min train + ~3 min mteb.

set -euo pipefail

REPO=/root/srt-adapter
PY=/root/srt_venv/bin/python
cd "${REPO}"
export PYTHONPATH=${REPO}
export HF_HOME=/root/hf_cache
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

NAME=v19a
DATA_DIR=${REPO}/data/supervised_sts_v17  # reuse v17/v18 corpus
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

# ─── 3. DDP CoSENT training (shortened schedule) ───
echo "[$(date)] === ${NAME}: DDP CoSENT train (warm=${WARM_PT}, lr=5e-6, scale=20, epochs=2, val_every=25) ===" >> "${VLOG}"
${PY} -m torch.distributed.run --standalone --nproc_per_node=2 \
  ${REPO}/scripts/train_supervised_sts_ddp.py \
  --backbone Qwen/Qwen2.5-7B \
  --warm-start "${WARM_PT}" \
  --train-data ${DATA_DIR}/train.jsonl \
  --dev-data   ${DATA_DIR}/dev.jsonl \
  --output-dir "${OUT_DIR}" \
  --batch-size 32 \
  --max-seq-len 128 \
  --lr 5e-6 \
  --warmup-steps 20 \
  --epochs 2 \
  --val-every 25 \
  --log-every 10 \
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
