#!/usr/bin/env bash
# v17: supervised STS regression on labeled pairs.
#
# Hypothesis: v15a (mean STS 0.5250) is at the InfoNCE/NLI ceiling on this
# adapter+backbone (v16 confirmed: another epoch produced -0.002 noise, val
# r@1 was flat at 0.252 throughout). The InfoNCE objective optimizes a
# discrimination proxy; MTEB STS measures Spearman of cosine vs human score.
# Switch loss to direct cosine-MSE on labeled pairs to optimize the metric.
#
# Recipe:
# - Warm-start: v15a best_adapter.pt
# - Train data: STSBenchmark train + dev + SICK-R train  (~12K labeled pairs)
#   * STSBenchmark dev included in train for supervision (selection is on
#     Spearman rank correlation, which generalizes from rank-preserving fits)
#   * stsb_multi_mt explicitly excluded -- would contaminate
#     STSBenchmarkMultilingualSTS (which uses dev+test of stsb_multi_mt)
# - Loss: MSE between cos(emb_s1, emb_s2) and score in [0,1]
# - LR: 1e-5 cosine, 50 warmup, 5 epochs (very short, ~900 steps)
# - Best-ckpt: STSBenchmark dev Spearman
# - Eval: same MTEB STS suite as v12-v16 for apples-to-apples comparison
#
# Wall-clock estimate: ~3 min data build + ~10 min train + ~3 min mteb + push.

set -euo pipefail

REPO=/root/srt-adapter
PY=/root/srt_venv/bin/python
cd "${REPO}"
export PYTHONPATH=${REPO}
export HF_HOME=/root/hf_cache
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

NAME=v17
DATA_DIR=${REPO}/data/supervised_sts_v17
OUT_DIR=${REPO}/checkpoints/adapter_${NAME}
HF_REPO=RiverRider/srt-adapter-${NAME}
WARM_PT=${REPO}/checkpoints/adapter_v15a/best_adapter.pt
VLOG=${REPO}/artifacts/${NAME}_chain.log

mkdir -p "${OUT_DIR}" "${REPO}/artifacts/mteb/${NAME}"

# ─── 1. build supervised STS corpus ───
if [ ! -s "${DATA_DIR}/train.jsonl" ]; then
  echo "[$(date)] === ${NAME}: build supervised STS data -> ${DATA_DIR} ===" >> "${VLOG}"
  ${PY} scripts/build_supervised_sts_data.py \
    --output-dir "${DATA_DIR}" >> "${VLOG}" 2>&1
fi

# ─── 2. start HF push watcher ───
echo "[$(date)] === ${NAME}: starting watcher -> ${HF_REPO} ===" >> "${VLOG}"
nohup bash ${REPO}/scripts/hf_push_watcher.sh "${OUT_DIR}/best_adapter.pt" "${HF_REPO}" \
  >> "${REPO}/artifacts/${NAME}_watcher.log" 2>&1 &
WPID=$!
echo "[$(date)] watcher PID=${WPID}" >> "${VLOG}"

# ─── 3. DDP supervised training ───
echo "[$(date)] === ${NAME}: DDP supervised STS train (warm=${WARM_PT}, lr=1e-5, world=2) ===" >> "${VLOG}"
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
  --dtype bfloat16 >> "${VLOG}" 2>&1

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
