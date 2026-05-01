#!/usr/bin/env bash
# v15 sequential chain on 2x Blackwell: v15a -> v15b -> v15c.
# Each variant: build corpus (if missing) -> DDP train (warm-start) ->
# MTEB STS eval -> push to HF Hub.
#
# v15a: NLI-only (~600K), no prefixes, warm-start v14
#       Goal: ablation showing whether MSMARCO+Quora helped or hurt curated tasks.
# v15b: NLI+MSMARCO+Quora (~700K), E5-style prefixes, warm-start v14
#       Goal: fix train/eval prefix mismatch (v12/v14 trained no-prefix but
#       evaluated with 'Represent this sentence for retrieval: ' prefix).
# v15c: NLI+MSMARCO+Quora+PubMedQA (~800K), E5-style prefixes, warm-start v15b
#       Goal: recover BIOSSES (v14 lost -0.041 vs v12) by adding biomedical
#       paired data.
#
# Total wall-clock estimate: ~6 hr (3 trains @ ~95min + 3 evals @ ~12min).
set -euo pipefail

REPO=/root/srt-adapter
PY=/root/srt_venv/bin/python
cd "${REPO}"
export PYTHONPATH=${REPO}
export HF_HOME=/root/hf_cache
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

LOG=${REPO}/artifacts/v15_chain.log
mkdir -p "${REPO}/artifacts"
echo "[$(date)] === v15 chain start ===" >> "${LOG}"

# ─────── helper: train + eval + push for one variant ───────
run_variant () {
  local NAME=$1        # e.g. v15a
  local DATA_DIR=$2    # e.g. data/contrastive_v15a
  local OUT_DIR=$3     # e.g. checkpoints/adapter_v15a
  local HF_REPO=$4     # e.g. RiverRider/srt-adapter-v15a
  local WARM_PT=$5     # warm-start path
  local QUERY_PROMPT=$6
  local PASSAGE_PROMPT=$7
  local VLOG=${REPO}/artifacts/${NAME}_chain.log

  mkdir -p "${OUT_DIR}" "${REPO}/artifacts/mteb/${NAME}"

  echo "[$(date)] === ${NAME}: starting watcher -> ${HF_REPO} ===" >> "${VLOG}"
  nohup bash ${REPO}/scripts/hf_push_watcher.sh "${OUT_DIR}/best_adapter.pt" "${HF_REPO}" \
    >> "${REPO}/artifacts/${NAME}_watcher.log" 2>&1 &
  local WPID=$!
  echo "[$(date)] watcher PID=${WPID}" >> "${VLOG}"

  echo "[$(date)] === ${NAME}: DDP train (warm=${WARM_PT}, world=2) ===" >> "${VLOG}"
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
    --lr 1e-4 \
    --warmup-steps 300 \
    --epochs 1 \
    --max-val-samples 2000 \
    --val-every 1000 \
    --log-every 100 \
    --grad-clip 1.0 \
    --num-workers 2 \
    --dtype bfloat16 >> "${VLOG}" 2>&1

  echo "[$(date)] === ${NAME}: MTEB STS eval (query='${QUERY_PROMPT}' passage='${PASSAGE_PROMPT}') ===" >> "${VLOG}"
  CUDA_VISIBLE_DEVICES=0 ${PY} scripts/mteb_eval.py \
    --backbone Qwen/Qwen2.5-7B \
    --adapter "${OUT_DIR}/best_adapter.pt" \
    --output-dir "${REPO}/artifacts/mteb/${NAME}" \
    --task-types STS \
    --task-langs eng \
    --batch-size 16 \
    --max-seq-len 256 \
    --query-prompt "${QUERY_PROMPT}" \
    --passage-prompt "${PASSAGE_PROMPT}" \
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
}

# ─────── corpora ───────
echo "[$(date)] building v15a corpus (NLI only)" >> "${LOG}"
if [ ! -s ${REPO}/data/contrastive_v15a/train.jsonl ]; then
  ${PY} scripts/build_contrastive_data.py \
    --output-dir data/contrastive_v15a \
    --include nli \
    --nli-limit 600000 \
    --negatives-per-row 1 \
    --val-fraction 0.01 \
    --prefix-style none >> "${LOG}" 2>&1
fi

echo "[$(date)] building v15b corpus (NLI+MSMARCO+Quora, e5 prefixes)" >> "${LOG}"
if [ ! -s ${REPO}/data/contrastive_v15b/train.jsonl ]; then
  ${PY} scripts/build_contrastive_data.py \
    --output-dir data/contrastive_v15b \
    --include nli,msmarco,quora \
    --nli-limit 300000 --msmarco-limit 300000 --quora-limit 100000 \
    --negatives-per-row 1 \
    --val-fraction 0.01 \
    --prefix-style e5 >> "${LOG}" 2>&1
fi

echo "[$(date)] building v15c corpus (v15b + pubmedqa, e5 prefixes)" >> "${LOG}"
if [ ! -s ${REPO}/data/contrastive_v15c/train.jsonl ]; then
  ${PY} scripts/build_contrastive_data.py \
    --output-dir data/contrastive_v15c \
    --include nli,msmarco,quora,pubmed \
    --nli-limit 300000 --msmarco-limit 300000 --quora-limit 100000 \
    --pubmed-limit 100000 \
    --negatives-per-row 1 \
    --val-fraction 0.01 \
    --prefix-style e5 >> "${LOG}" 2>&1
fi

# ─────── runs ───────
WARM_V14=${REPO}/checkpoints/adapter_v14/best_adapter.pt

echo "[$(date)] === starting v15a ===" >> "${LOG}"
run_variant v15a \
  data/contrastive_v15a \
  ${REPO}/checkpoints/adapter_v15a \
  RiverRider/srt-adapter-v15a \
  ${WARM_V14} \
  "Represent this sentence for retrieval: " ""

echo "[$(date)] === starting v15b ===" >> "${LOG}"
run_variant v15b \
  data/contrastive_v15b \
  ${REPO}/checkpoints/adapter_v15b \
  RiverRider/srt-adapter-v15b \
  ${WARM_V14} \
  "query: " "passage: "

echo "[$(date)] === starting v15c (warm-start v15b) ===" >> "${LOG}"
WARM_V15B=${REPO}/checkpoints/adapter_v15b/best_adapter.pt
run_variant v15c \
  data/contrastive_v15c \
  ${REPO}/checkpoints/adapter_v15c \
  RiverRider/srt-adapter-v15c \
  ${WARM_V15B} \
  "query: " "passage: "

echo "[$(date)] === v15 chain DONE ===" >> "${LOG}"
