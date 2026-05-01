#!/usr/bin/env bash
# v23 beyond-STS: eval v22c_a050 on a curated small subset of MTEB tasks
# beyond STS to find out whether the campaign's STS-tuned conclusions
# generalize. Picked to fit in ~30-45 min on a single GPU.
#
# Tasks (all small, English):
#   Classification: Banking77Classification (3.1K test), EmotionClassification
#                   (2K test), MTOPDomainClassification (4K test)
#   Retrieval:      SciFact (300 queries), NFCorpus (323 queries),
#                   ArguAna (1.4K queries)
#   Reranking:      SciDocsRR (1K), AskUbuntuDupQuestions (10K)
#   PairClass:      SprintDuplicateQuestions (101K pairs but fast)
#
# Output: artifacts/mteb/v22c_a050_beyond/summary.json
set -euo pipefail
REPO=/root/srt-adapter
PY=/root/srt_venv/bin/python
cd "${REPO}"
export PYTHONPATH=${REPO}
export HF_HOME=/root/hf_cache
export CUDA_VISIBLE_DEVICES=0

VLOG=${REPO}/artifacts/v23_beyond_chain.log
CKPT=${REPO}/checkpoints/adapter_v22c_a050/best_adapter.pt
TASKS="Banking77Classification,EmotionClassification,MTOPDomainClassification,SciFact,NFCorpus,ArguAna,SciDocsRR,AskUbuntuDupQuestions,SprintDuplicateQuestions"

echo "[$(date)] === v23 beyond-STS: v22c_a050 on ${TASKS} ===" >> "${VLOG}"

${PY} scripts/mteb_eval.py \
  --backbone Qwen/Qwen2.5-7B \
  --adapter "${CKPT}" \
  --output-dir ${REPO}/artifacts/mteb/v22c_a050_beyond \
  --task-names "${TASKS}" \
  --batch-size 16 --max-seq-len 256 --dtype bfloat16 >> "${VLOG}" 2>&1

echo "[$(date)] === v23 beyond-STS DONE ===" >> "${VLOG}"
