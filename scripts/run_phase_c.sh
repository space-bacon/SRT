#!/usr/bin/env bash
# Phase C: BoN curation -> SFT distillation.
#
# Step 1 (curate): Sample K=32 candidates per target from warm-start AV,
#                  score by AR.reconstruct mse_nrm, keep top-1.
# Step 2 (sft):    Cross-entropy on gold tokens conditioned on target vec.
#
# Rationale: REINFORCE plateaus at fve_nrm=0.62 (N1i-v2). Soft bridge
# (N2-v1..v3) introduced bias via continuous-embedding reconstruction
# that doesn't correspond to any one-hot token sequence; val crashed.
# BoN curates *verified* discrete gold; SFT teaches AV to emit them.
set -euo pipefail

cd "$(dirname "$0")/.."
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}

ITER="${ITER:-1}"
WARM="${WARM:-artifacts/nla/n1i_v2_best/av_step002500.pt}"
TARGETS="${TARGETS:-artifacts/nla/targets_q7b_L20_10k.pt}"
K="${K:-32}"
LIMIT="${LIMIT:-}"

CURATED="artifacts/nla/curated/bon_iter${ITER}.jsonl"
SFT_OUT="artifacts/nla/sft_iter${ITER}"
mkdir -p "$(dirname $CURATED)" "$SFT_OUT"

echo "=== Phase C iter $ITER ==="
echo "warm-start: $WARM"
echo "targets:    $TARGETS"
echo "curated:    $CURATED"
echo "sft out:    $SFT_OUT"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then LIMIT_ARG="--limit $LIMIT"; fi

python scripts/curate_bon.py \
    --targets "$TARGETS" \
    --init-from "$WARM" \
    --k "$K" --batch-size 4 \
    --temperature 1.2 --top-p 0.95 \
    --max-new-tokens 64 \
    $LIMIT_ARG \
    --out "$CURATED"

python scripts/train_nla_sft.py \
    --targets "$TARGETS" \
    --pairs "$CURATED" \
    --init-from "$WARM" \
    --epochs 3 --batch-size 16 --lr 3e-5 \
    --val-fraction 0.05 --val-every 500 --val-vectors 512 \
    --out "$SFT_OUT"
