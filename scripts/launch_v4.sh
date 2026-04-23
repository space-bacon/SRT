#!/usr/bin/env bash
#
# v4 warm-start training run on A6000.
#
# Architectural changes from v3 (warm-started from step94k/best_adapter.pt):
#   1. BEN.r_head: nn.Tanh() removed   →  unbounded output (was capped ±1)
#   2. RRM.inject: linear+sigmoid-gate  →  FiLM (gamma + beta) modulation
#   3. inject_reg_weight: 0.5 → 0.0     (was forcing arbitrary directions)
#   4. NEW: community_supcon_weight=0.5 (forces prototypes apart)
#
# v3 weights for community_head, mah_heads, ben.r_head[Linear], ben.regime_head,
# rrm.gru, chain_predictor are all loaded.  RRM.gamma_proj/beta_proj are
# fresh-initialized (FiLM is identity at init, then learns).
#
# Plan:  ~20K steps @ lr 1e-4 with linear warmup → cosine decay.
#        Polishing run, not bootstrap.  ~10 hours wall time.

set -euo pipefail

cd /root/srt-adapter

# Resume optimizer/scheduler if there was a previous v4 checkpoint
RESUME_FLAG=""
if [[ -f /root/srt-adapter/checkpoints/adapter_v4/training_checkpoint.pt ]]; then
    RESUME_FLAG="--resume /root/srt-adapter/checkpoints/adapter_v4/training_checkpoint.pt"
    echo "Resuming v4 from existing checkpoint."
else
    echo "Cold v4 start (warm-starting from v3 best_adapter.pt)."
fi

LOG=/root/srt-adapter/checkpoints/adapter_v4/adapter_v4_stdout.log
mkdir -p "$(dirname "$LOG")"

# Avoid CUDA fragmentation OOMs on relaunch (seen 2026-04-22 14:56).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python scripts/train.py \
    --backbone Qwen/Qwen2.5-7B \
    --train-data /root/srt-adapter/data/all_train.jsonl \
    --val-data   /root/srt-adapter/data/all_val.jsonl \
    --output-dir /root/srt-adapter/checkpoints/adapter_v4 \
    --warm-start /root/srt-adapter/checkpoints/adapter_v3/best_adapter.pt \
    --batch-size 16 \
    --epochs 1 \
    --max-train-samples 320000 \
    --lr 1e-4 \
    --warmup-steps 500 \
    --max-seq-len 512 \
    --val-every 1000 \
    --max-val-samples 5000 \
    --log-every 100 \
    --grad-clip 1.0 \
    --dtype bfloat16 \
    $RESUME_FLAG \
    > "$LOG" 2>&1 &

echo "Launched v4 training, PID $!"
echo "Tail with:  tail -f $LOG"
