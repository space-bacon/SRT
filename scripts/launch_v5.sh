#!/usr/bin/env bash
#
# v5 warm-start training run on A6000.
#
# Changes from v4:
#   1. SupCon now applied to community_head ENCODED (pre-mixing) output,
#      not VECTOR.  v4's vector = weights @ prototypes collapsed when the
#      assignment head went near-one-hot, killing SupCon's gradient by
#      symmetry (loss flatlined at log(B-1) = 2.71 for 6300 steps).
#   2. community_supcon_weight: 0.5 -> 2.0
#   3. --reset-community: drop community_head.* from warm-start so the
#      head re-initializes out of v4's degenerate basin.
#
# All v4 BEN/RRM gains preserved (warm-starting from v4_step6000 best).

set -euo pipefail

cd /root/srt-adapter

# Resume optimizer/scheduler if there was a previous v5 checkpoint
RESUME_FLAG=""
if [[ -f /root/srt-adapter/checkpoints/adapter_v5/training_checkpoint.pt ]]; then
    RESUME_FLAG="--resume /root/srt-adapter/checkpoints/adapter_v5/training_checkpoint.pt"
    echo "Resuming v5 from existing checkpoint."
else
    echo "Cold v5 start (warm-starting from v4 best_adapter.pt, resetting community_head)."
fi

LOG=/root/srt-adapter/checkpoints/adapter_v5/adapter_v5_stdout.log
mkdir -p "$(dirname "$LOG")"

# Avoid CUDA fragmentation OOMs on relaunch (carried over from v4).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python scripts/train.py \
    --backbone Qwen/Qwen2.5-7B \
    --train-data /root/srt-adapter/data/all_train.jsonl \
    --val-data   /root/srt-adapter/data/all_val.jsonl \
    --output-dir /root/srt-adapter/checkpoints/adapter_v5 \
    --warm-start /root/srt-adapter/checkpoints/adapter_v4/best_adapter.pt \
    --reset-community \
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

echo "Launched v5 training, PID $!"
echo "Tail with:  tail -f $LOG"
