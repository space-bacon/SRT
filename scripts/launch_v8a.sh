#!/usr/bin/env bash
#
# v8a warm-start training run on A6000.
#
# Hypothesis: the v7 PCA finding shows the community prototypes barely move
# from random initialization (mean abs delta v5->v6 ~3e-5 against magnitudes
# of 0.5-1.5). The encoder does the discriminative work; the soft-argmax
# over K random anchors discards information at readout. Removing the
# discrete prototype basis entirely (--no-prototypes) should preserve or
# improve archetype recall while exposing a continuous community coordinate.
#
# v8a changes from v7 (warm-starting from v7 step 6000 best_adapter.pt):
#   - --no-prototypes : community_head emits encoder output directly.
#     Drops community_entropy_loss (no soft assignment to entropize).
#     community_supcon_loss continues to operate on `encoded` exactly as
#     before -- this was always the loss doing the work.
#   - All other hyperparameters identical to v7 (10K steps, lr 5e-5,
#     div_supcon=0.3, listnet=0.5, chain_aux=0.05).
#
# The v7 prototype tensor in the warm-start checkpoint will be reported as
# an "unexpected key" by load_state_dict(strict=False) and simply dropped.
#
# Goal: confirm or refute that the prototype basis was architectural debt.
# Success criterion: archetype recall@1 >= 0.15 (matches v7) on the same
# 986-generation probe; trajectory metrics (path length, log-det covariance)
# show structure absent in v7.

set -euo pipefail

cd /root/srt-adapter

RESUME_FLAG=""
if [[ -f /root/srt-adapter/checkpoints/adapter_v8a/training_checkpoint.pt ]]; then
    RESUME_FLAG="--resume /root/srt-adapter/checkpoints/adapter_v8a/training_checkpoint.pt"
    echo "Resuming v8a from existing checkpoint."
else
    echo "Cold v8a start (warm-starting from v7 best_adapter.pt step 6000)."
fi

LOG=/root/srt-adapter/checkpoints/adapter_v8a/adapter_v8a_stdout.log
mkdir -p "$(dirname "$LOG")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python scripts/train.py \
    --backbone Qwen/Qwen2.5-7B \
    --train-data /root/srt-adapter/data/all_train.jsonl \
    --val-data   /root/srt-adapter/data/all_val.jsonl \
    --output-dir /root/srt-adapter/checkpoints/adapter_v8a \
    --warm-start /root/srt-adapter/checkpoints/adapter_v7/best_adapter.pt \
    --no-prototypes \
    --batch-size 16 \
    --epochs 1 \
    --max-train-samples 160000 \
    --lr 5e-5 \
    --warmup-steps 250 \
    --max-seq-len 512 \
    --val-every 1000 \
    --max-val-samples 5000 \
    --log-every 100 \
    --grad-clip 1.0 \
    --dtype bfloat16 \
    --divergence-supcon-weight 0.3 \
    --listnet-weight 0.5 \
    --chain-residual-aux-weight 0.05 \
    $RESUME_FLAG \
    > "$LOG" 2>&1 &

echo "Launched v8a training, PID $!"
echo "Tail with:  tail -f $LOG"
