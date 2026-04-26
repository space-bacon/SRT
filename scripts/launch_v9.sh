#!/usr/bin/env bash
#
# v9 warm-start training run on A6000.
#
# Hypothesis: v8a established that, freed from the discrete prototype
# bottleneck, the encoder organizes Reddit subreddits AND an external
# 33-archetype taxonomy along a structured trajectory manifold (paper §5.9):
# archetype recall@1 = 0.230 at 7.6× chance using ONLY Reddit-subreddit
# supervision plus the geometric SupCon kernel. The natural follow-up:
# promote the archetype taxonomy itself from a held-out probe to a training
# signal and see whether the encoder's archetype geometry sharpens further.
#
# v9 changes from v8a (warm-starting from v8a step 10000 best_adapter.pt):
#   - Mix the 986 archetype-conditioned generations into the train loader
#     via WeightedRandomSampler (target ≈25% of rows per batch).
#   - Add a second SupCon loss keyed by archetype_id ∈ [1, 33] applied to
#     the same community.encoded vector that community_supcon already
#     operates on. Reddit rows carry archetype_id=-1 and are masked out.
#   - Keep community_supcon running at v8a strength (weight 2.0, temp 0.10)
#     so we don't lose the Reddit-side signal that was driving v8a's gains.
#   - Identical d_community=64, --no-prototypes, lr=5e-5, 10K steps.
#
# Goal: archetype recall@1 > 0.30 (from 0.230) and centroid off-diag cosine
# below 0.83 (from 0.873) without regressing Reddit retrieval below 0.46
# or the 2.7-bit val CE preservation.

set -euo pipefail

cd /root/srt-adapter

RESUME_FLAG=""
if [[ -f /root/srt-adapter/checkpoints/adapter_v9/training_checkpoint.pt ]]; then
    RESUME_FLAG="--resume /root/srt-adapter/checkpoints/adapter_v9/training_checkpoint.pt"
    echo "Resuming v9 from existing checkpoint."
else
    echo "Cold v9 start (warm-starting from v8a best_adapter.pt step 10000)."
fi

LOG=/root/srt-adapter/checkpoints/adapter_v9/adapter_v9_stdout.log
mkdir -p "$(dirname "$LOG")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python scripts/train.py \
    --backbone Qwen/Qwen2.5-7B \
    --train-data /root/srt-adapter/data/all_train.jsonl \
    --val-data   /root/srt-adapter/data/all_val.jsonl \
    --archetype-data /root/srt-adapter/artifacts/archetype_probe/generations.jsonl \
    --output-dir /root/srt-adapter/checkpoints/adapter_v9 \
    --warm-start /root/srt-adapter/checkpoints/adapter_v8a/best_adapter.pt \
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
    --community-supcon-weight 2.0 \
    --community-supcon-temperature 0.10 \
    --archetype-supcon-weight 2.0 \
    --archetype-supcon-temperature 0.10 \
    --archetype-mix-fraction 0.25 \
    --divergence-supcon-weight 0.3 \
    --listnet-weight 0.5 \
    --chain-residual-aux-weight 0.05 \
    $RESUME_FLAG \
    > "$LOG" 2>&1 &

echo "Launched v9 training, PID $!"
echo "Tail with:  tail -f $LOG"
