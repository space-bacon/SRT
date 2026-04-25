#!/usr/bin/env bash
#
# v8b warm-start training run on A6000.
#
# Hypothesis: v8a removed the prototype bottleneck and lifted Reddit
# recall@1 0.413 -> 0.484 and archetype recall@1 0.149 -> 0.230. But
# v8a's archetype centroid off-diagonal cosine is still 0.873 (highly
# correlated) and trajectory anisotropy is 23,333 (one direction
# dominates). The encoder is doing the work but is collapsing onto a
# narrow manifold. SupCon is now the sole discriminative pressure
# (no entropy reg, no prototype mixing) -- so push it harder.
#
# v8b changes from v8a (warm-starting from v8a step 10000):
#   - community_supcon_weight: 2.0 -> 4.0  (double the contrastive pull)
#   - community_supcon_temperature: 0.1 -> 0.05  (sharpen contrast)
#   - everything else identical to v8a
#
# d_community stays 64 so the v8a encoder weights warm-start cleanly
# (changing d_community would break encoder shape and force cold start).
#
# Success criteria (vs v8a):
#   - val CE preserved (no regression on token loss)
#   - Reddit recall@1 >= 0.484
#   - archetype off-diag cos < 0.85 (centroids more orthogonal)
#   - archetype recall@1 >= 0.23
# Failure mode to watch for:
#   - SupCon overpowers ListNet -> r_hat calibration degrades
#   - sharper temperature -> training instability / loss explodes

set -euo pipefail

cd /root/srt-adapter

RESUME_FLAG=""
if [[ -f /root/srt-adapter/checkpoints/adapter_v8b/training_checkpoint.pt ]]; then
    RESUME_FLAG="--resume /root/srt-adapter/checkpoints/adapter_v8b/training_checkpoint.pt"
    echo "Resuming v8b from existing checkpoint."
else
    echo "Cold v8b start (warm-starting from v8a best_adapter.pt step 10000)."
fi

LOG=/root/srt-adapter/checkpoints/adapter_v8b/adapter_v8b_stdout.log
mkdir -p "$(dirname "$LOG")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python scripts/train.py \
    --backbone Qwen/Qwen2.5-7B \
    --train-data /root/srt-adapter/data/all_train.jsonl \
    --val-data   /root/srt-adapter/data/all_val.jsonl \
    --output-dir /root/srt-adapter/checkpoints/adapter_v8b \
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
    --divergence-supcon-weight 0.3 \
    --listnet-weight 0.5 \
    --chain-residual-aux-weight 0.05 \
    --community-supcon-weight 4.0 \
    --community-supcon-temperature 0.05 \
    $RESUME_FLAG \
    > "$LOG" 2>&1 &

echo "Launched v8b training, PID $!"
echo "Tail with:  tail -f $LOG"
