#!/usr/bin/env bash
#
# v7 warm-start training run on A6000.
#
# Hypothesis: v6's divergence_supcon_weight=1.0 over-sharpened community
# embeddings, recovering recall@1 (0.36 -> 0.41) and tightening the
# divergence geometry but collapsing the v5 bedrock/battleground decoding
# split (factual prompts now disagree across communities the way charged
# prompts used to).
#
# v7 changes from v6 (warm-starting from v6 step 12000 best_adapter.pt):
#   - divergence_supcon_weight: 1.0 -> 0.3
#     Keep the channel active (preserve some geometry gains) but stop
#     forcing community-conditional divergence everywhere.
#   - listnet_weight: 0.5 (unchanged, helped calibration)
#   - chain_residual_aux_weight: 0.05 (unchanged)
#
# Goal: recover the v5 counterfactual contrast on factual prompts while
# retaining v6's improved hallucination AUROCs and ECE.
#
# Short run: 10K steps (half of v6) since we are tuning, not relearning.

set -euo pipefail

cd /root/srt-adapter

RESUME_FLAG=""
if [[ -f /root/srt-adapter/checkpoints/adapter_v7/training_checkpoint.pt ]]; then
    RESUME_FLAG="--resume /root/srt-adapter/checkpoints/adapter_v7/training_checkpoint.pt"
    echo "Resuming v7 from existing checkpoint."
else
    echo "Cold v7 start (warm-starting from v6 best_adapter.pt step 12000)."
fi

LOG=/root/srt-adapter/checkpoints/adapter_v7/adapter_v7_stdout.log
mkdir -p "$(dirname "$LOG")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python scripts/train.py \
    --backbone Qwen/Qwen2.5-7B \
    --train-data /root/srt-adapter/data/all_train.jsonl \
    --val-data   /root/srt-adapter/data/all_val.jsonl \
    --output-dir /root/srt-adapter/checkpoints/adapter_v7 \
    --warm-start /root/srt-adapter/checkpoints/adapter_v6/best_adapter.pt \
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

echo "Launched v7 training, PID $!"
echo "Tail with:  tail -f $LOG"
