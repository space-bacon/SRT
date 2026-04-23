#!/usr/bin/env bash
#
# v6 warm-start training run on A6000.
#
# Changes from v5 (warm-starting from v5_step17000/best_adapter.pt):
#   1. divergence_supcon_weight: 0 -> 1.0
#      Mean-pool the last MAH layer's divergence per sample, contrast by
#      community id. Analog of the v5 community-SupCon fix applied to the
#      metapragmatic channel. Should sharpen between-community geometry of
#      the divergence vectors, currently only constrained by the chain loss.
#   2. listnet_weight: 0 -> 0.5
#      ListNet ranking loss on r̂ within each sequence. Pointwise smooth-L1
#      tolerates large rank errors at the tails (where supercritical mass
#      lives). ListNet pushes the *ordering* of r̂ to match r_true, which is
#      what every downstream consumer (top-k probes, heatmaps, percentile
#      thresholds) actually relies on.
#   3. chain_residual_aux_weight: 0 -> 0.05
#      Tiny auxiliary penalty pulling per-token chain residual toward 0.5,
#      so the now-exposed chain_residual_per_token channel stays informative
#      for inference probes (hallucination, etc) instead of collapsing to
#      ~0 everywhere.
#
# All v5 community-SupCon gains preserved (community_supcon_weight=2.0).
# No architectural changes (no new parameters in the adapter).

set -euo pipefail

cd /root/srt-adapter

RESUME_FLAG=""
if [[ -f /root/srt-adapter/checkpoints/adapter_v6/training_checkpoint.pt ]]; then
    RESUME_FLAG="--resume /root/srt-adapter/checkpoints/adapter_v6/training_checkpoint.pt"
    echo "Resuming v6 from existing checkpoint."
else
    echo "Cold v6 start (warm-starting from v5 best_adapter.pt)."
fi

LOG=/root/srt-adapter/checkpoints/adapter_v6/adapter_v6_stdout.log
mkdir -p "$(dirname "$LOG")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python scripts/train.py \
    --backbone Qwen/Qwen2.5-7B \
    --train-data /root/srt-adapter/data/all_train.jsonl \
    --val-data   /root/srt-adapter/data/all_val.jsonl \
    --output-dir /root/srt-adapter/checkpoints/adapter_v6 \
    --warm-start /root/srt-adapter/checkpoints/adapter_v5/best_adapter.pt \
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

echo "Launched v6 training, PID $!"
echo "Tail with:  tail -f $LOG"
