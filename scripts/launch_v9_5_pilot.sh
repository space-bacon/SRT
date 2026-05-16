#!/usr/bin/env bash
#
# v9.5 pilot: 1K-step finetune from adapter_v9/best_adapter.pt with
# inject-site SupCon enabled (weight 1.0, temp 0.1).
#
# Hypothesis: the v9 separatrix forcing test under unit-norm centroids
# showed encoder-side archetype-supcon does NOT propagate to a directional
# inject-site signal the LM head can decode. This pilot adds SupCon
# directly on the inject vectors (mean-pooled, both inject layers
# concatenated) to test whether the inject pathway can be shaped by the
# loss. If forcing direction moves on the separatrix probe under unit-norm
# centroids after this pilot, the full v10 with inject-supcon is justified.
#
# Cost: ~30-45min on A6000.

set -euo pipefail

cd /root/srt-adapter

OUT=/root/srt-adapter/checkpoints/adapter_v9_5_pilot
mkdir -p "$OUT"
LOG=$OUT/train_stdout.log

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/root/srt-adapter
export HF_HOME=/root/hf_cache
export HF_HUB_OFFLINE=1

nohup /root/srt_venv/bin/python scripts/train.py \
    --backbone Qwen/Qwen2.5-7B \
    --train-data /root/srt-adapter/data/all_train.jsonl \
    --val-data   /root/srt-adapter/data/all_val.jsonl \
    --archetype-data /root/srt-adapter/artifacts/archetype_probe/generations.jsonl \
    --output-dir "$OUT" \
    --warm-start /root/srt-adapter/checkpoints/adapter_v9/best_adapter.pt \
    --no-prototypes \
    --batch-size 16 \
    --epochs 1 \
    --max-train-samples 16000 \
    --lr 1e-5 \
    --warmup-steps 50 \
    --max-seq-len 512 \
    --val-every 200 \
    --max-val-samples 2000 \
    --log-every 50 \
    --grad-clip 1.0 \
    --dtype bfloat16 \
    --community-supcon-weight 2.0 \
    --community-supcon-temperature 0.10 \
    --archetype-supcon-weight 2.0 \
    --archetype-supcon-temperature 0.10 \
    --archetype-mix-fraction 0.25 \
    --inject-supcon-weight 1.0 \
    --inject-supcon-temperature 0.10 \
    --divergence-supcon-weight 0.3 \
    --listnet-weight 0.5 \
    --chain-residual-aux-weight 0.05 \
    > "$LOG" 2>&1 &

echo "Launched v9.5 pilot, PID $!"
echo "Tail with: tail -f $LOG"
