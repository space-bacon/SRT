#!/usr/bin/env bash
#
# v11: short-horizon (5K), low-lr (1e-5), warm-start from v10. Goal:
# recover pilot-regime separatrix forcing while keeping v10 CE/recall.
# See docs/V11_PLAN.md for full rationale.
#
# Cost: ~3.5h on A6000 (5000 steps × ~0.4 step/s).

set -euo pipefail

cd /root/srt-adapter

OUT=/root/srt-adapter/checkpoints/adapter_v11
mkdir -p "$OUT"
LOG=$OUT/train_stdout.log

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/root/srt-adapter
export HF_HOME=/root/hf_cache
export HF_HUB_OFFLINE=1

# Generate the train/heldout split if missing. Deterministic given seed.
if [ ! -f /root/srt-adapter/data/archetype_probe/generations_heldout_v1.jsonl ]; then
    /root/srt_venv/bin/python scripts/split_archetype_heldout.py \
        --input artifacts/archetype_probe/generations.jsonl \
        --train-out data/archetype_probe/generations_train_v1.jsonl \
        --val-out   data/archetype_probe/generations_heldout_v1.jsonl
fi

nohup /root/srt_venv/bin/python scripts/train.py \
    --backbone Qwen/Qwen2.5-7B \
    --train-data /root/srt-adapter/data/corpus_c1/train.jsonl \
    --val-data   /root/srt-adapter/data/corpus_c1/val.jsonl \
    --archetype-data     /root/srt-adapter/data/archetype_probe/generations_train_v1.jsonl \
    --val-archetype-data /root/srt-adapter/data/archetype_probe/generations_heldout_v1.jsonl \
    --val-archetype-mix-fraction 0.25 \
    --output-dir "$OUT" \
    --warm-start /root/srt-adapter/checkpoints/adapter_v10/best_adapter.pt \
    --no-prototypes \
    --batch-size 16 \
    --epochs 1 \
    --max-train-samples 80000 \
    --lr 1e-5 \
    --warmup-steps 100 \
    --max-seq-len 512 \
    --val-every 500 \
    --max-val-samples 5000 \
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

echo "Launched v11, PID $!"
echo "Tail with: tail -f $LOG"
