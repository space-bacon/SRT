#!/usr/bin/env bash
#
# v10: full-scale inject-site SupCon training on the C1 corpus.
#
# Builds on the v9.5 pilot (which restored directional separatrix forcing
# under unit-norm centroids by adding SupCon directly on the inject
# vectors). v10 trains from scratch on the full C1 corpus (332K rows
# across 15 schools — semiotics core, philosophy, wikipedia, wildchat,
# oasst2, hh-rlhf, gutenberg, arxiv-bulk) at the v9.5 pilot's loss
# weights.
#
# Config deltas from v9:
#   - corpus:        Reddit (1M) → C1 unified (332K rows, 15 schools)
#   - inject-supcon: OFF → 1.0 weight, 0.10 temp
#   - warm-start:    none (from scratch on new distribution)
#   - epochs:        1 → 2 (~41K steps; ~28h on A6000)
#
# Note: most C1 rows lack r_true, so listnet / divergence / chain-aux /
# bifurcation / regime losses silently no-op on those batches (mask is
# all-False). Active losses on C1 batches: CE + community_supcon (via
# 15 school-derived community ids); on the 4-row archetype mix per
# batch: + archetype_supcon + inject_supcon.

set -euo pipefail

cd /root/srt-adapter

OUT=/root/srt-adapter/checkpoints/adapter_v10
mkdir -p "$OUT"
LOG=$OUT/train_stdout.log

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/root/srt-adapter
export HF_HOME=/root/hf_cache
export HF_HUB_OFFLINE=1

nohup /root/srt_venv/bin/python scripts/train.py \
    --backbone Qwen/Qwen2.5-7B \
    --train-data /root/srt-adapter/data/corpus_c1/train.jsonl \
    --val-data   /root/srt-adapter/data/corpus_c1/val.jsonl \
    --archetype-data /root/srt-adapter/artifacts/archetype_probe/generations.jsonl \
    --output-dir "$OUT" \
    --warm-start /root/srt-adapter/checkpoints/adapter_v9/best_adapter.pt \
    --no-prototypes \
    --batch-size 16 \
    --epochs 2 \
    --lr 5e-5 \
    --warmup-steps 500 \
    --max-seq-len 512 \
    --val-every 2000 \
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

echo "Launched v10, PID $!"
echo "Tail: tail -f $LOG"
