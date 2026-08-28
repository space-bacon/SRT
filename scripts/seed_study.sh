#!/bin/bash
# Give the adapter line its first real error bar.
#
# The repo contains six *_seed{0,1,2} directories that are byte-identical
# COPIES of their base run, so the published v22c_a050-over-v18 gap of +0.0048
# has no variance estimate. MTEB evaluation of a fixed checkpoint is
# deterministic; only retraining moves anything.
#
# Retrains v18 from its v15a warm start with three training seeds, one seed per
# GPU in parallel, so the study costs one run's wall clock.
set -u
export HF_HOME=/root/.hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/srt-adapter
export PYTHONPATH=/root/srt-adapter
L=/root/logs

for s in 0 1 2; do
  echo "=== $(date -u +%H:%M:%S) launching seed $s on GPU $s"
  # v18 recipe verbatim: CoSENT, lr 1e-5, scale 20, batch 32, 5 epochs.
  CUDA_VISIBLE_DEVICES=$s /venv/main/bin/python scripts/train_supervised_sts_ddp.py \
    --warm-start /root/v15a_best_adapter.pt \
    --train-data /root/stsb_data/train.jsonl \
    --dev-data /root/stsb_data/dev.jsonl \
    --output-dir "/root/v18_seed${s}" \
    --loss cosent --cosent-scale 20 --lr 1e-5 --epochs 5 \
    --batch-size 32 --max-seq-len 128 --seed "$s" \
    > "$L/seed${s}.log" 2>&1 < /dev/null &
done

sleep 150
for s in 0 1 2; do
  echo "--- seed $s @150s:"; grep -E 'step|dev|spearman|Error|Traceback' "$L/seed${s}.log" | tail -2
done

wait
echo "=== $(date -u +%H:%M:%S) ALL SEEDS DONE"
for s in 0 1 2; do
  echo "seed $s ckpt: $(ls -lh /root/v18_seed${s}/best_adapter.pt 2>/dev/null | awk '{print $5}')"
done
