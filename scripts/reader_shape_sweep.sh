#!/bin/bash
# How small can the shared-space reader get, and where is the budget best spent?
#
# The reader checkpoint is only the prefix MLP; Qwen3-0.6B ships separately.
#   P = hidden * d_model * (1 + n_tok)      d_model = 1024
# so the second layer (hidden -> n_tok*d_model) is ~all of it. Every prior sweep
# moved n_tok and left hidden at 2048, which is untuned.
#
# Pairs A/B and C/D hold the budget roughly fixed and trade width against hidden.
# E is the floor probe. Everything else matches the np16/np32/np64 runs exactly
# (all5 gallery, all5 head, 3 epochs, n=500) so medians are directly comparable
# to 40 / 33 / 37, and the two human arms must come back 39 and 28.
set -u
cd /root/srt-adapter
PY=/venv/main/bin/python

#      name  np  hidden
RUNS="A 32 512
B 8 2048
C 32 128
D 2 2048
E 32 64"

echo "$RUNS" | while read -r NAME NP H; do
  TAG="np${NP}_h${H}"
  CKPT="/root/reader_all5_${TAG}.pt"
  EVAL="/root/verb_eval_all5_${TAG}.json"
  [ -f "$EVAL" ] && { echo "=== [$NAME $TAG] already scored, skipping"; continue; }

  echo "=== [$NAME $TAG] train  $(date +%T)"
  $PY scripts/train_shared_space_verbalizer.py \
      --vecs /root/gal_all5.npy --caps /root/full_caps.json \
      --device cuda --bf16 --np "$NP" --hidden "$H" --holdout 5000 \
      --epochs 3 --batch 32 --lr 3e-4 --max-len 32 --save-every 100000 \
      --out "$CKPT"

  echo "=== [$NAME $TAG] eval  $(date +%T)"
  $PY scripts/eval_sunstone_verbalizer.py \
      --ckpt "$CKPT" --vecs /root/gal_all5.npy \
      --caps /root/full_caps.json --head-path /root/head_all5.pt \
      --n 500 --out "$EVAL"

  echo "=== [$NAME $TAG] done  $(date +%T)  ckpt $(du -h "$CKPT" | cut -f1)"
done

echo "=== SWEEP COMPLETE $(date +%T)"
