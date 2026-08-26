#!/usr/bin/env bash
# Caption-coverage A/B, end to end, one arm after the other.
#
# Everything is held fixed except how many captions the HEAD was shown:
# same gemma L47 image states, same images, same rng(0) holdout, same reader
# recipe (3 epochs, batch 32, lr 3e-4, np 16, hidden 2048), same eval.
#
#   cap0  first caption only, reproducing the shipped recipe as a fair control
#   all5  one of five resampled per epoch
#
# The shipped head is NOT the control here: it never held the eval images out
# of head training, so its gold arm is a memorised pair. Both arms below hold
# the same 5,000 images out of head training as well as reader training.
set -euo pipefail
cd /root/srt-adapter
export HF_HOME=/root/.hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/venv/main/bin/python

for MODE in cap0 all5; do
  echo "=== [$MODE] refit head  $(date +%T)"
  $PY scripts/refit_sunstone_head.py --mode "$MODE" \
      --out "/root/head_${MODE}.pt" --gallery-out "/root/gal_${MODE}.npy"

  echo "=== [$MODE] train reader  $(date +%T)"
  $PY scripts/train_shared_space_verbalizer.py \
      --vecs "/root/gal_${MODE}.npy" --caps /root/full_caps.json \
      --device cuda --bf16 --np 16 --hidden 2048 --holdout 5000 \
      --epochs 3 --batch 32 --lr 3e-4 --max-len 32 --save-every 5000 \
      --out "/root/reader_${MODE}.pt"

  echo "=== [$MODE] evaluate  $(date +%T)"
  $PY scripts/eval_sunstone_verbalizer.py \
      --ckpt "/root/reader_${MODE}.pt" --vecs "/root/gal_${MODE}.npy" \
      --caps /root/full_caps.json --head-path "/root/head_${MODE}.pt" \
      --n 500 --out "/root/verb_eval_${MODE}.json"
done

echo "=== ALL ARMS DONE  $(date +%T)"
for MODE in cap0 all5; do
  echo "--- $MODE"
  $PY - "$MODE" <<'EOF'
import json, sys
d = json.load(open(f"/root/verb_eval_{sys.argv[1]}.json"))
for k, v in d["arms"].items():
    print(f"  {k:32s} R@1 {v['r@1']:.4f}  median {v['median_rank']:>8,.0f}")
for k, v in d["head_to_head"].items():
    print(f"  {k:32s} {v}")
EOF
done
