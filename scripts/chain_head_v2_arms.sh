#!/usr/bin/env bash
# Ablation over the levers that do not grow the browser payload.
#
# Predictions registered before the first arm runs, so the result cannot be
# rationalised afterwards:
#   captions5   biggest single gain. 4K->118K pairs bought 4.2x, and this is the
#               only remaining 5x in data we already hold.
#   layers3     positive but modest, low single digits.
#   meanpool    could go either way; last-token was inherited, not chosen.
#   imgmlp      unknown, and the only one that is free in bytes.
#   epochs40    small positive; loss was still falling at 6.
#   all         better than any single lever, worse than their sum.
#
# Encoding is cached, so only the first arm pays for it.
set -uo pipefail
PY=/venv/main/bin/python
export HF_HOME=/root/.hf_home
COMMON="--raw-dir /root/raw118k --val-states /root/qwen38_coco_ls/images.pt \
        --ann /root/annotations --cache /root/capcache"

run () {
  local name="$1"; shift
  if [ -f "/root/v2_${name}.json" ]; then echo "[arm] $name present, skipping"; return; fi
  echo "[arm $(date -u +%H:%M:%SZ)] $name"
  $PY -u /root/browser_head_v2.py $COMMON --arm "$name" --out "/root/v2_${name}.json" "$@" \
    || echo "[arm] $name FAILED"
}

run baseline    --captions-per-image 1 --text-layers 28 --pooling last --img-head linear --epochs 20
run captions5   --captions-per-image 5 --text-layers 28 --pooling last --img-head linear --epochs 20
run layers3     --captions-per-image 1 --text-layers 22 25 28 --pooling last --img-head linear --epochs 20
run meanpool    --captions-per-image 1 --text-layers 28 --pooling mean --img-head linear --epochs 20
run imgmlp      --captions-per-image 1 --text-layers 28 --pooling last --img-head mlp --epochs 20
run epochs40    --captions-per-image 1 --text-layers 28 --pooling last --img-head linear --epochs 40

# Everything that helped, together. Kept last so the single-lever arms are
# already banked if this one runs long.
run all         --captions-per-image 5 --text-layers 22 25 28 --pooling last --img-head mlp \
                --epochs 40 --save-head /root/browser_head_v2_all.pt

echo "[arm $(date -u +%H:%M:%SZ)] ARMS_DONE"
for f in /root/v2_*.json; do
  $PY -c "import json,sys; d=json.load(open('$f')); v=d['val_only']; \
print(f\"{d['arm']:11s} i2t R@1 {v['i2t_r@1']:.4f}  t2i R@1 {v['t2i_r@1']:.4f}  \
head {d['shipped_text_head_mb']}MB\")"
done
