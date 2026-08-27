#!/bin/bash
# Settle the tap-layer question at 30,000 images instead of 4,000.
#
# L20 beats the shipped L47 tap by +0.061 r@1 at 4,000 training images and the
# gap widens monotonically from 250 up, so it is not a small-sample effect. This
# takes it to 30,000, which is 7.5x more data, at a quarter the cost of the full
# train2017 re-encode.
#
# Nothing else may hold the GPU while this runs: gemma-4 needs ~61 GB and two
# copies do not fit on a 96 GB card. That already cost one eval today.
set -eu
cd /root/srt-adapter
PY=/venv/main/bin/python

if [ ! -d /root/train2017 ] || [ "$(ls /root/train2017 2>/dev/null | wc -l)" -lt 30000 ]; then
  echo "=== fetching train2017  $(date +%T)"
  cd /root
  curl -sSL -o train2017.zip http://images.cocodataset.org/zips/train2017.zip
  echo "=== unzipping  $(date +%T)"
  unzip -q -o train2017.zip -d /root
  rm -f train2017.zip
  cd /root/srt-adapter
fi
echo "=== train2017 images: $(ls /root/train2017 | wc -l)  $(date +%T)"

echo "=== encode + fit at 30k  $(date +%T)"
$PY scripts/l60_head_refit.py \
    --img-dir /root/train2017 \
    --caps /root/annotations/captions_train2017.json \
    --layers 20,47 --n 30000 --holdout 2000 \
    --cache /root/train30k_states.npz \
    --out /root/tap_layer_30k.json

echo "=== scaling curve on the same states  $(date +%T)"
$PY scripts/tap_layer_scaling.py \
    --cache /root/train30k_states.npz \
    --cache-layers 20,47 --layers 20,47 \
    --sizes 2000,4000,8000,16000,28000 \
    --seeds 3 --holdout 2000 \
    --out /root/tap_layer_scaling_30k.json

echo "=== DONE  $(date +%T)"
