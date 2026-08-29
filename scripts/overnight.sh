#!/bin/bash
# Everything that should happen tonight, in order, unattended.
#
# Stage 1 is the headline: does cross-vendor retention survive on satellite.
# Stage 2 measures the confound Dipankar Sarkar named, by deliberately doing
# the wrong thing (carrying the photo mean onto satellite) so we can report
# what it would have cost. Stage 3 repeats the control on radiology, which is
# further from web photographs than satellite is.
#
# Stages are independent: a failure prints and the queue moves on, so a bad
# download at 3am does not cost the whole night.
#
#   bash scripts/overnight.sh
set -u
export HF_HOME=/root/.hf_home
export PYTHONPATH=/root/srt-adapter
cd /root/srt-adapter
PY=/venv/main/bin/python
L=/root/logs
mkdir -p $L

banner () { echo; echo "############ $* ############"; date '+%F %T'; echo; }

# The fit's loader only accepts sharded filenames. Alias rather than edit the
# loader, which is the one that produced the published COCO number.
alias_shards () {
  local prefix=$1 manifest=$2
  for t in gemma4 qwen3omni mistral aria; do
    [ -f "/root/${prefix}_${t}.npz" ] && \
      ln -sf "/root/${prefix}_${t}.npz" "/root/${prefix}_${t}_s0.npz"
  done
  cp -f "$manifest" "${manifest%.json}_s0.json"
}

fit () {
  local prefix=$1 manifest=$2 out=$3; shift 3
  local M="${manifest%.json}_s0.json"
  $PY scripts/xvendor_fit_n.py \
    --vendor "qwen3omni:/root/${prefix}_qwen3omni_s0.npz:$M" \
    --vendor "gemma4:/root/${prefix}_gemma4_s0.npz:$M" \
    --vendor "mistral:/root/${prefix}_mistral_s0.npz:$M" \
    --vendor "aria:/root/${prefix}_aria_s0.npz:$M" \
    --out "$out" "$@"
}

encode_vendor () {
  local tag=$1 model=$2 prefix=$3 manifest=$4 imgdir=$5
  if [ -f "/root/${prefix}_${tag}.npz" ]; then
    echo "=== $tag: already encoded, skipping"; return
  fi
  echo "=== $tag: downloading $model"
  $PY - "$model" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], max_workers=8,
                  ignore_patterns=["consolidated*", "original/*", "*.pth", "*.gguf"])
PY
  echo "=== $tag: encoding"
  $PY scripts/omni_encode_all.py --model "$model" --gpu 0 \
      --manifest "$manifest" --img-dir "$imgdir" \
      --out "/root/${prefix}_${tag}.npz" > "$L/${prefix}_${tag}.log" 2>&1
  echo "=== $tag: done, freeing weights"
  rm -rf "/root/.hf_home/hub/models--${model//\//--}"
  df -h / | tail -1
}

########################################################################
banner "STAGE 1  satellite: wait for four vendors, then fit"

until [ -f /root/rsicd_gemma4.npz ] && [ -f /root/rsicd_qwen3omni.npz ] \
   && [ -f /root/rsicd_mistral.npz ] && [ -f /root/rsicd_aria.npz ]; do
  sleep 120
done
ls -lh /root/rsicd_*.npz | grep -v partial
alias_shards rsicd /root/rsicd_manifest.json
fit rsicd /root/rsicd_manifest.json /root/rsicd_xvendor4.json \
    --save-map /root/rsicd_map.pt

########################################################################
banner "STAGE 2  ablation: carry the photo mean onto satellite"
# Expected to be WORSE. The point is to put a number on how much worse, since
# that penalty is indistinguishable from the effect the control tests for.
if [ -f /root/xvendor4_map.pt ]; then
  fit rsicd /root/rsicd_manifest.json /root/rsicd_xvendor4_cocomu.json \
      --mu-from /root/xvendor4_map.pt
else
  echo "SKIPPED: /root/xvendor4_map.pt not on the box"
fi

########################################################################
banner "STAGE 3  radiology: fetch, encode four vendors, fit"

if [ ! -f /root/roco_manifest.json ]; then
  $PY scripts/get_roco.py --n 5000 --out-dir /root/roco \
      --manifest /root/roco_manifest.json
fi

encode_vendor gemma4    "google/gemma-4-31B-it"                        roco /root/roco_manifest.json /root/roco
encode_vendor qwen3omni "Qwen/Qwen3-Omni-30B-A3B-Instruct"             roco /root/roco_manifest.json /root/roco
encode_vendor mistral   "mistralai/Mistral-Small-3.1-24B-Instruct-2503" roco /root/roco_manifest.json /root/roco
encode_vendor aria      "rhymes-ai/Aria"                               roco /root/roco_manifest.json /root/roco

alias_shards roco /root/roco_manifest.json
fit roco /root/roco_manifest.json /root/roco_xvendor4.json

########################################################################
banner "SUMMARY"
$PY - <<'PY'
import json, os
BASE = ("photographs (COCO)", 0.9876, 0.955, 1.023)
runs = [("satellite (RSICD)",   "/root/rsicd_xvendor4.json"),
        ("  same, photo mean",  "/root/rsicd_xvendor4_cocomu.json"),
        ("radiology (ROCO)",    "/root/roco_xvendor4.json")]
print(f"{'domain':24s} {'cross':>7s} {'within':>7s} {'retention':>10s}  95% CI")
print(f"{BASE[0]:24s} {'':>7s} {'':>7s} {BASE[1]:10.4f}  "
      f"[{BASE[2]:.3f}, {BASE[3]:.3f}]   <- published baseline")
for name, path in runs:
    if not os.path.exists(path):
        print(f"{name:24s} {'did not complete':>26s}")
        continue
    s = json.load(open(path))["summary"]
    ci = s.get("retention_ci", {})
    print(f"{name:24s} {s['cross_vendor_mean_r@1']:7.4f} "
          f"{s['within_vendor_mean_r@1']:7.4f} {s['retention']:10.4f}  "
          f"[{ci.get('ci95_low', float('nan')):.3f}, "
          f"{ci.get('ci95_high', float('nan')):.3f}]")
print("\nWithin-vendor is printed beside every ratio on purpose: if within-vendor")
print("retrieval collapses on a domain, the ratio is measured against a floor.")
PY
banner "QUEUE COMPLETE"
