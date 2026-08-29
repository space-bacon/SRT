#!/bin/bash
# Encode RSICD through the remaining three vendors, one at a time.
#
# The box has 104 GB free and the three hosts total 168 GB, so they cannot all
# be resident. Each is downloaded, used, and deleted before the next starts.
# gemma-4 is handled separately since it was already cached.
#
#   bash scripts/rsicd_chain.sh
set -u
export HF_HOME=/root/.hf_home
export PYTHONPATH=/root/srt-adapter
cd /root/srt-adapter
L=/root/logs
PY=/venv/main/bin/python

# tag model frames
run_vendor () {
  local tag=$1 model=$2 frames=$3
  echo "=== $tag: downloading $model" | tee -a $L/rsicd_chain.log
  $PY - "$model" <<'PY' >> $L/rsicd_chain.log 2>&1
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], max_workers=8,
                  ignore_patterns=["consolidated*", "original/*", "*.pth", "*.gguf"])
PY
  echo "=== $tag: encoding" | tee -a $L/rsicd_chain.log
  $PY scripts/omni_encode_all.py --model "$model" --gpu 0 \
      --manifest /root/rsicd_manifest.json --img-dir /root/rsicd \
      --video-frames "$frames" \
      --out "/root/rsicd_${tag}.npz" > "$L/rsicd_${tag}.log" 2>&1
  echo "=== $tag: done, freeing weights" | tee -a $L/rsicd_chain.log
  # Keep the states, drop the weights so the next host fits.
  rm -rf "/root/.hf_home/hub/models--${model//\//--}"
  df -h / | tail -1 | tee -a $L/rsicd_chain.log
}

# Wait for the gemma encode already in flight.
while pgrep -f "omni_encode_all.py --model google/gemm[a]" > /dev/null; do sleep 60; done
echo "=== gemma4 finished, starting the rest" | tee -a $L/rsicd_chain.log
rm -rf /root/.hf_home/hub/models--google--gemma-4-31B-it

run_vendor qwen3omni "Qwen/Qwen3-Omni-30B-A3B-Instruct" 8
run_vendor mistral   "mistralai/Mistral-Small-3.1-24B-Instruct-2503" 8
run_vendor aria      "rhymes-ai/Aria" 8

echo "=== ALL VENDORS DONE" | tee -a $L/rsicd_chain.log
ls -lh /root/rsicd_*.npz | tee -a $L/rsicd_chain.log
