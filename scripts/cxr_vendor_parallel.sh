#!/bin/bash
# One vendor per GPU, all four encoding at once.
#
# On a single card this was 2.3 hours per vendor and nine hours for the set,
# which is why earlier runs downloaded and deleted hosts in sequence. With four
# cards and 2 TB the whole thing is one pass: every host stays resident and the
# wall clock is the slowest vendor rather than the sum.
#
# Each host is 48-62 GB in bf16 against 96 GB per card, so one model per GPU
# fits without sharding.
#
#   bash scripts/cxr_vendor_parallel.sh [manifest] [img-dir]
set -u
export HF_HOME=/root/.hf_home
export PYTHONPATH=/root/srt-adapter
cd /root/srt-adapter
PY=/venv/main/bin/python
L=/root/logs
MAN=${1:-/root/cxr14_manifest.json}
DIR=${2:-/root/cxr14}
TAGS="gemma4 qwen3omni mistral aria"

launch () {
  local gpu=$1 tag=$2 model=$3
  if [ -f "/root/cxr14_${tag}.npz" ]; then
    echo "  gpu $gpu  $tag already encoded, skipping"; return
  fi
  setsid nohup $PY scripts/omni_encode_all.py --model "$model" --gpu "$gpu" \
      --manifest "$MAN" --img-dir "$DIR" --out "/root/cxr14_${tag}.npz" \
      > "$L/cxr14_${tag}.log" 2>&1 < /dev/null &
  echo "  gpu $gpu  -> $tag"
}

echo "=== launching four encoders, $(date '+%F %T')"
launch 0 gemma4    "google/gemma-4-31B-it"
launch 1 qwen3omni "Qwen/Qwen3-Omni-30B-A3B-Instruct"
launch 2 mistral   "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
launch 3 aria      "rhymes-ai/Aria"

# Gate on the output files, not on pgrep: the pattern lives in this script's own
# command line and would match itself.
while true; do
  missing=""
  for t in $TAGS; do
    [ -f "/root/cxr14_${t}.npz" ] || missing="$missing $t"
  done
  [ -z "$missing" ] && break
  sleep 120
done
echo "=== all four encoded, $(date '+%F %T')"

for t in $TAGS; do
  echo "=== probing $t"
  $PY scripts/cxr_probe.py --states "/root/cxr14_${t}.npz" --manifest "$MAN" \
      --out "/root/cxr14_probe_${t}.json"
done

$PY scripts/cxr_vendor_compare.py --out /root/cxr14_vendor_compare.json
echo "=== DONE $(date '+%F %T')"
