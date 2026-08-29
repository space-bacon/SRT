#!/bin/bash
# Every GPU on one vendor at a time, rather than one vendor per GPU.
#
# One vendor per card finishes when the slowest vendor does, which is 13.8 hours
# here, and leaves gemma-4's card idle for five of them. Sharding each vendor
# across all four and running vendors back to back is the same 45.5 GPU-hours
# but 11.4 wall-clock, and the first vendor lands in 2.2 hours instead of 8.8.
#
# Vendors run fastest first so the earliest result is also the one we already
# have a 35k pilot for, which makes it a check on the pipeline as well.
#
#   bash scripts/cxr_shard_chain.sh
set -u
export HF_HOME=/root/.hf_home
export PYTHONPATH=/root/srt-adapter
cd /root/srt-adapter
PY=/venv/main/bin/python
L=/root/logs
MAN=/root/cxr14_manifest.json
DIR=/root/cxr14
N=4

# Contiguous shards, so concatenating in order restores the manifest's row order.
$PY - "$N" "$MAN" <<'PY'
import json, sys
n, man = int(sys.argv[1]), sys.argv[2]
rows = json.load(open(man))["rows"]
size = (len(rows) + n - 1) // n
for i in range(n):
    part = rows[i * size:(i + 1) * size]
    json.dump({"counts": {"image": len(part)}, "rows": part},
              open(f"/root/cxr14_shard_s{i}.json", "w"))
    print(f"  shard {i}: {len(part)} rows")
print(f"  total {len(rows)}")
PY

run_vendor () {
  local tag=$1 model=$2
  if [ -f "/root/cxr14_${tag}.npz" ]; then
    echo "=== $tag already merged, skipping"; return
  fi
  echo "=== $tag: encoding on $N GPUs, $(date '+%F %T')"
  for i in $(seq 0 $((N - 1))); do
    setsid nohup $PY scripts/omni_encode_all.py --model "$model" --gpu "$i" \
        --manifest "/root/cxr14_shard_s${i}.json" --img-dir "$DIR" \
        --out "/root/cxr14_${tag}_s${i}.npz" \
        > "$L/cxr14_${tag}_s${i}.log" 2>&1 < /dev/null &
  done
  # Gate on the shard outputs, never on pgrep: this script's own command line
  # carries the pattern and would match itself.
  while true; do
    done_n=0
    for i in $(seq 0 $((N - 1))); do
      [ -f "/root/cxr14_${tag}_s${i}.npz" ] && done_n=$((done_n + 1))
    done
    [ "$done_n" -eq "$N" ] && break
    sleep 60
  done
  echo "=== $tag: merging, $(date '+%F %T')"
  $PY scripts/merge_shards.py --tag "$tag" --shards $N
  rm -f /root/cxr14_${tag}_s*.npz /root/cxr14_${tag}_s*.partial.npz
  echo "=== $tag: probing"
  $PY scripts/cxr_probe.py --states "/root/cxr14_${tag}.npz" --manifest "$MAN" \
      --out "/root/cxr14_probe_${tag}.json"
  df -h / | tail -1
}

run_vendor gemma4    "google/gemma-4-31B-it"
run_vendor aria      "rhymes-ai/Aria"
run_vendor mistral   "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
run_vendor qwen3omni "Qwen/Qwen3-Omni-30B-A3B-Instruct"

echo
echo "=== ALL VENDORS DONE $(date '+%F %T')"
$PY scripts/cxr_vendor_compare.py --out /root/cxr14_vendor_compare.json
