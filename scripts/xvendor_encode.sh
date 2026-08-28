#!/bin/bash
# Encode the same manifest through a second vendor, sharded across all GPUs.
#
# gemma-4 has an audio_token_id but no audio tower, so audio is skipped rather
# than failed row by row. Its video processor also wants 32 frames where
# Qwen3-Omni takes 8, which is why the frame count is a parameter.
set -u
export HF_HOME=/root/.hf_home
cd /root/srt-adapter
L=/root/logs
MODEL=${1:-google/gemma-4-31B-it}
TAG=${2:-gemma4}
FRAMES=${3:-32}
SKIP=${4:-audio}
N=4

/venv/main/bin/python - "$N" "$SKIP" <<'PY'
import json, sys
n = int(sys.argv[1]); skip = {s for s in sys.argv[2].split(',') if s}
rows = [r for r in json.load(open('/root/omni_manifest.json'))['rows']
        if r['modality'] not in skip]
size = (len(rows) + n - 1) // n
for i in range(n):
    part = rows[i*size:(i+1)*size]
    json.dump({'counts': {}, 'rows': part},
              open(f'/root/xv_manifest_s{i}.json', 'w'))
    print(f'  shard {i}: {len(part)} rows')
print(f'  total after skipping {sorted(skip)}: {len(rows)}')
PY

for i in $(seq 0 $((N-1))); do
  setsid nohup /venv/main/bin/python scripts/omni_encode_all.py \
    --model "$MODEL" --gpu "$i" --video-frames "$FRAMES" \
    --manifest "/root/xv_manifest_s${i}.json" \
    --out "/root/${TAG}_states_s${i}.npz" \
    > "$L/xv${i}.log" 2>&1 < /dev/null &
  echo "launched $TAG shard $i on gpu $i"
done
echo "=== CROSS-VENDOR SHARDS LAUNCHED"
