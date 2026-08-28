#!/bin/bash
# Encode the manifest across all four GPUs, one shard each.
#
# The box's advantage is four resident 98GB cards; a single-GPU encode wastes
# three quarters of it. Shards are contiguous slices so a failed shard is
# re-runnable on its own without redoing the rest.
set -u
export HF_HOME=/root/.hf_home
cd /root/srt-adapter
L=/root/logs
N=${1:-4}

/venv/main/bin/python - "$N" <<'PY'
import json, sys
n = int(sys.argv[1])
d = json.load(open('/root/omni_manifest.json'))
rows = d['rows']
size = (len(rows) + n - 1) // n
for i in range(n):
    part = rows[i*size:(i+1)*size]
    json.dump({'counts': {}, 'rows': part},
              open(f'/root/omni_manifest_s{i}.json', 'w'))
    print(f'  shard {i}: {len(part)} rows')
PY

for i in $(seq 0 $((N-1))); do
  setsid nohup /venv/main/bin/python scripts/omni_encode_all.py \
    --gpu "$i" --manifest "/root/omni_manifest_s${i}.json" \
    --out "/root/omni_states_s${i}.npz" > "$L/enc${i}.log" 2>&1 < /dev/null &
  echo "launched shard $i on gpu $i"
done
echo "=== ALL SHARDS LAUNCHED"
