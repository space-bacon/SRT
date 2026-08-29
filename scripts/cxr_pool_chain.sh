#!/bin/bash
# Finish the gemma-4 baseline, then test whether pooling is the real ceiling.
#
# The four-vendor comparison answers whether the representation is shared across
# independently trained models, which belongs to the cross-vendor work. It does
# not make the clinical number better: three more 30B-class models will land
# near gemma-4. The clinical ceiling is set somewhere else.
#
# A 1024px film becomes 256 image tokens, one per 64x64 block, and we then
# average all 256. The first probe's ordering says that is the binding
# constraint: Emphysema 0.853 and Cardiomegaly 0.842, which survive averaging,
# against Nodule 0.652 and Mass 0.680, which do not.
#
# So this runs one more gemma-4 pass keeping three layers and three poolings,
# nine variants for the price of one encode.
#
#   bash scripts/cxr_pool_chain.sh
set -u
export HF_HOME=/root/.hf_home
export PYTHONPATH=/root/srt-adapter
cd /root/srt-adapter
PY=/venv/main/bin/python
L=/root/logs
MAN=/root/cxr14_manifest.json
N=4

echo "=== waiting on the gemma-4 baseline shards already in flight"
while true; do
  d=0
  for i in $(seq 0 $((N - 1))); do
    [ -f "/root/cxr14_gemma4_s${i}.npz" ] && d=$((d + 1))
  done
  [ "$d" -eq "$N" ] && break
  sleep 60
done

if [ ! -f /root/cxr14_gemma4.npz ]; then
  echo "=== merging baseline, $(date '+%F %T')"
  $PY scripts/merge_shards.py --tag gemma4 --shards $N
  rm -f /root/cxr14_gemma4_s*.npz
fi
if [ ! -f /root/cxr14_probe_gemma4.json ]; then
  echo "=== probing baseline on the full 112,120"
  $PY scripts/cxr_probe.py --states /root/cxr14_gemma4.npz --manifest "$MAN" \
      --out /root/cxr14_probe_gemma4.json
fi

echo
echo "=== pooling variants: 3 layers x 3 poolings, one pass, $(date '+%F %T')"
for i in $(seq 0 $((N - 1))); do
  setsid nohup $PY scripts/cxr_encode_pool.py --gpu "$i" \
      --manifest "/root/cxr14_shard_s${i}.json" --img-dir /root/cxr14 \
      --out "/root/cxr14_pool_s${i}.npz" \
      > "$L/cxr14_pool_s${i}.log" 2>&1 < /dev/null &
done
while true; do
  d=0
  for i in $(seq 0 $((N - 1))); do
    [ -f "/root/cxr14_pool_s${i}.npz" ] && d=$((d + 1))
  done
  [ "$d" -eq "$N" ] && break
  sleep 60
done

echo "=== merging pooling variants, $(date '+%F %T')"
$PY scripts/merge_shards.py --tag pool --shards $N
rm -f /root/cxr14_pool_s*.npz
$PY scripts/cxr_pool_sweep.py --states /root/cxr14_pool.npz --manifest "$MAN" \
    --out /root/cxr14_pool_sweep.json
echo "=== DONE $(date '+%F %T')"
