#!/bin/bash
# Run the joint fit as soon as all four encode shards land.
# Gated on the log string each shard prints on success, never on pgrep.
set -u
export HF_HOME=/root/.hf_home
cd /root/srt-adapter
L=/root/logs

echo "=== $(date -u +%H:%M:%S) waiting for 4 shards"
while true; do
  n=$(grep -l "^wrote /root/omni_states_s" $L/enc*.log 2>/dev/null | wc -l)
  [ "$n" -ge 4 ] && break
  sleep 15
done
echo "=== $(date -u +%H:%M:%S) all shards written"

echo "--- per-shard yield ---"
grep -h -E "^encoded|^  (image|audio|video):" $L/enc*.log

echo "=== $(date -u +%H:%M:%S) fitting shared space"
/venv/main/bin/python scripts/omni_joint_fit.py \
  --states '/root/omni_states_s*.npz' --out /root/omni_joint.json 2>&1 \
  | grep -vE "Warning:"

echo "=== $(date -u +%H:%M:%S) JOINT FIT DONE"
