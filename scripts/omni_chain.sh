#!/bin/bash
# Run the omni pipeline in dependency order, unattended.
#
# Every gate is a log string, never a pgrep: pgrep patterns self-match the
# launcher's own cmdline and have wedged chains here before. Each stage aborts
# the chain on failure rather than feeding garbage to the next one.
set -u
export HF_HOME=/root/.hf_home
cd /root/srt-adapter
L=/root/logs

echo "=== $(date -u +%H:%M:%S) waiting for model pulls"
until grep -q "ALL PULLS DONE" $L/pull.log 2>/dev/null; do sleep 30; done
echo "=== $(date -u +%H:%M:%S) models in"

echo "=== $(date -u +%H:%M:%S) modality smoke test"
/venv/main/bin/python scripts/omni_smoke.py --gpu 0 > $L/smoke2.log 2>&1
grep -E "^  (text|image|audio|video)" $L/smoke2.log
USABLE=$(grep "usable modalities:" $L/smoke2.log | tail -1)
echo "  $USABLE"
if ! echo "$USABLE" | grep -q "image"; then
  echo "!!! ABORT: image path failed, nothing downstream is trustworthy"
  exit 1
fi

echo "=== $(date -u +%H:%M:%S) building manifest"
/venv/main/bin/python scripts/build_omni_manifest.py > $L/manifest.log 2>&1
tail -6 $L/manifest.log
if [ ! -f /root/omni_manifest.json ]; then
  echo "!!! ABORT: no manifest"
  exit 1
fi

echo "=== $(date -u +%H:%M:%S) CHAIN READY"
echo "next: per-model encode, then joint_space_fit.py"
