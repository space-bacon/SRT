#!/usr/bin/env bash
# v22 master chain: run v22c (souping eval) -> v22a (STSB-anchored) -> v22b (multi-teacher)
# sequentially so they don't fight over GPUs.
# Total wall-clock ~3.5h.
set -euo pipefail
REPO=/root/srt-adapter
cd "${REPO}"
MLOG=${REPO}/artifacts/v22_master.log

echo "[$(date)] ==== v22 master start ====" >> "${MLOG}"

echo "[$(date)] launching v22c..." >> "${MLOG}"
bash ${REPO}/scripts/v22c_chain.sh
echo "[$(date)] v22c done" >> "${MLOG}"

echo "[$(date)] launching v22a..." >> "${MLOG}"
bash ${REPO}/scripts/v22a_chain.sh
echo "[$(date)] v22a done" >> "${MLOG}"

echo "[$(date)] launching v22b..." >> "${MLOG}"
bash ${REPO}/scripts/v22b_chain.sh
echo "[$(date)] v22b done" >> "${MLOG}"

echo "[$(date)] ==== v22 master done ====" >> "${MLOG}"
