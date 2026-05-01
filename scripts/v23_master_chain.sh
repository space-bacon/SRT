#!/usr/bin/env bash
# v23 master: noise floor (parallel 2-GPU) -> beyond-STS (single GPU).
set -euo pipefail
REPO=/root/srt-adapter
cd "${REPO}"
MLOG=${REPO}/artifacts/v23_master.log

echo "[$(date)] ==== v23 master start ====" >> "${MLOG}"

echo "[$(date)] launching v23_noise_chain..." >> "${MLOG}"
bash ${REPO}/scripts/v23_noise_chain.sh
echo "[$(date)] noise chain done" >> "${MLOG}"

echo "[$(date)] launching v23_beyond_chain..." >> "${MLOG}"
bash ${REPO}/scripts/v23_beyond_chain.sh
echo "[$(date)] beyond chain done" >> "${MLOG}"

echo "[$(date)] ==== v23 master done ====" >> "${MLOG}"
