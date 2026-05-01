#!/usr/bin/env bash
# Idempotent sync of SRT artifacts to Supabase Storage.
#
# Loads SUPABASE_URL and SUPABASE_SECRET_KEY from .env (in repo root).
# Uploads files with x-upsert: true so re-runs overwrite cleanly. No
# diffing — Supabase de-dupes by content hash on its side.
#
# What it syncs:
#   - LOCAL adapter checkpoints in artifacts/checkpoints/{v18,v20,v21*}/
#       -> srt-checkpoints/<version>/best_adapter.pt
#       -> srt-checkpoints/<version>/config.json (when present)
#   - LOCAL MTEB summaries in artifacts/mteb/<version>/summary.json
#       -> srt-checkpoints/<version>/mteb_summary.json
#   - REMOTE corpora on the vast.ai box (data/supervised_sts_v*/{train,dev}.jsonl)
#       -> srt-corpora/<version>/{train,dev}.jsonl
#
# Usage:
#   ./scripts/sync_supabase.sh                    # full sync
#   ./scripts/sync_supabase.sh --no-corpora       # skip remote box step
#   ./scripts/sync_supabase.sh --version v21a     # only one version
#
# Cron example (every 6 hours):
#   0 */6 * * * cd /path/to/srt-adapter && ./scripts/sync_supabase.sh >> /tmp/srt_sync.log 2>&1

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO}"

# shellcheck disable=SC1091
set -a; source .env; set +a

: "${SUPABASE_URL:?SUPABASE_URL missing in .env}"
: "${SUPABASE_SECRET_KEY:?SUPABASE_SECRET_KEY missing in .env}"

SSH_HOST=${SSH_HOST:-"root@ssh2.vast.ai"}
SSH_PORT=${SSH_PORT:-14083}
REMOTE_REPO=${REMOTE_REPO:-/root/srt-adapter}

WANT_CORPORA=1
WANT_VERSION=""

while [ $# -gt 0 ]; do
  case "$1" in
    --no-corpora) WANT_CORPORA=0 ;;
    --version)    WANT_VERSION="$2"; shift ;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

upload_local() {
  local file="$1" bucket="$2" remote_path="$3"
  if [ ! -s "$file" ]; then
    echo "  [skip] $file (missing or empty)"; return 0
  fi
  local size; size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file")
  printf "  [up ] %-55s -> %s/%s (%d B) " "$file" "$bucket" "$remote_path" "$size"
  local code
  code=$(curl -sS -o /tmp/srt_sync.out -w "%{http_code}" \
    -X POST "${SUPABASE_URL}/storage/v1/object/${bucket}/${remote_path}" \
    -H "Authorization: Bearer ${SUPABASE_SECRET_KEY}" \
    -H "apikey: ${SUPABASE_SECRET_KEY}" \
    -H "x-upsert: true" \
    --data-binary "@${file}")
  echo "HTTP ${code}"
  if [ "$code" != "200" ] && [ "$code" != "201" ]; then
    echo "    !! $(cat /tmp/srt_sync.out)"; return 1
  fi
}

upload_remote() {
  local remote_file="$1" bucket="$2" remote_path="$3"
  local size
  size=$(ssh -p "${SSH_PORT}" "${SSH_HOST}" "stat -c%s ${remote_file}" 2>/dev/null || echo "")
  if [ -z "$size" ] || [ "$size" = "0" ]; then
    echo "  [skip] ${remote_file} (missing on box)"; return 0
  fi
  printf "  [up*] %-55s -> %s/%s (%d B) " "$remote_file" "$bucket" "$remote_path" "$size"
  local code
  code=$(ssh -p "${SSH_PORT}" "${SSH_HOST}" "cat ${remote_file}" \
    | curl -sS -o /tmp/srt_sync.out -w "%{http_code}" \
        -X POST "${SUPABASE_URL}/storage/v1/object/${bucket}/${remote_path}" \
        -H "Authorization: Bearer ${SUPABASE_SECRET_KEY}" \
        -H "apikey: ${SUPABASE_SECRET_KEY}" \
        -H "x-upsert: true" \
        -H "Content-Length: ${size}" \
        --data-binary @-)
  echo "HTTP ${code}"
  if [ "$code" != "200" ] && [ "$code" != "201" ]; then
    echo "    !! $(cat /tmp/srt_sync.out)"; return 1
  fi
}

want() {
  [ -z "$WANT_VERSION" ] || [ "$1" = "$WANT_VERSION" ]
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === SRT supabase sync ==="

echo "--- local: adapter checkpoints + MTEB summaries -> srt-checkpoints"
for dir in artifacts/checkpoints/v* artifacts/checkpoints/step*; do
  [ -d "$dir" ] || continue
  ver=$(basename "$dir")
  want "$ver" || continue
  [ -s "${dir}/best_adapter.pt" ] && upload_local "${dir}/best_adapter.pt" "srt-checkpoints" "${ver}/best_adapter.pt" || true
  [ -s "${dir}/config.json" ]     && upload_local "${dir}/config.json"     "srt-checkpoints" "${ver}/config.json"     || true
  if [ -s "artifacts/mteb/${ver}/summary.json" ]; then
    upload_local "artifacts/mteb/${ver}/summary.json" "srt-checkpoints" "${ver}/mteb_summary.json"
  fi
done

if [ "$WANT_CORPORA" = "1" ]; then
  echo "--- remote box: corpora -> srt-corpora"
  remote_versions=$(ssh -p "${SSH_PORT}" "${SSH_HOST}" \
    "ls -1 ${REMOTE_REPO}/data/ 2>/dev/null | grep '^supervised_sts_'" || true)
  for full_dir in $remote_versions; do
    ver=${full_dir#supervised_sts_}
    want "$ver" || continue
    upload_remote "${REMOTE_REPO}/data/${full_dir}/dev.jsonl"   "srt-corpora" "${full_dir}/dev.jsonl"
    upload_remote "${REMOTE_REPO}/data/${full_dir}/train.jsonl" "srt-corpora" "${full_dir}/train.jsonl"
  done
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === sync done ==="
