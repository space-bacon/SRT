#!/usr/bin/env bash
# Drain a list of jobs on one GPU, sequentially, unattended.
#
# Hand-feeding one model per GPU and polling for the next free card wastes the
# operator and leaves cards idle between jobs. This takes a file of shell
# commands, one per line, and runs them in order on a fixed GPU until the file
# is done. Failures are recorded and do not stop the queue, because a single OOM
# should not idle a card for an hour.
#
#   scripts/gpu_queue.sh 2 queues/gpu2.txt
#
# Status for every queue at once:
#   scripts/gpu_queue.sh status
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

if [[ "${1:-}" == "status" ]]; then
  printf '%-6s %-9s %s\n' GPU STATE 'CURRENT / LAST'
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader |
    while IFS=, read -r i mem util; do
      st="idle"; [[ -f logs/queue_gpu$i.pid ]] && kill -0 "$(cat logs/queue_gpu$i.pid)" 2>/dev/null && st="running"
      last=$(tail -1 "logs/queue_gpu$i.log" 2>/dev/null | tr -s ' ' | cut -c1-72)
      printf '%-6s %-9s %s | %s\n' "$i" "$st" "${mem}${util}" "$last"
    done
  echo
  echo "recent failures:"
  grep -h "^QUEUE-FAIL" logs/queue_gpu*.log 2>/dev/null | tail -8 || echo "  none"
  exit 0
fi

gpu="${1:?usage: gpu_queue.sh <gpu-id> <jobfile>}"
jobs="${2:?usage: gpu_queue.sh <gpu-id> <jobfile>}"
log="logs/queue_gpu${gpu}.log"
mkdir -p logs

{
  echo "QUEUE-START gpu=$gpu jobs=$jobs $(date -Is)"
  n=0
  while IFS= read -r cmd; do
    [[ -z "$cmd" || "$cmd" == \#* ]] && continue
    n=$((n + 1))
    echo "QUEUE-JOB $n gpu=$gpu $(date -Is) :: $cmd"
    # Fragmentation is the usual cause of a late-run OOM on these cards.
    if CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        bash -c "$cmd"; then
      echo "QUEUE-OK $n gpu=$gpu $(date -Is)"
    else
      echo "QUEUE-FAIL $n gpu=$gpu rc=$? $(date -Is) :: $cmd"
    fi
    # Let the driver reclaim the card before the next model loads.
    sleep 10
  done < "$jobs"
  echo "QUEUE-DONE gpu=$gpu $(date -Is)"
  rm -f "logs/queue_gpu${gpu}.pid"
} >> "$log" 2>&1 &

echo $! > "logs/queue_gpu${gpu}.pid"
echo "queue on gpu $gpu -> $log (pid $(cat logs/queue_gpu${gpu}.pid))"
