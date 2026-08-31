#!/usr/bin/env bash
# One pending list, one dispatcher, no hand-assignment to cards.
#
# Pointing jobs at specific GPUs by hand went wrong three times in one session:
# a queue started on a card a manual job still held, three jobs burned in 41
# seconds against a full card, and a latency benchmark run alongside a training
# job that silently corrupted its numbers. Every one of those was an assignment
# mistake, not a code bug.
#
# So assignment stops being a decision. Append work to queues/pending.txt and the
# dispatcher hands each task to a card that is actually free, one task per card,
# claiming atomically so nothing runs twice.
#
#   scripts/dispatch.sh add "python scripts/foo.py --bar"     # queue work
#   scripts/dispatch.sh add --gpus 4 "python scripts/big.py"  # needs 4 cards
#   scripts/dispatch.sh start                                 # run the dispatcher
#   scripts/dispatch.sh status                                # what is where
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
mkdir -p queues logs

PENDING="queues/pending.txt"
STATE="logs/dispatch.state"
LOCK="logs/dispatch.lock"
FREE_MB=4000

touch "$PENDING" "$STATE"

# A task's identity is the hash of its text, so re-adding the same command is
# visible as a duplicate rather than silently running twice.
task_id() { printf '%s' "$1" | cksum | cut -d' ' -f1; }

claimed() { grep -q "^$1 " "$STATE" 2>/dev/null; }

gpu_free() {
  local used
  used=$(nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  [[ -z "$used" ]] && return 1
  (( used < FREE_MB ))
}

case "${1:-status}" in
add)
  shift
  n=1
  [[ "${1:-}" == "--gpus" ]] && { n="$2"; shift 2; }
  cmd="$*"
  [[ -z "$cmd" ]] && { echo "nothing to add"; exit 1; }
  id=$(task_id "$cmd")
  if grep -q "	$id	" "$PENDING" 2>/dev/null || claimed "$id"; then
    echo "already queued or run: $id"
    exit 0
  fi
  printf '%s\t%s\t%s\n' "$n" "$id" "$cmd" >> "$PENDING"
  echo "queued $id (needs $n gpu) :: $cmd"
  ;;

start)
  exec 9>"$LOCK"
  flock -n 9 || { echo "a dispatcher is already running"; exit 1; }
  echo "dispatcher up $(date -Is)" >> logs/dispatch.log
  {
    while :; do
      # Rebuild the free list each pass; another agent may take a card at any time.
      free=()
      for g in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
        gpu_free "$g" && free+=("$g")
      done

      progressed=0
      while IFS=$'\t' read -r need id cmd; do
        [[ -z "${id:-}" ]] && continue
        claimed "$id" && continue
        (( ${#free[@]} >= need )) || continue

        assign=$(IFS=,; echo "${free[*]:0:$need}")
        free=("${free[@]:$need}")
        printf '%s %s %s\n' "$id" "$assign" "$(date -Is)" >> "$STATE"
        echo "DISPATCH $id -> gpu $assign :: $cmd" >> logs/dispatch.log

        (
          log="logs/task_${id}.log"
          if CUDA_VISIBLE_DEVICES="$assign" \
             PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
             bash -c "$cmd" > "$log" 2>&1; then
            echo "DONE $id gpu=$assign $(date -Is)" >> logs/dispatch.log
          else
            echo "FAIL $id gpu=$assign rc=$? $(date -Is) :: $cmd" >> logs/dispatch.log
          fi
        ) &
        progressed=1
        # One launch per pass: the card needs a moment before it reads as busy.
        sleep 20
        break
      done < "$PENDING"

      (( progressed )) || sleep 30
    done
  } &
  echo $! > logs/dispatch.pid
  echo "dispatcher started (pid $(cat logs/dispatch.pid)), log logs/dispatch.log"
  ;;

status)
  echo "GPUs:"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader |
    sed 's/^/  /'
  echo
  echo "pending (unclaimed):"
  while IFS=$'\t' read -r need id cmd; do
    [[ -z "${id:-}" ]] && continue
    claimed "$id" || printf '  %s  needs %s  %s\n' "$id" "$need" "${cmd:0:64}"
  done < "$PENDING"
  echo
  echo "recent dispatch log:"
  tail -6 logs/dispatch.log 2>/dev/null | sed 's/^/  /' || echo "  (none)"
  ;;

stop)
  [[ -f logs/dispatch.pid ]] && kill "$(cat logs/dispatch.pid)" 2>/dev/null &&
    echo "dispatcher stopped" || echo "no dispatcher running"
  rm -f logs/dispatch.pid
  ;;

*)
  echo "usage: dispatch.sh {add [--gpus N] CMD | start | status | stop}"
  exit 1
  ;;
esac
