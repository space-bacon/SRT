#!/usr/bin/env bash
set -uo pipefail

# Phase 0/1 A/B on ResearchClawBench.
#   Phase 0 = bare Qwen2.5-7B-Instruct (injectors OFF)
#   Phase 1 = SRT-steered            (injectors ON)
# Same tasks for both so the score delta is attributable to the adapter.
#
# Architecture (venv isolation — REQUIRED):
#   venv A (SRT_VENV)     : runs the SRT server only. Kept pristine.
#   venv B (HARNESS_VENV) : runs cli_eval + the agent's Bash. Disposable.
#
# Run on the box:
#   bash /workspace/srt-adapter/scripts/rcb_run_srt_phase01.sh

SRT_ROOT=/workspace/srt-adapter
RCB_ROOT=/workspace/ResearchClawBench
SRT_VENV=/workspace/srt-adapter/.venv
HARNESS_VENV=/workspace/harness_venv
CKPT="$SRT_ROOT/artifacts/checkpoints/v4/best_adapter.pt"
PORT=8000
CONFIG=eval_configs/srt_phase01.yaml

export HF_HOME=/workspace/.hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -a; source "$SRT_ROOT/.env"; set +a
export AGENT_MODEL_NAME="srt-qwen2.5-7b"
export AGENT_API_BASE="http://127.0.0.1:${PORT}/v1"
export AGENT_API_KEY="dummy"
export JUDGE_MODEL_NAME="${MODEL_NAME:-}"
export JUDGE_API_BASE="${API_BASE:-}"
export JUDGE_API_KEY="${API_KEY:-}"

if [[ -z "$JUDGE_MODEL_NAME" || -z "$JUDGE_API_BASE" || -z "$JUDGE_API_KEY" ]]; then
  echo "Judge env incomplete. Set MODEL_NAME/API_BASE/API_KEY in $SRT_ROOT/.env"; exit 1
fi

start_server() {  # $1=inject|no-inject  $2=logfile
  local mode="$1" log="$2"
  pkill -9 -f serve_srt_openai 2>/dev/null; sleep 3
  PYTHONPATH="$SRT_ROOT" HF_HOME=/workspace/.hf_home \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$SRT_VENV/bin/python3" "$SRT_ROOT/scripts/serve_srt_openai.py" \
    --backbone Qwen/Qwen2.5-7B-Instruct --adapter-ckpt "$CKPT" \
    --served-name srt-qwen2.5-7b --port "$PORT" "--$mode" > "$log" 2>&1 &
  for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "server up ($mode)"; return 0
    fi
    sleep 3
  done
  echo "server failed ($mode); see $log"; tail -5 "$log"; return 1
}

run_eval() {  # $1=arm label
  echo "=== START $1 ==="
  ( cd "$RCB_ROOT" && "$HARNESS_VENV/bin/python3" -m evaluation.cli_eval "$CONFIG" )
  echo "=== DONE $1 ==="
}

start_server "no-inject" /workspace/srt_server_bare.log || exit 1
run_eval "BARE (no-inject)"

start_server "inject" /workspace/srt_server_inject.log || exit 1
run_eval "SRT (inject)"

pkill -9 -f serve_srt_openai 2>/dev/null
echo "Phase 0/1 complete. Batches under $RCB_ROOT/workspaces/cli_runs/"
echo "Compare eval_report_*.md mean scores between the two batches."
