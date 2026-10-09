#!/usr/bin/env bash
# TP=2 replica on two L4 GPUs. usage: MODEL=<dir> serve_l4.sh <mtp 0|1> [max_model_len] [attention_backend] [kv_dtype]
source /workspace/venv/bin/activate
MODEL=${MODEL:-/workspace/models/qwen3.8-27b-fp8}
MTP=${1:-0}; MML=${2:-65536}; BACKEND=${3:-}; KVD=${4:-fp8}
pkill -f "vllm serve" 2>/dev/null; sleep 5
TAG=$(basename $MODEL)_mtp${MTP}_${BACKEND:-default}_$KVD
EXTRA=""
[ "$MTP" = "1" ] && EXTRA="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":3}"
[ -n "$BACKEND" ] && EXTRA="$EXTRA --attention-backend $BACKEND"
setsid nohup vllm serve $MODEL --served-model-name qwen --port 8001 \
  --tensor-parallel-size 2 --max-model-len $MML --gpu-memory-utilization 0.92 --kv-cache-dtype $KVD \
  --max-num-seqs 16 --enable-prefix-caching --reasoning-parser qwen3 $EXTRA \
  > /workspace/logs/l4_$TAG.log 2>&1 < /dev/null &
echo launched $TAG
