#!/usr/bin/env bash
# One vLLM server per GPU (data-parallel replicas) on ports 8001.. ; usage: serve4.sh [ngpu] [extra vllm args...]
set -u
N=${1:-4}; shift || true
source /workspace/venv/bin/activate
mkdir -p /workspace/logs
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i nohup vllm serve /workspace/models/qwen3.8-27b-fp8 \
    --served-model-name qwen \
    --port $((8001+i)) \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 64 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    "$@" > /workspace/logs/vllm_$i.log 2>&1 &
  sleep 2
done
echo "launched $N servers"
