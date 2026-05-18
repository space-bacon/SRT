#!/usr/bin/env bash
set -euo pipefail
cd /workspace/srt-adapter
source .venv/bin/activate
export HF_HOME=/workspace/.cache/huggingface
export PYTHONUNBUFFERED=1
echo "=== [$(date -Is)] STAGE 1: sample targets ==="
python scripts/sample_targets.py   --backbone Qwen/Qwen2.5-7B --layer 20   --num-sequences 10000 --seq-len 256 --batch-size 16   --out artifacts/nla/targets_q7b_L20_10k.pt
echo "=== [$(date -Is)] STAGE 2: train ==="
python scripts/train_nla.py   --targets artifacts/nla/targets_q7b_L20_10k.pt   --steps 2000 --batch-size 8 --lr 3e-5   --val-every 250 --val-vectors 512   --out artifacts/nla/n1a_smoke
echo "=== [$(date -Is)] DONE ==="
