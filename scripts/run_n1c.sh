#!/usr/bin/env bash
set -euo pipefail
cd /workspace/srt-adapter
source .venv/bin/activate
export HF_HOME=/workspace/.cache/huggingface
export PYTHONUNBUFFERED=1
echo "=== [$(date -Is)] N1c: direct entropy+KL loss terms ==="
python scripts/train_nla.py   --targets artifacts/nla/targets_q7b_L20_10k.pt   --steps 2000 --batch-size 8 --lr 3e-5   --beta-kl 0.05 --gamma-entropy 0.5   --val-every 250 --val-vectors 512   --out artifacts/nla/n1a_smoke
echo "=== [$(date -Is)] DONE ==="
