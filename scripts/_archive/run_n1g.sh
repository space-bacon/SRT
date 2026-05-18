#!/usr/bin/env bash
set -euo pipefail
cd /workspace/srt-adapter
source .venv/bin/activate
export HF_HOME=/workspace/.cache/huggingface
export PYTHONUNBUFFERED=1
echo "=== [$(date -Is)] N1g: bs=32 beta_kl=0.3 h_min=1.0 (h_max disabled) ==="
# Interpolate between N1e (beta=0.2, peak 0.603, KL drifted to 2.8) and
# N1f (beta=0.5, peak 0.581, KL pinned at 0.13 but over-constrained).
# Disable upper hinge by setting h_max very high.
python scripts/train_nla.py \
  --targets artifacts/nla/targets_q7b_L20_10k.pt \
  --steps 2000 --batch-size 32 --lr 3e-5 \
  --beta-kl 0.3 --gamma-entropy 0.5 --h-min 1.0 --h-max 999.0 \
  --val-every 250 --val-vectors 512 \
  --out artifacts/nla/n1a_smoke
echo "=== [$(date -Is)] DONE ==="
