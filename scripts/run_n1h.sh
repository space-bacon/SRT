#!/usr/bin/env bash
set -euo pipefail
cd /workspace/srt-adapter
source .venv/bin/activate
export HF_HOME=/workspace/.cache/huggingface
export PYTHONUNBUFFERED=1
echo "=== [$(date -Is)] N1h: N1g recipe extended to 6000 steps ==="
# N1g (2000 steps) peaked at fve_nrm=0.6192 at step 1750 and was still
# climbing at step 2000 (0.6149). Hypothesis: it had not converged.
# Triple the budget; keep all other hyperparameters identical.
python scripts/train_nla.py \
  --targets artifacts/nla/targets_q7b_L20_10k.pt \
  --steps 6000 --batch-size 32 --lr 3e-5 \
  --beta-kl 0.3 --gamma-entropy 0.5 --h-min 1.0 --h-max 999.0 \
  --val-every 500 --val-vectors 512 \
  --out artifacts/nla/n1a_smoke
echo "=== [$(date -Is)] DONE ==="
