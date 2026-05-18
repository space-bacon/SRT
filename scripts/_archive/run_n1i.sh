#!/usr/bin/env bash
set -euo pipefail
cd /workspace/srt-adapter
source .venv/bin/activate
export HF_HOME=/workspace/.cache/huggingface
export PYTHONUNBUFFERED=1
echo "=== [$(date -Is)] N1i: Phase-1 advantage-clip REINFORCE ==="
# N1g/N1h plateaued at fve_nrm=0.62 because pure REINFORCE with no
# advantage clipping let a few outlier sequences dominate every batch
# (pg_loss spikes of -50). Adv-clip 2.0 alone caps that. N1i-v1 also
# tried PPO ratio-clip + 2 inner epochs + token-mean logp, which
# regressed val to 0.28 (token-mean shrank gradient ~32x). This is
# the corrected recipe: adv-clip only, sum-logp REINFORCE, no PPO.
#
# All other hyperparameters identical to N1g/N1h.
python scripts/train_nla.py \
  --targets artifacts/nla/targets_q7b_L20_10k.pt \
  --steps 3000 --batch-size 32 --lr 3e-5 \
  --beta-kl 0.3 --gamma-entropy 0.5 --h-min 1.0 --h-max 999.0 \
  --adv-clip 2.0 --ppo-clip 0.0 --ppo-epochs 1 \
  --val-every 500 --val-vectors 512 \
  --out artifacts/nla/n1a_smoke
echo "=== [$(date -Is)] DONE ==="
