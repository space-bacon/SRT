#!/usr/bin/env bash
set -euo pipefail
cd /workspace/srt-adapter
source .venv/bin/activate
export HF_HOME=/workspace/.cache/huggingface
export PYTHONUNBUFFERED=1
echo "=== [$(date -Is)] N2: Phase-2 soft-embedding bridge ==="
# REINFORCE-only plateau at fve_nrm=0.6181 (N1g/N1h/N1i-v2 all converged
# to identical val_fve_nrm curve to 4 decimals). Diagnosis: information
# bottleneck of one scalar reward per sequence against a 3584-dim target.
#
# Soft-embedding bridge: feed softmax(av_logits/tau) @ E_token through AR
# instead of sampled ids -> per-token-per-dim dense gradient on AV. Still
# corpus-free (no text dataset). Hybrid loss:
#   loss = alpha * mse_nrm(v_hat_soft, v_target)
#        + (1 - alpha) * (pg_loss + kl_loss + ent_loss)
# alpha linearly annealed 1.0 -> 0.5 over first 50% of steps, held at 0.5
# thereafter (keeps dense gradient active long-term; REINFORCE term keeps
# the policy faithful to the discrete sampling distribution).
#
# Warm-start from N1i-v2 step-2500 (the REINFORCE ceiling) so we are
# *fine-tuning* the plateau, not re-discovering it.
#
# Pass criteria:
#   - val_fve_nrm >= 0.70 by step 1500 -> ride the bridge to Phase 3
#   - val_fve_nrm >= 0.75 by step 3000 -> phased plan on track
#   - soft_loss decreasing -> dense gradient is actually doing work
#   - H stays in [0.5, 2.5] -> policy not collapsing into soft-mode regime
# N2-v3 (post-mortem of N2-v2):
#   v2 used: tau=0.5, alpha=1.0 (pure soft), h_min=1.5 h_max=3.0, lr=3e-5.
#   Failure mode in v2: with alpha=1.0 the prior loss formula
#     loss = alpha*soft + (1-alpha)*(pg+kl+ent)
#   silenced kl_loss and ent_loss entirely (they got weight 0). Nothing
#   stopped the policy from racing to uniform softmax (H=9.2 nats by step
#   200), at which point soft_probs @ E_token = mean(E_token) = a constant
#   degenerate embedding -- a trivial minimum of soft_loss that has zero
#   relation to discrete eval. val crashed 0.62 -> 0.31.
#   v3 fix (in train_nla.py): regularizers are now added with full weight
#   regardless of alpha:
#     loss = alpha*soft + (1-alpha)*pg + kl_loss + ent_loss
#   This keeps the entropy hinge braking at H>3 and the KL anchor pulling
#   AV toward base Qwen. v3 launcher reuses the same hparams as v2 -- the
#   bug was purely in the loss formula, not the schedule.
python scripts/train_nla.py \
  --targets artifacts/nla/targets_q7b_L20_10k.pt \
  --init-from artifacts/nla/n1i_v2_best/av_step002500.pt \
  --steps 3000 --batch-size 32 --lr 3e-5 \
  --beta-kl 0.3 --gamma-entropy 0.5 --h-min 1.5 --h-max 3.0 \
  --adv-clip 2.0 --ppo-clip 0.0 --ppo-epochs 1 \
  --soft-bridge --soft-tau 0.5 \
  --soft-alpha-init 1.0 --soft-alpha-final 1.0 --soft-warmup-frac 0.0 \
  --val-every 500 --val-vectors 512 \
  --out artifacts/nla/n2_v3
echo "=== [$(date -Is)] DONE ==="
