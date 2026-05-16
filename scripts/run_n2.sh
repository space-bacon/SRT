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
# N2-v2 (post-mortem of N2-v1):
#   v1 settings: tau=1.0, alpha 1.0->0.5, h_min=1.0, lr=1e-5.
#   Failure modes observed in v1:
#     1. Entropy collapsed to 0.06 nats by step 100 -> soft_probs became
#        near-one-hot -> bridge gradient vanished through the matmul
#        soft_probs @ E_token.
#     2. As alpha annealed down, the REINFORCE term re-entered and
#        re-introduced single-scalar noise; H bounced 0.06 -> 0.95 -> 0.59;
#        loss went wildly negative; soft loss drifted back up.
#     3. Net: val=0.6171 at step 500, identical to N1i-v2 warm-start
#        ceiling. Bridge made zero discrete-eval progress.
#   v2 corrections:
#     - tau=0.5: sharper softmax narrows the gap between bridge-objective
#       (soft probs @ E) and discrete eval (argmax @ E).
#     - alpha_init=alpha_final=1.0, warmup=0.0: pure soft bridge, kill
#       the REINFORCE noise entirely (KL + entropy hinge are still active
#       because they are computed from the same logits and don't depend
#       on the policy gradient).
#     - h_min=1.5: prevent the entropy collapse that killed v1 by step 100.
#       Bridge needs soft probs spread across >=1 nat to have non-trivial
#       gradient through softmax.
#     - lr=3e-5: 3x v1 to compensate for the gradient attenuation from
#       sharper softmax (gradient ~ p*(1-p), shrinks at extremes).
python scripts/train_nla.py \
  --targets artifacts/nla/targets_q7b_L20_10k.pt \
  --init-from artifacts/nla/n1i_v2_best/av_step002500.pt \
  --steps 3000 --batch-size 32 --lr 3e-5 \
  --beta-kl 0.3 --gamma-entropy 0.5 --h-min 1.5 --h-max 3.0 \
  --adv-clip 2.0 --ppo-clip 0.0 --ppo-epochs 1 \
  --soft-bridge --soft-tau 0.5 \
  --soft-alpha-init 1.0 --soft-alpha-final 1.0 --soft-warmup-frac 0.0 \
  --val-every 500 --val-vectors 512 \
  --out artifacts/nla/n2_v2
echo "=== [$(date -Is)] DONE ==="
