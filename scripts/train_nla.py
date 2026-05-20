"""SRT-NLA training loop — Phase N1 (corpus-free RL).

Loads a parquet/pt shard produced by ``scripts/sample_targets.py``, runs
the ``ActivationVerbalizer`` to produce candidate descriptions, then
re-encodes the descriptions with ``ActivationReconstructor`` and applies
the 5-term reward from ``srt.nla.loss.nla_reward``.

N1 keeps things minimal: REINFORCE on the projection + prefix
parameters. GRPO upgrade lands in N1.5.

Usage:
    python scripts/train_nla.py \\
        --targets artifacts/nla/targets_qwen7b_L20.pt \\
        --steps 5000 \\
        --batch-size 8 \\
        --lr 1e-5 \\
        --out artifacts/nla/checkpoints/n1_v0
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from srt.nla import (
    ActivationReconstructor,
    ActivationVerbalizer,
    NLAConfig,
    NLAMeta,
    nla_reward,
    save_meta,
)
from srt.nla.loss import fraction_variance_explained, mse_nrm

logger = logging.getLogger("train_nla")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", required=True, type=Path)
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--val-vectors", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    # Direct loss-term weights (NOT reward weights — these flow gradient
    # straight through AV's logp, bypassing the REINFORCE baseline.)
    p.add_argument("--beta-kl", type=float, default=0.05,
                   help="KL(av || base) coefficient added to loss.")
    p.add_argument("--gamma-entropy", type=float, default=0.5,
                   help="Entropy hinge coefficient. Loss contribution is "
                        "gamma * (max(0, h_min - H_mean) + max(0, H_mean - h_max)) "
                        "so the term is silent inside [h_min, h_max] and the task "
                        "signal dominates. A plain -gamma*H bonus drives H runaway "
                        "to several nats (seen at N1c step 160: H=5.9). A one-sided "
                        "floor-only hinge still allows runaway (N1e: bimodal regime, "
                        "H spikes 5-9 nats causing pg_loss to blow up to -87).")
    p.add_argument("--h-min", type=float, default=1.5,
                   help="Entropy floor in nats. Lower hinge activates below this.")
    p.add_argument("--h-max", type=float, default=3.0,
                   help="Entropy ceiling in nats. Upper hinge activates above this. "
                        "Natural-language token entropy is typically 1-2 nats; 3.0 "
                        "leaves headroom while bounding runaway.")
    # PPO-lite knobs. Pure REINFORCE-with-batch-baseline (N1a..N1h) plateaus
    # around fve_nrm=0.62 because a small fraction of sequences with large
    # |seq_logp| dominate every batch and clip_grad_norm then drowns the
    # signal from the rest. PPO clipping bounds the per-sequence importance
    # ratio so no single sample can dominate. See docs/nla_mission.md.
    p.add_argument("--adv-clip", type=float, default=2.0,
                   help="Symmetric clip on the per-sequence advantage. "
                        "Set to 0 to disable.")
    p.add_argument("--ppo-clip", type=float, default=0.0,
                   help="PPO importance-ratio clip epsilon. 0 = plain "
                        "REINFORCE (default; PPO clip only helps when "
                        "ppo-epochs > 1).")
    p.add_argument("--ppo-epochs", type=int, default=1,
                   help="Inner gradient steps per sampled batch (PPO-style "
                        "sample reuse). 1 = no reuse. N1i-v1 showed that at "
                        "lr=3e-5 the ratio between inner epochs is ~1.0 so "
                        "inner epochs were wasted compute.")
    # Phase-2 soft-embedding bridge. REINFORCE alone (N1a..N1i) plateaus at
    # fve_nrm=0.62: one scalar reward per sequence is information-bottlenecked
    # against a 3584-dim reconstruction target. The soft bridge runs AR on
    # softmax(av_logits/tau) @ E_token instead of the sampled token ids,
    # giving a per-token-per-dim dense gradient w.r.t. AV. Loss is blended:
    # loss = alpha * mse_nrm(v_hat_soft, v_target) + (1-alpha)*(pg+kl+ent).
    # alpha is linearly annealed from soft_alpha_init to soft_alpha_final
    # over the first soft_warmup_frac of steps, then held constant. Still
    # corpus-free (no text dataset used).
    p.add_argument("--soft-bridge", action="store_true",
                   help="Enable Phase-2 soft-embedding bridge.")
    p.add_argument("--soft-tau", type=float, default=1.0,
                   help="Softmax temperature for the soft-embedding bridge.")
    p.add_argument("--soft-alpha-init", type=float, default=1.0,
                   help="Initial weight on the soft-bridge MSE term.")
    p.add_argument("--soft-alpha-final", type=float, default=0.5,
                   help="Final weight on the soft-bridge MSE term (held "
                        "after warmup so dense gradient stays available).")
    p.add_argument("--soft-warmup-frac", type=float, default=0.5,
                   help="Fraction of total steps over which alpha is "
                        "linearly annealed from init to final.")
    p.add_argument("--init-from", type=Path, default=None,
                   help="Path to a saved AV trainable-state dict to warm-start "
                        "from (e.g. the N1i-v2 step-2500 checkpoint).")
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _load_targets(path: Path) -> tuple[list[str], torch.Tensor]:
    """Return (sequences, pooled_activations_(N,d) float32 cpu)."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    seqs: list[str] = obj["sequences"]
    acts: list[torch.Tensor] = obj["activations"]
    pooled = torch.stack([a[-1] for a in acts])  # last-token pool to match AR default
    return seqs, pooled


def _first_eos_mask(gen_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Attention mask that keeps every token up to and including the first EOS.

    With Qwen's ``pad_token_id = eos_token_id`` convention, masking via
    ``ids != pad_id`` silently drops the legitimate sentence-ending EOS,
    which (a) shifts AR's last-token pool one token earlier and (b) zeroes
    out the EOS slot in REINFORCE so the model never learns to stop.
    """
    B, T = gen_ids.shape
    is_eos = gen_ids == eos_id
    pos = torch.arange(T, device=gen_ids.device).unsqueeze(0).expand(B, T)
    # first EOS index per row; T (out of range) if no EOS produced.
    first_eos = torch.where(
        is_eos.any(dim=-1, keepdim=True),
        is_eos.float().argmax(dim=-1, keepdim=True).long(),
        torch.full((B, 1), T, device=gen_ids.device, dtype=torch.long),
    )
    return (pos <= first_eos).long()


def _alpha_schedule(step: int, args: argparse.Namespace) -> float:
    """Linear anneal of soft-bridge weight from init to final."""
    if not args.soft_bridge:
        return 0.0
    warmup = max(1, int(args.steps * args.soft_warmup_frac))
    if step >= warmup:
        return args.soft_alpha_final
    frac = step / warmup
    return args.soft_alpha_init + frac * (args.soft_alpha_final - args.soft_alpha_init)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = NLAConfig(
        backbone_id=args.backbone,
        backbone_dtype=args.dtype,
        extraction_layer=args.layer,
        max_new_tokens=args.max_new_tokens,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("loading shared backbone %s on %s", args.backbone, device)
    backbone = AutoModelForCausalLM.from_pretrained(args.backbone, torch_dtype=_dtype(args.dtype))
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.to(device)
    backbone.eval()
    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    av = ActivationVerbalizer(cfg, backbone=backbone, tokenizer=tok).to(device)
    ar = ActivationReconstructor(cfg, backbone=backbone).to(device)

    trainable = [p for p in av.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    logger.info("trainable params: %s", f"{n_trainable:,}")
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    if args.init_from is not None:
        logger.info("warm-starting AV from %s", args.init_from)
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        state = ckpt["trainable"] if isinstance(ckpt, dict) and "trainable" in ckpt else ckpt
        missing, unexpected = av.load_state_dict(state, strict=False)
        logger.info("warm-start: loaded %d params; missing=%s unexpected=%s",
                    len(state), list(missing), list(unexpected))

    logger.info("loading targets from %s", args.targets)
    _, target_pool = _load_targets(args.targets)
    from srt.nla.targets_check import assert_targets_healthy
    assert_targets_healthy(args.targets, target_pool)
    target_pool = target_pool.to(device)
    N = target_pool.size(0)
    logger.info("targets: N=%d d=%d", N, target_pool.size(1))

    val_idx = torch.randperm(N, device=device)[: args.val_vectors]
    val_targets = target_pool[val_idx]

    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.batch_size,), device=device)
        v_target = target_pool[idx]

        gen_ids = av.generate(
            v_target, max_new_tokens=args.max_new_tokens, do_sample=True
        )
        attn = _first_eos_mask(gen_ids, tok.eos_token_id)
        v_hat = ar.reconstruct(gen_ids, attention_mask=attn)

        valid = attn.float()
        valid_counts = valid.sum(dim=1).clamp(min=1.0)

        # Base policy logp for KL anchor — computed once per sampled batch
        # (the base model is frozen so this never changes across inner epochs).
        with torch.no_grad():
            base_out = backbone(
                input_ids=gen_ids, attention_mask=attn, use_cache=False
            )
            base_logp_full = F.log_softmax(base_out.logits.float(), dim=-1)
            base_logp_kl = base_logp_full[:, :-1, :].detach()

        # Reward depends only on (sampled tokens, target) — both fixed across
        # inner epochs — so we compute it once outside the loop.
        reward = nla_reward(
            v_hat.float(),
            v_target.float(),
            lambda_mag=cfg.lambda_mag,
            reduce=False,
        )
        with torch.no_grad():
            advantage = reward.total - reward.total.mean()
            if args.adv_clip > 0:
                advantage = advantage.clamp(-args.adv_clip, args.adv_clip)

        # Snapshot the behaviour-policy log-prob (the policy that actually
        # produced these tokens). Required for the PPO importance ratio;
        # for plain REINFORCE (--ppo-clip=0 and --ppo-epochs=1) the ratio
        # is identically 1 and this is a no-op.
        with torch.no_grad():
            prefix_old = av._inject_prefix(v_target)
            tok_embeds_old = backbone.get_input_embeddings()(gen_ids)
            inputs_embeds_old = torch.cat([prefix_old, tok_embeds_old], dim=1)
            full_attn = torch.cat(
                [torch.ones(prefix_old.shape[:2], dtype=attn.dtype, device=device), attn],
                dim=1,
            )
            P = prefix_old.size(1)
            out_old = backbone(
                inputs_embeds=inputs_embeds_old, attention_mask=full_attn,
                use_cache=False,
            )
            logits_old = out_old.logits[:, P - 1 : -1].float()
            logp_old_all = F.log_softmax(logits_old, dim=-1)
            tok_logp_old = logp_old_all.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)
            # Sum-over-valid-tokens (the unbiased REINFORCE estimator).
            # Mean-over-tokens was tried in N1i-v1 and silently killed the
            # signal: dividing by ~T=32 made the effective gradient ~32x
            # smaller at the same lr, and it implicitly upweights short
            # sequences. Variance control happens via adv_clip below, not
            # by shrinking the logp scale.
            seq_logp_old = (tok_logp_old * attn).sum(dim=1)

        # ── inner PPO epochs over the same sampled batch ──────────
        # Rebuild inputs_embeds *inside* the loop: the prefix is produced by
        # av._inject_prefix which has trainable parameters, so its graph is
        # consumed by the first .backward(). Re-running the projection each
        # iteration with the same v_target/gen_ids gives a fresh graph that
        # reflects the latest AV weights (correct PPO semantics).
        for inner in range(max(1, args.ppo_epochs)):
            prefix = av._inject_prefix(v_target)
            tok_embeds = backbone.get_input_embeddings()(gen_ids)
            inputs_embeds = torch.cat([prefix, tok_embeds], dim=1)
            out = backbone(
                inputs_embeds=inputs_embeds, attention_mask=full_attn,
                use_cache=False,
            )
            logits = out.logits[:, P - 1 : -1].float()  # (B, T_gen, V)
            logp = F.log_softmax(logits, dim=-1)
            tok_logp = logp.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)
            seq_logp = (tok_logp * attn).sum(dim=1)

            # Per-token entropy (used by the two-sided hinge).
            p_av = logp.exp()
            H_tok = -(p_av * logp).sum(dim=-1)
            token_entropy = (H_tok * valid).sum(dim=1) / valid_counts

            # KL(av || base) at positions 1..T-1.
            av_logp_kl = logp[:, 1:, :]
            p_kl = av_logp_kl.exp()
            kl_tok = (p_kl * (av_logp_kl - base_logp_kl)).sum(dim=-1)
            kl_mask = valid[:, 1:]
            kl_counts = kl_mask.sum(dim=1).clamp(min=1.0)
            kl_per_sample = (kl_tok * kl_mask).sum(dim=1) / kl_counts

            # PPO clipped surrogate. Maximise min(r*A, clip(r, 1±ε)*A) is
            # equivalent to minimising -that quantity. With ε=0 the clip
            # is degenerate (r forced to 1) so we fall back to plain PG.
            ratio = (seq_logp - seq_logp_old).exp()
            if args.ppo_clip > 0:
                ratio_clipped = ratio.clamp(
                    1.0 - args.ppo_clip, 1.0 + args.ppo_clip
                )
                surrogate = torch.min(ratio * advantage, ratio_clipped * advantage)
                pg_loss = -surrogate.mean()
            else:
                pg_loss = -(advantage * seq_logp).mean()

            kl_loss = args.beta_kl * kl_per_sample.mean()
            H_mean = token_entropy.mean()
            ent_lo = torch.clamp(args.h_min - H_mean, min=0.0)
            ent_hi = torch.clamp(H_mean - args.h_max, min=0.0)
            ent_loss = args.gamma_entropy * (ent_lo + ent_hi)

            # Phase-2 soft-embedding bridge. AR re-encodes
            # softmax(av_logits/tau) @ E_token instead of the sampled ids,
            # so AV gets a per-token-per-dim gradient from mse_nrm. The
            # sampled rollout (gen_ids) is what AV's logits are conditioned
            # on; the gradient flows through the *probabilities* assigned
            # to that rollout, not through a re-sample.
            #
            # NOTE on regularizer weighting: kl_loss and ent_loss are added
            # with FULL weight regardless of alpha. N2-v2 tried
            #   loss = alpha*soft + (1-alpha)*(pg+kl+ent)
            # which silenced the entropy hinge at alpha=1.0; the policy
            # raced to uniform (H=9.2 nats by step 200) because uniform
            # softmax @ E_token = constant embedding = a degenerate minimum
            # of the bridge loss that's totally divorced from discrete
            # eval. val crashed warm-start 0.62 -> 0.31. The hinge MUST
            # always brake. alpha only blends the two policy-objective
            # terms (soft bridge vs REINFORCE pg).
            alpha = _alpha_schedule(step, args)
            if args.soft_bridge and alpha > 0.0:
                # Matmul in backbone dtype (bf16) to avoid allocating a
                # float32 copy of the 152K x 3584 embedding table every
                # step (~2.2 GB). Some precision is lost in soft_probs but
                # the bridge target is direction, not magnitude — empirically
                # tolerable.
                E_tok = backbone.get_input_embeddings().weight  # (V, d) bf16
                soft_probs = F.softmax(logits / args.soft_tau, dim=-1).to(E_tok.dtype)
                soft_embeds = soft_probs @ E_tok  # (B, T_gen, d) bf16
                v_hat_soft = ar.reconstruct_from_embeds(
                    soft_embeds, attention_mask=attn
                )
                soft_loss = mse_nrm(v_hat_soft.float(), v_target.float())
                loss = (
                    alpha * soft_loss
                    + (1.0 - alpha) * pg_loss
                    + kl_loss
                    + ent_loss
                )
            else:
                soft_loss = torch.tensor(0.0, device=device)
                loss = pg_loss + kl_loss + ent_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

        if step % args.log_every == 0:
            dt = time.time() - t0
            logger.info(
                "step %d  loss=%.4f  pg=%.4f  kl_l=%.4f  ent_l=%.4f  "
                "soft=%.4f a=%.2f  "
                "reward=%.4f  mse=%.4f  mag=%.4f  KL=%.4f  H=%.4f  fve=%.4f  (%.1fs)",
                step,
                loss.item(),
                pg_loss.item(),
                kl_loss.item(),
                ent_loss.item(),
                soft_loss.item(),
                alpha,
                reward.total.mean().item(),
                reward.mse.mean().item(),
                reward.mag.mean().item(),
                kl_per_sample.mean().item(),
                token_entropy.mean().item(),
                fraction_variance_explained(v_hat.float(), v_target.float()).item(),
                dt,
            )

        if step % args.val_every == 0:
            av.eval()
            with torch.no_grad():
                v_ids = av.generate(val_targets, max_new_tokens=args.max_new_tokens, do_sample=False)
                v_attn = _first_eos_mask(v_ids, tok.eos_token_id)
                v_hat_val = ar.reconstruct(v_ids, attention_mask=v_attn)
                fve_val = fraction_variance_explained(v_hat_val.float(), val_targets.float()).item()
            av.train()
            logger.info("val @ %d  fve_nrm=%.4f", step, fve_val)
            ckpt = args.out / f"av_step{step:06d}.pt"
            trainable_state = {
                n: p.detach().cpu()
                for n, p in av.named_parameters()
                if p.requires_grad
            }
            torch.save({"trainable": trainable_state, "step": step}, ckpt)
            save_meta(
                NLAMeta(
                    backbone_id=args.backbone,
                    backbone_revision=None,
                    extraction_layer=args.layer,
                    d_vector=av.proj.weight.size(1),
                    pool=cfg.pool,
                    pair="av",
                    phase="N1",
                    notes=f"step={step} fve_nrm={fve_val:.4f}",
                ),
                args.out / f"av_step{step:06d}.meta.yaml",
            )

    logger.info("done")


if __name__ == "__main__":
    main()
