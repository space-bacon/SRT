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
from srt.nla.loss import fraction_variance_explained

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

    logger.info("loading targets from %s", args.targets)
    _, target_pool = _load_targets(args.targets)
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

        # Per-sample reward (reduce=False) — required for a meaningful
        # REINFORCE baseline. With reduce=True every component collapses to
        # a scalar and advantage = r - r.mean() is identically zero.
        reward = nla_reward(
            v_hat.float(),
            v_target.float(),
            lambda_mag=cfg.lambda_mag,
            reduce=False,
        )

        with torch.no_grad():
            advantage = reward.total - reward.total.mean()

        prefix = av._inject_prefix(v_target)  # (B, P, d)
        tok_embeds = backbone.get_input_embeddings()(gen_ids)
        inputs_embeds = torch.cat([prefix, tok_embeds], dim=1)
        full_attn = torch.cat(
            [torch.ones(prefix.shape[:2], dtype=attn.dtype, device=device), attn], dim=1
        )
        out = backbone(
            inputs_embeds=inputs_embeds, attention_mask=full_attn, use_cache=False
        )
        # Shift: predict gen_ids from prefix + earlier gen_ids.
        # Cast logits to float32 for log_softmax — bf16 log_softmax loses
        # ~3 digits of precision in the tails and biases REINFORCE grads.
        P = prefix.size(1)
        logits = out.logits[:, P - 1 : -1].float()  # (B, T_gen, V)
        logp = F.log_softmax(logits, dim=-1)
        tok_logp = logp.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)  # (B, T_gen)
        # Sum (not mean) of token log-probs: averaging shrinks the gradient
        # scale with sequence length and decouples the policy gradient from
        # the actual sample log-probability. Rely on clip_grad_norm_ below
        # for global scale control.
        seq_logp = (tok_logp * attn).sum(dim=1)
        loss = -(advantage * seq_logp).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step % args.log_every == 0:
            dt = time.time() - t0
            logger.info(
                "step %d  loss=%.4f  reward=%.4f  mse=%.4f  mag=%.4f  fve=%.4f  (%.1fs)",
                step,
                loss.item(),
                reward.total.mean().item(),
                reward.mse.mean().item(),
                reward.mag.mean().item(),
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
