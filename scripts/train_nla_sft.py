"""Supervised fine-tuning for SRT-NLA Phase C (distillation).

Loads a JSONL of ``(target_idx, gold_ids)`` pairs produced by
``scripts/curate_bon.py`` and trains AV to reproduce the gold tokens
under teacher-forcing. Pure cross-entropy — no RL, no entropy hinge,
no soft bridge. Validation tracks discrete-eval fve_nrm on a held-out
slice of the target pool (same metric as REINFORCE / N1).

The target vectors come from the same .pt file used at curation time
(referenced by ``target_idx``); the script re-loads the pool and looks
up each record's target by index.

Usage:
    python scripts/train_nla_sft.py \\
        --targets artifacts/nla/targets_q7b_L20_10k.pt \\
        --pairs artifacts/nla/curated/bon_iter1.jsonl \\
        --init-from artifacts/nla/n1i_v2_best/av_step002500.pt \\
        --val-fraction 0.05 --epochs 3 --batch-size 16 --lr 3e-5 \\
        --out artifacts/nla/sft_iter1
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from srt.nla import ActivationReconstructor, ActivationVerbalizer, NLAConfig
from srt.nla.loss import mse_nrm
from srt.nla.targets_check import assert_targets_healthy

logger = logging.getLogger("train_nla_sft")

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", required=True, type=Path)
    p.add_argument("--pairs", required=True, type=Path,
                   help="JSONL produced by curate_bon.py.")
    p.add_argument("--init-from", type=Path, default=None)
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=64,
                   help="Generation budget for validation eval.")
    p.add_argument("--max-seq-len", type=int, default=64,
                   help="Truncate gold token sequences to this length.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.05,
                   help="Hold out this fraction of pairs for validation.")
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-vectors", type=int, default=512,
                   help="Number of held-out targets sampled for discrete eval.")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--patience", type=int, default=0,
                   help="Early-stop after this many val checks without improvement. 0=disabled.")
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def _load_targets(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    acts = obj["activations"]
    return torch.stack([a[-1] for a in acts])


def _load_pairs(path: Path) -> list[dict]:
    recs = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _collate(
    batch: list[dict],
    targets: torch.Tensor,
    pad_id: int,
    max_len: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (v_targets (B, d), gold_ids (B, T), attn (B, T))."""
    B = len(batch)
    ids_list = [r["gold_ids"][:max_len] for r in batch]
    T = max(len(x) for x in ids_list)
    gold = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        L = len(ids)
        gold[i, :L] = torch.tensor(ids, dtype=torch.long)
        attn[i, :L] = 1
    v = targets[[r["target_idx"] for r in batch]]
    return v.to(device), gold.to(device), attn.to(device)


@torch.no_grad()
def _val_fve(
    av: ActivationVerbalizer,
    ar: ActivationReconstructor,
    val_targets: torch.Tensor,
    pad_id: int,
    eos_id: int,
    max_new_tokens: int,
    batch_size: int,
) -> float:
    """Discrete-eval fve_nrm: generate greedy + low-temp, score against target."""
    fve_sum = 0.0
    n = 0
    for i in range(0, val_targets.size(0), batch_size):
        v = val_targets[i : i + batch_size]
        gen_ids = av.generate(v, max_new_tokens=max_new_tokens, do_sample=False)
        B, T = gen_ids.shape
        is_eos = gen_ids == eos_id
        pos = torch.arange(T, device=gen_ids.device).unsqueeze(0).expand(B, T)
        first_eos = torch.where(
            is_eos.any(dim=-1, keepdim=True),
            is_eos.float().argmax(dim=-1, keepdim=True).long(),
            torch.full((B, 1), T, device=gen_ids.device, dtype=torch.long),
        )
        attn = (pos <= first_eos).long()
        v_hat = ar.reconstruct(gen_ids, attention_mask=attn).float()
        v_tgt = v.float()
        v_hat_n = F.normalize(v_hat, dim=-1)
        v_tgt_n = F.normalize(v_tgt, dim=-1)
        mse_per = ((v_hat_n - v_tgt_n) ** 2).sum(dim=-1)
        fve_sum += float((1.0 - mse_per / 2.0).sum().item())
        n += B
    return fve_sum / max(n, 1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = NLAConfig(
        backbone_id=args.backbone,
        backbone_dtype=args.dtype,
        extraction_layer=args.layer,
        max_new_tokens=args.max_new_tokens,
    )

    logger.info("loading backbone %s on %s", args.backbone, device)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.backbone, torch_dtype=_DTYPE_MAP[args.dtype]
    )
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    av = ActivationVerbalizer(cfg, backbone=backbone, tokenizer=tok).to(device)
    ar = ActivationReconstructor(cfg, backbone=backbone).to(device)

    trainable = [p for p in av.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    logger.info("trainable params: %s", f"{n_trainable:,}")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # Estimate total optimizer steps for cosine schedule.
    # (Recomputed after pairs load; placeholder None until then.)
    sched = None

    if args.init_from is not None:
        logger.info("warm-start AV from %s", args.init_from)
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        state = ckpt["trainable"] if isinstance(ckpt, dict) and "trainable" in ckpt else ckpt
        missing, unexpected = av.load_state_dict(state, strict=False)
        logger.info("loaded %d params; missing=%d unexpected=%d",
                    len(state), len(missing), len(unexpected))

    pool = _load_targets(args.targets)
    assert_targets_healthy(args.targets, pool)
    pool = pool.to(device)
    pairs = _load_pairs(args.pairs)
    logger.info("loaded %d pairs; target pool N=%d", len(pairs), pool.size(0))

    # Deterministic train/val split on pair indices.
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(pairs), generator=g).tolist()
    n_val = max(1, int(len(pairs) * args.val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    val_targets = pool[[pairs[i]["target_idx"] for i in val_idx[: args.val_vectors]]]
    logger.info("train pairs=%d  val targets=%d", len(train_idx), val_targets.size(0))

    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id

    # Build LR scheduler now that we know dataset size.
    steps_per_epoch = max(1, (len(train_idx) + args.batch_size - 1) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    if args.lr_schedule == "cosine":
        import math
        warmup = max(0, args.warmup_steps)

        def _lr_lambda(s: int) -> float:
            if warmup > 0 and s < warmup:
                return (s + 1) / float(warmup)
            progress = (s - warmup) / max(1, total_steps - warmup)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        logger.info("cosine LR: total_steps=%d warmup=%d peak_lr=%g",
                    total_steps, warmup, args.lr)

    best_val = -1.0
    best_path = args.out / "best_av.pt"
    bad_vals = 0
    stop_training = False

    step = 0
    t0 = time.time()
    log_loss = 0.0
    log_n = 0

    for epoch in range(1, args.epochs + 1):
        torch.manual_seed(args.seed + epoch)
        shuffled = torch.randperm(len(train_idx)).tolist()
        epoch_pairs = [pairs[train_idx[i]] for i in shuffled]

        for i in range(0, len(epoch_pairs), args.batch_size):
            batch = epoch_pairs[i : i + args.batch_size]
            v, gold, attn = _collate(batch, pool, pad_id, args.max_seq_len, device)

            # Build teacher-forced inputs: [prefix(v), gold_embeds]
            prefix = av._inject_prefix(v)
            P = prefix.size(1)
            tok_embeds = backbone.get_input_embeddings()(gold)
            inputs_embeds = torch.cat([prefix, tok_embeds], dim=1)
            full_attn = torch.cat(
                [torch.ones(prefix.shape[:2], dtype=attn.dtype, device=device), attn],
                dim=1,
            )
            out = backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attn,
                use_cache=False,
            )
            logits = out.logits[:, P - 1 : -1].float()  # (B, T, V)
            # Cross-entropy on gold tokens, masked to valid positions.
            V = logits.size(-1)
            ce_tok = F.cross_entropy(
                logits.reshape(-1, V),
                gold.reshape(-1),
                reduction="none",
            ).view_as(gold)
            mask = attn.float()
            loss = (ce_tok * mask).sum() / mask.sum().clamp(min=1.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()
            if sched is not None:
                sched.step()

            step += 1
            log_loss += float(loss.item()) * v.size(0)
            log_n += v.size(0)

            if step % args.log_every == 0:
                logger.info(
                    "ep%d step %d  ce=%.4f  (%.1fs)",
                    epoch, step, log_loss / log_n, time.time() - t0,
                )
                log_loss = 0.0
                log_n = 0

            if step % args.val_every == 0:
                av.eval()
                val_fve = _val_fve(
                    av, ar, val_targets, pad_id, eos_id,
                    args.max_new_tokens, args.batch_size,
                )
                av.train()
                logger.info("val @ %d  fve_nrm=%.4f", step, val_fve)
                if val_fve > best_val:
                    best_val = val_fve
                    bad_vals = 0
                    trainable_state = {
                        n: p.detach().cpu()
                        for n, p in av.named_parameters() if p.requires_grad
                    }
                    torch.save(
                        {"trainable": trainable_state, "val_fve_nrm": val_fve, "step": step},
                        best_path,
                    )
                    logger.info("  new best val=%.4f  saved %s", best_val, best_path)
                else:
                    bad_vals += 1
                    logger.info("  no improvement (%d/%d)", bad_vals, args.patience)
                    if args.patience > 0 and bad_vals >= args.patience:
                        logger.info("early stop: %d val checks without improvement", bad_vals)
                        stop_training = True
                        break
        if stop_training:
            break

    # Final val.
    av.eval()
    val_fve = _val_fve(
        av, ar, val_targets, pad_id, eos_id, args.max_new_tokens, args.batch_size,
    )
    logger.info("final val fve_nrm=%.4f  best=%.4f", val_fve, best_val)


if __name__ == "__main__":
    main()
