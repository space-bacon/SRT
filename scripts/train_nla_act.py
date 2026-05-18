"""Direct-activation SFT for SRT-NLA.

Replaces CE-on-gold-tokens with a *direct activation matching loss*:

    forward: backbone([proj(v_t), gold_ids])
    h_last:  hidden_states[L] at last valid position
    loss   = 1 - fve_nrm(h_last, v_t)

Diagnostic ``scripts/oracle_check.py`` confirms that gold_ids reproduce
v_t with median fve_nrm ~ 0.9999 under teacher-forcing with the original
BOS context. The only thing standing between us and that ceiling is the
position-0 embedding — proj(v_t) — which must approximate the BOS
embedding well enough that the rest of the context (the gold tokens
themselves carry essentially all the activation information).

Usage:
    python scripts/train_nla_act.py \\
        --targets artifacts/nla/targets_q7b_L20_seq64_10k.pt \\
        --pairs   artifacts/nla/curated/gold_pairs_seq64.jsonl \\
        --init-from artifacts/nla/sft_iter3b_npref8/best_av.pt \\
        --num-prefix-tokens 0 \\
        --epochs 3 --batch-size 16 --lr 3e-4 \\
        --val-every 200 --val-vectors 512 \\
        --out artifacts/nla/act_seq64_np0
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

from srt.nla import ActivationVerbalizer, NLAConfig

logger = logging.getLogger("train_nla_act")

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", required=True, type=Path)
    p.add_argument("--pairs", required=True, type=Path)
    p.add_argument("--init-from", type=Path, default=None)
    p.add_argument("--num-prefix-tokens", type=int, default=0)
    p.add_argument("--prefix-mode", choices=["static", "mlp"], default="static")
    p.add_argument("--prefix-mlp-hidden", type=int, default=256)
    p.add_argument("--num-inject-slots", type=int, default=1,
                   help="number of slots that hold proj(v); >1 = multi-position injection")
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--max-seq-len", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--val-vectors", type=int, default=512)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--ce-weight", type=float, default=0.0,
                   help="weight on CE-on-gold-tokens loss (hybrid mode)")
    p.add_argument("--act-weight", type=float, default=1.0,
                   help="weight on (1 - fve_nrm) activation loss")
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


def _last_token_h(hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """hidden: (B, T, d). attn: (B, T) 1=valid. Returns h at last valid position per row."""
    lengths = attn.sum(dim=-1)  # (B,)
    last_idx = (lengths - 1).clamp(min=0)
    return hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]


def _fve_nrm(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    hn = F.normalize(h.float(), dim=-1)
    vn = F.normalize(v.float(), dim=-1)
    return 1.0 - ((hn - vn) ** 2).sum(dim=-1) / 2.0


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
        num_prefix_tokens=args.num_prefix_tokens,
        prefix_mode=args.prefix_mode,
        prefix_mlp_hidden=args.prefix_mlp_hidden,
        num_inject_slots=args.num_inject_slots,
    )

    logger.info("loading backbone %s", args.backbone)
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

    trainable = [p for p in av.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    logger.info("AV num_prefix_tokens=%d  trainable=%s", args.num_prefix_tokens, f"{n_trainable:,}")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    if args.init_from is not None:
        logger.info("warm-start AV from %s", args.init_from)
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        state = ckpt["trainable"] if isinstance(ckpt, dict) and "trainable" in ckpt else ckpt
        # drop shape-incompatible keys
        own = av.state_dict()
        compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        missing, unexpected = av.load_state_dict(compatible, strict=False)
        logger.info("warm-start: %d/%d keys (missing=%d unexpected=%d)",
                    len(compatible), len(state), len(missing), len(unexpected))

    pool = _load_targets(args.targets).to(device)
    pairs = _load_pairs(args.pairs)
    logger.info("loaded %d pairs; target pool N=%d", len(pairs), pool.size(0))

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(pairs), generator=g).tolist()
    n_val = max(1, int(len(pairs) * args.val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    val_pairs = [pairs[i] for i in val_idx[: args.val_vectors]]
    logger.info("train pairs=%d  val pairs=%d", len(train_idx), len(val_pairs))

    pad_id = tok.pad_token_id

    steps_per_epoch = max(1, (len(train_idx) + args.batch_size - 1) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    sched = None
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
        logger.info("cosine LR: total_steps=%d warmup=%d peak_lr=%g", total_steps, warmup, args.lr)

    @torch.no_grad()
    def run_val() -> tuple[float, float]:
        """Return (teacher_forced_fve, discrete_fve). Discrete = AV.generate + AR."""
        av.eval()
        tf_sum = 0.0
        disc_sum = 0.0
        n = 0
        eos_id = tok.eos_token_id
        for i in range(0, len(val_pairs), args.batch_size):
            batch = val_pairs[i : i + args.batch_size]
            v, gold, attn = _collate(batch, pool, pad_id, args.max_seq_len, device)
            # --- teacher-forced ---
            prefix = av._inject_prefix(v)
            P = prefix.size(1)
            tok_embeds = backbone.get_input_embeddings()(gold)
            inputs_embeds = torch.cat([prefix, tok_embeds], dim=1)
            full_attn = torch.cat(
                [torch.ones(prefix.shape[:2], dtype=attn.dtype, device=device), attn], dim=1,
            )
            out = backbone(
                inputs_embeds=inputs_embeds, attention_mask=full_attn,
                output_hidden_states=True, use_cache=False,
            )
            h_all = out.hidden_states[args.layer]
            h_last = _last_token_h(h_all, full_attn)
            tf_sum += float(_fve_nrm(h_last, v).sum().item())
            # --- discrete: AV greedy-generate, run frozen backbone on generated ids ---
            gen_ids = av.generate(v, max_new_tokens=args.max_seq_len, do_sample=False)
            B, T = gen_ids.shape
            is_eos = gen_ids == eos_id
            pos = torch.arange(T, device=gen_ids.device).unsqueeze(0).expand(B, T)
            first_eos = torch.where(
                is_eos.any(dim=-1, keepdim=True),
                is_eos.float().argmax(dim=-1, keepdim=True).long(),
                torch.full((B, 1), T, device=gen_ids.device, dtype=torch.long),
            )
            gen_attn = (pos <= first_eos).long()
            out2 = backbone(
                input_ids=gen_ids, attention_mask=gen_attn,
                output_hidden_states=True, use_cache=False,
            )
            h2_last = _last_token_h(out2.hidden_states[args.layer], gen_attn)
            disc_sum += float(_fve_nrm(h2_last, v).sum().item())
            n += v.size(0)
        av.train()
        return tf_sum / max(n, 1), disc_sum / max(n, 1)

    best_val = -1.0
    best_path = args.out / "best_av.pt"
    bad_vals = 0
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
            prefix = av._inject_prefix(v)
            P = prefix.size(1)
            tok_embeds = backbone.get_input_embeddings()(gold)
            inputs_embeds = torch.cat([prefix, tok_embeds], dim=1)
            full_attn = torch.cat(
                [torch.ones(prefix.shape[:2], dtype=attn.dtype, device=device), attn], dim=1,
            )
            out = backbone(
                inputs_embeds=inputs_embeds, attention_mask=full_attn,
                output_hidden_states=True, use_cache=False,
            )
            h_all = out.hidden_states[args.layer]
            h_last = _last_token_h(h_all, full_attn)
            fve = _fve_nrm(h_last, v)
            act_loss = (1.0 - fve).mean()
            if args.ce_weight > 0.0:
                logits = out.logits[:, P - 1 : -1].float()  # predict gold[t] from prefix..gold[t-1]
                V = logits.size(-1)
                ce_tok = F.cross_entropy(
                    logits.reshape(-1, V), gold.reshape(-1), reduction="none",
                ).view_as(gold)
                mask = attn.float()
                ce_loss = (ce_tok * mask).sum() / mask.sum().clamp(min=1.0)
            else:
                ce_loss = torch.tensor(0.0, device=device)
            loss = args.act_weight * act_loss + args.ce_weight * ce_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()
            if sched is not None:
                sched.step()

            step += 1
            log_loss += float(act_loss.item()) * v.size(0)
            log_n += v.size(0)

            if step % args.log_every == 0:
                logger.info(
                    "ep%d step %d  act_loss=%.4f  ce=%.4f  train_fve=%.4f  (%.1fs)",
                    epoch, step, log_loss / log_n, float(ce_loss.item()),
                    1.0 - log_loss / log_n,
                    time.time() - t0,
                )
                log_loss = 0.0
                log_n = 0

            if step % args.val_every == 0:
                tf, disc = run_val()
                logger.info("val @ %d  tf_fve=%.4f  disc_fve=%.4f", step, tf, disc)
                sel = disc if args.ce_weight > 0.0 else tf
                if sel > best_val:
                    best_val = sel
                    bad_vals = 0
                    state = {n: p.detach().cpu() for n, p in av.named_parameters() if p.requires_grad}
                    torch.save({"trainable": state, "step": step, "val_tf": tf, "val_disc": disc, "cfg": vars(args)}, best_path)
                    logger.info("  new best (sel=%.4f tf=%.4f disc=%.4f) saved %s", sel, tf, disc, best_path)
                else:
                    bad_vals += 1
                    logger.info("  no improvement (%d/%d)", bad_vals, args.patience)
                    if args.patience > 0 and bad_vals >= args.patience:
                        logger.info("early stop")
                        return

    tf, disc = run_val()
    logger.info("final tf_fve=%.4f  disc_fve=%.4f  best=%.4f", tf, disc, best_val)


if __name__ == "__main__":
    main()
