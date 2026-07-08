"""Draft-conditioned SFT for SRT-NLA: retrieval-then-edit decoding.

Attack on the greedy gap (leverage.md move 3). Zero-training NN retrieval
scores rho_cen ~ 0.71 while the trained AV's greedy decode scores ~ 0.28.
This trainer conditions the AV on the retrieved neighbour's text (the
"draft") in addition to the injected vector v:

    [proj(v), prefix_embeds..., draft_ids, SEP, gold_ids]

CE is applied to gold_ids only. The worst case for the trained model is
copying the draft (which already matches the NN baseline); CE toward the
gold text teaches it to *edit* the draft toward v. At inference the draft
is the nearest-neighbour of v in a retrieval pool by centered cosine, so
the whole pipeline needs nothing beyond what deployment already has.

Drafts during training are leave-one-out nearest neighbours within the
train split (centered cosine, mu = train mean), so the train/deploy draft
distributions match. ``--draft-dropout`` occasionally blanks the draft so
the model retains draft-free capability.

Validation reports, per val target:
  - draft_cen   : centered fve of the draft text itself (copy baseline)
  - greedy_cen  : draft-conditioned greedy decode
  - bestK_cen   : draft-conditioned best-of-K sampled (oracle-scored)
Selection metric: greedy_cen (the quantity this experiment exists to move).

Usage (Qwen2.5-7B L20, warm-start from the CE np16 checkpoint):
    python scripts/train_nla_draft.py \\
        --targets artifacts/nla/targets_q7b_L20_seq64_10k.pt \\
        --pairs   artifacts/nla/curated/gold_pairs_seq64.jsonl \\
        --init-from artifacts/nla/ce_seq64_np16/best_av.pt \\
        --num-prefix-tokens 16 \\
        --epochs 2 --batch-size 16 --lr 1e-4 \\
        --val-every 200 --val-vectors 200 --val-bestof 8 \\
        --out artifacts/nla/draft_seq64_np16
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from srt.nla import ActivationVerbalizer, NLAConfig, load_frozen_backbone

logger = logging.getLogger("train_nla_draft")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", required=True, type=Path)
    p.add_argument("--pairs", required=True, type=Path)
    p.add_argument("--init-from", type=Path, default=None)
    p.add_argument("--num-prefix-tokens", type=int, default=16)
    p.add_argument("--num-inject-slots", type=int, default=1)
    p.add_argument("--inject-norm", choices=["none", "embed"], default="none")
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--max-seq-len", type=int, default=64)
    p.add_argument("--max-draft-len", type=int, default=64,
                   help="drafts truncated to this many tokens. Use full gold "
                        "length: truncation destroys the copy baseline on "
                        "position-decorrelating backbones (gemma-4 L47: "
                        "fve 0.53 @48 tok vs 0.69 full).")
    p.add_argument("--separator", default="\n\u2192 ",
                   help="text placed between draft and gold/generation")
    p.add_argument("--draft-dropout", type=float, default=0.1,
                   help="fraction of training rows whose draft is blanked")
    p.add_argument("--prepend-bos", action="store_true",
                   help="prepend BOS in every prefix-free re-encode (val "
                        "scoring). REQUIRED on BOS-sensitive backbones "
                        "(gemma-4).")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--val-vectors", type=int, default=200)
    p.add_argument("--val-bestof", type=int, default=8)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def _load_targets(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "targets" in obj:
        return obj["targets"].float()
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


def _fve_nrm(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """fve units: 0.5*(1+cos). Matches nla_anchors.py / the paper frame.
    (train_nla_act.py's same-named helper returns RAW COSINE — do not
    compare its val numbers to anchors without converting.)"""
    hn = F.normalize(h.float(), dim=-1)
    vn = F.normalize(v.float(), dim=-1)
    return 0.5 * (1.0 + (hn * vn).sum(dim=-1))


def _centered_nn(
    query: torch.Tensor, pool: torch.Tensor, mu: torch.Tensor, exclude: int | None = None
) -> int:
    """Index of nearest pool row to query by centered cosine."""
    q = F.normalize((query - mu).float(), dim=-1)
    P = F.normalize((pool - mu).float(), dim=-1)
    sims = P @ q
    if exclude is not None:
        sims[exclude] = -2.0
    return int(sims.argmax().item())


def _eos_attn(gen_ids: torch.Tensor, eos_id: int | None) -> torch.Tensor:
    """Attention mask that keeps positions up to and including the first EOS
    (all positions when a row has no EOS). Matches train_nla_act.py's val."""
    B, T = gen_ids.shape
    if eos_id is None:
        return torch.ones_like(gen_ids)
    is_eos = gen_ids == eos_id
    pos = torch.arange(T, device=gen_ids.device).unsqueeze(0).expand(B, T)
    first_eos = torch.where(
        is_eos.any(dim=-1, keepdim=True),
        is_eos.float().argmax(dim=-1, keepdim=True).long(),
        torch.full((B, 1), T, device=gen_ids.device, dtype=torch.long),
    )
    return (pos <= first_eos).long()


@torch.no_grad()
def _encode_last_h(
    backbone, ids: torch.Tensor, layer: int, attn: torch.Tensor | None = None,
    bos_id: int | None = None,
) -> torch.Tensor:
    """Prefix-free re-encode: (B, T) ids -> (B, d) hidden at `layer`, last
    valid (attended) position per row. ``bos_id`` prepends a BOS column
    (required on BOS-sensitive backbones such as gemma-4)."""
    if attn is None:
        attn = torch.ones_like(ids)
    if bos_id is not None:
        col = torch.full((ids.size(0), 1), bos_id, dtype=ids.dtype, device=ids.device)
        ids = torch.cat([col, ids], dim=1)
        attn = torch.cat([torch.ones_like(col), attn], dim=1)
    out = backbone(input_ids=ids, attention_mask=attn,
                   output_hidden_states=True, use_cache=False)
    h = out.hidden_states[layer]
    last_idx = (attn.sum(-1) - 1).clamp(min=0).long()
    rows = torch.arange(h.size(0), device=h.device)
    return h[rows, last_idx].float()


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
        num_inject_slots=args.num_inject_slots,
        inject_norm=args.inject_norm,
    )

    logger.info("loading backbone %s", args.backbone)
    backbone, tok = load_frozen_backbone(args.backbone, args.dtype, device=device)
    pad_id = tok.pad_token_id
    sep_ids = tok(args.separator, add_special_tokens=False)["input_ids"]
    logger.info("separator %r -> %s", args.separator, sep_ids)

    av = ActivationVerbalizer(cfg, backbone=backbone, tokenizer=tok).to(device)
    trainable = [p for p in av.parameters() if p.requires_grad]
    logger.info("trainable=%s", f"{sum(p.numel() for p in trainable):,}")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    if args.init_from is not None:
        logger.info("warm-start AV from %s", args.init_from)
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        state = ckpt["trainable"] if isinstance(ckpt, dict) and "trainable" in ckpt else ckpt
        own = av.state_dict()
        compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        av.load_state_dict(compatible, strict=False)
        logger.info("warm-start: %d/%d keys", len(compatible), len(state))

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

    # Retrieval pool = TRAIN split only (deployment never retrieves the answer).
    train_tidx = torch.tensor([pairs[i]["target_idx"] for i in train_idx], dtype=torch.long)
    train_pool = pool[train_tidx.to(pool.device)]          # (Ntr, d)
    mu = train_pool.mean(0)
    logger.info("retrieval pool=%d  ||mu||=%.1f", train_pool.size(0), float(mu.norm()))

    # Precompute leave-one-out train drafts once (Ntr x Ntr centered cosine).
    logger.info("precomputing leave-one-out train drafts ...")
    tp_n = F.normalize((train_pool - mu), dim=-1)
    sims = tp_n @ tp_n.T
    sims.fill_diagonal_(-2.0)
    train_nn = sims.argmax(dim=1).tolist()                 # per-train-row NN row
    del sims
    # Val drafts (query val target against full train pool).
    val_draft_ids: list[list[int]] = []
    for r in val_pairs:
        vq = pool[r["target_idx"]]
        j = _centered_nn(vq, train_pool, mu)
        val_draft_ids.append(pairs[train_idx[j]]["gold_ids"][: args.max_draft_len])

    steps_per_epoch = max(1, (len(train_idx) + args.batch_size - 1) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    sched = None
    if args.lr_schedule == "cosine":
        import math
        warmup = max(0, args.warmup_steps)

        def _lr_lambda(s: int) -> float:
            if warmup > 0 and s < warmup:
                return (s + 1) / float(warmup)
            progress = min(max((s - warmup) / max(1, total_steps - warmup), 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    def _collate_draft(batch_rows: list[int], drop_mask: list[bool]):
        """Rows are indices into train_idx. Returns (v, ids, labels, attn)."""
        seqs, labels = [], []
        for k, row in enumerate(batch_rows):
            r = pairs[train_idx[row]]
            gold = r["gold_ids"][: args.max_seq_len]
            if drop_mask[k]:
                ctx = list(sep_ids)
            else:
                draft = pairs[train_idx[train_nn[row]]]["gold_ids"][: args.max_draft_len]
                ctx = list(draft) + list(sep_ids)
            seqs.append(ctx + list(gold))
            labels.append([-100] * len(ctx) + list(gold))
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
        lab = torch.full((len(seqs), T), -100, dtype=torch.long)
        attn = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, (s, l) in enumerate(zip(seqs, labels)):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            lab[i, : len(l)] = torch.tensor(l, dtype=torch.long)
            attn[i, : len(s)] = 1
        v = pool[[pairs[train_idx[row]]["target_idx"] for row in batch_rows]]
        return v.to(device), ids.to(device), lab.to(device), attn.to(device)

    @torch.no_grad()
    def run_val() -> dict[str, float]:
        av.eval()
        bos_re = tok.bos_token_id if args.prepend_bos else None
        draft_sum = greedy_sum = best_sum = 0.0
        n = 0
        K = max(1, args.val_bestof)
        for r, dids in zip(val_pairs, val_draft_ids):
            v = pool[r["target_idx"]].unsqueeze(0)                     # (1, d)
            v_c = v - mu
            ctx = torch.tensor([list(dids) + list(sep_ids)], dtype=torch.long, device=device)
            # copy baseline: encode the draft text alone
            h_d = _encode_last_h(backbone, torch.tensor([dids], device=device),
                                 args.layer, bos_id=bos_re)
            draft_sum += float(_fve_nrm(h_d - mu, v_c).item())
            # greedy, draft-conditioned
            gen = av.generate(v, max_new_tokens=args.max_seq_len, do_sample=False,
                              context_ids=ctx)
            h_g = _encode_last_h(backbone, gen, args.layer,
                                 attn=_eos_attn(gen, tok.eos_token_id), bos_id=bos_re)
            greedy_sum += float(_fve_nrm(h_g - mu, v_c).item())
            # best-of-K sampled, draft-conditioned (per-target batch: no padding)
            if K > 1:
                v_rep = v.expand(K, -1)
                ctx_rep = ctx.expand(K, -1)
                genK = av.generate(v_rep, max_new_tokens=args.max_seq_len,
                                   do_sample=True, temperature=1.0, context_ids=ctx_rep)
                h_k = _encode_last_h(backbone, genK, args.layer,
                                     attn=_eos_attn(genK, tok.eos_token_id), bos_id=bos_re)
                best_sum += float(_fve_nrm(h_k - mu, (v_c).expand(K, -1)).max().item())
            n += 1
        av.train()
        return {
            "draft_cen": draft_sum / max(n, 1),
            "greedy_cen": greedy_sum / max(n, 1),
            f"best{K}_cen": best_sum / max(n, 1),
        }

    best_val = -1.0
    best_path = args.out / "best_av.pt"
    bad_vals = 0
    step = 0
    t0 = time.time()
    log_loss, log_n = 0.0, 0
    rng = torch.Generator().manual_seed(args.seed + 1)

    for epoch in range(1, args.epochs + 1):
        torch.manual_seed(args.seed + epoch)
        shuffled = torch.randperm(len(train_idx)).tolist()

        for i in range(0, len(shuffled), args.batch_size):
            rows = shuffled[i : i + args.batch_size]
            drop = (torch.rand(len(rows), generator=rng) < args.draft_dropout).tolist()
            v, ids, lab, attn = _collate_draft(rows, drop)
            prefix = av._inject_prefix(v)
            P = prefix.size(1)
            tok_embeds = backbone.get_input_embeddings()(ids)
            inputs_embeds = torch.cat([prefix, tok_embeds], dim=1)
            full_attn = torch.cat(
                [torch.ones(prefix.shape[:2], dtype=attn.dtype, device=device), attn], dim=1
            )
            out = backbone(inputs_embeds=inputs_embeds, attention_mask=full_attn,
                           use_cache=False)
            logits = out.logits[:, P - 1 : -1].float()      # predicts ids[t]
            V = logits.size(-1)
            ce = F.cross_entropy(
                logits.reshape(-1, V), lab.reshape(-1), ignore_index=-100
            )
            opt.zero_grad(set_to_none=True)
            ce.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()
            if sched is not None:
                sched.step()

            step += 1
            log_loss += float(ce.item()) * v.size(0)
            log_n += v.size(0)
            if step % args.log_every == 0:
                logger.info("ep%d step %d  ce=%.4f  (%.1fs)",
                            epoch, step, log_loss / log_n, time.time() - t0)
                log_loss, log_n = 0.0, 0

            if step % args.val_every == 0:
                m = run_val()
                logger.info("val @ %d  %s", step,
                            "  ".join(f"{k}={x:.4f}" for k, x in m.items()))
                sel = m["greedy_cen"]
                if sel > best_val:
                    best_val = sel
                    bad_vals = 0
                    state = {n_: p.detach().cpu() for n_, p in av.named_parameters()
                             if p.requires_grad}
                    torch.save({"trainable": state, "step": step, "val": m,
                                "cfg": vars(args)}, best_path)
                    logger.info("  new best greedy_cen=%.4f saved %s", sel, best_path)
                else:
                    bad_vals += 1
                    logger.info("  no improvement (%d/%d)", bad_vals, args.patience)
                    if args.patience > 0 and bad_vals >= args.patience:
                        logger.info("early stop")
                        return

    m = run_val()
    logger.info("final %s  best_greedy_cen=%.4f",
                "  ".join(f"{k}={x:.4f}" for k, x in m.items()), best_val)


if __name__ == "__main__":
    main()
