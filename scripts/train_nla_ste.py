"""Straight-through Gumbel/argmax SFT for SRT-NLA.

Trains directly on the discrete-eval objective:

    rollout:  inputs_embeds = [proj(v)]
    for k in 1..T:
        forward backbone(inputs_embeds), get logits at last pos
        chosen_id = argmax(logits)
        soft prob = softmax(logits / tau)
        STE: onehot(chosen) + softprob - softprob.detach()
        next_emb = onehot @ E   (E = frozen embed matrix)
        append next_emb
    final forward on full sequence with output_hidden_states
    h_last  = hidden_states[L] at last position
    loss    = 1 - fve_nrm(h_last, v_target)

Forward = discrete argmax (matches eval). Backward = soft probs (gradients
flow into proj through the embedding matrix, training proj to bend
generation toward sequences that *reproduce* v_target).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from srt.nla import ActivationVerbalizer, NLAConfig

logger = logging.getLogger("train_nla_ste")

_DTYPE = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", required=True, type=Path)
    p.add_argument("--pairs", required=True, type=Path)
    p.add_argument("--init-from", type=Path, default=None)
    p.add_argument("--num-prefix-tokens", type=int, default=0)
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--rollout-len", type=int, default=64,
                   help="number of tokens to roll out (must equal target position)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--val-vectors", type=int, default=256)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="enable HF gradient checkpointing on backbone")
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def _load_targets(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return torch.stack([a[-1] for a in obj["activations"]])


def _load_pairs(path: Path) -> list[dict]:
    recs = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _fve_nrm(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    hn = F.normalize(h.float(), dim=-1)
    vn = F.normalize(v.float(), dim=-1)
    return 1.0 - ((hn - vn) ** 2).sum(dim=-1) / 2.0


def ste_rollout(
    av: ActivationVerbalizer,
    backbone,
    v: torch.Tensor,
    T: int,
    layer: int,
    temperature: float,
    embed_matrix: torch.Tensor,
    backbone_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autoregressively roll out T tokens with STE.

    Returns (h_last (B,d) at layer L from final forward, gen_ids (B,T)).
    """
    B = v.size(0)
    device = v.device

    prefix = av._inject_prefix(v)  # (B, P, d) in backbone dtype
    cur = prefix
    P = prefix.size(1)
    gen_ids_list = []

    for k in range(T):
        attn = torch.ones(cur.shape[:2], dtype=torch.long, device=device)
        out = backbone(
            inputs_embeds=cur,
            attention_mask=attn,
            output_hidden_states=False,
            use_cache=False,
        )
        logits = out.logits[:, -1, :]  # (B, V)
        probs = F.softmax(logits.float() / temperature, dim=-1)
        chosen = probs.argmax(dim=-1)  # (B,)
        gen_ids_list.append(chosen.detach())
        V = probs.size(-1)
        hard = F.one_hot(chosen, V).float()
        onehot = hard + probs - probs.detach()  # STE
        next_emb = (onehot @ embed_matrix.float()).to(backbone_dtype).unsqueeze(1)  # (B,1,d)
        cur = torch.cat([cur, next_emb], dim=1)

    # Final forward with hidden_states to read layer L
    attn = torch.ones(cur.shape[:2], dtype=torch.long, device=device)
    out_final = backbone(
        inputs_embeds=cur,
        attention_mask=attn,
        output_hidden_states=True,
        use_cache=False,
    )
    h_last = out_final.hidden_states[layer][:, -1, :]  # (B, d)
    gen_ids = torch.stack(gen_ids_list, dim=1)  # (B, T)
    return h_last, gen_ids


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _DTYPE[args.dtype]

    cfg = NLAConfig(
        backbone_id=args.backbone,
        backbone_dtype=args.dtype,
        extraction_layer=args.layer,
        num_prefix_tokens=args.num_prefix_tokens,
    )

    logger.info("loading backbone %s", args.backbone)
    backbone = AutoModelForCausalLM.from_pretrained(args.backbone, torch_dtype=dtype)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.to(device).eval()
    if args.grad_checkpoint:
        backbone.gradient_checkpointing_enable()
        # gradient checkpointing requires use_cache=False (we already do)
    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    av = ActivationVerbalizer(cfg, backbone=backbone, tokenizer=tok).to(device)
    trainable = [p for p in av.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    logger.info("AV num_prefix=%d trainable=%s", args.num_prefix_tokens, f"{n_trainable:,}")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    if args.init_from is not None:
        logger.info("warm-start AV from %s", args.init_from)
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        state = ckpt["trainable"] if isinstance(ckpt, dict) and "trainable" in ckpt else ckpt
        own = av.state_dict()
        compat = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        missing, unexpected = av.load_state_dict(compat, strict=False)
        logger.info("warm-start: %d/%d keys (missing=%d unexpected=%d)",
                    len(compat), len(state), len(missing), len(unexpected))

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

    embed_matrix = backbone.get_input_embeddings().weight  # (V,d) frozen
    eos_id = tok.eos_token_id

    steps_per_epoch = max(1, (len(train_idx) + args.batch_size - 1) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    sched = None
    if args.lr_schedule == "cosine":
        warmup = max(0, args.warmup_steps)

        def _lr_lambda(s: int) -> float:
            if warmup > 0 and s < warmup:
                return (s + 1) / float(warmup)
            progress = (s - warmup) / max(1, total_steps - warmup)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        logger.info("cosine LR: total_steps=%d warmup=%d peak=%g", total_steps, warmup, args.lr)

    @torch.no_grad()
    def run_val() -> float:
        av.eval()
        fve_sum = 0.0
        n = 0
        for i in range(0, len(val_pairs), args.batch_size):
            batch = val_pairs[i : i + args.batch_size]
            v = pool[[r["target_idx"] for r in batch]].to(device)
            # Use AV.generate (HF) for eval — matches downstream usage.
            gen_ids = av.generate(v, max_new_tokens=args.rollout_len, do_sample=False)
            B_, T_ = gen_ids.shape
            is_eos = gen_ids == eos_id
            pos = torch.arange(T_, device=device).unsqueeze(0).expand(B_, T_)
            first_eos = torch.where(
                is_eos.any(dim=-1, keepdim=True),
                is_eos.float().argmax(dim=-1, keepdim=True).long(),
                torch.full((B_, 1), T_, device=device, dtype=torch.long),
            )
            gen_attn = (pos <= first_eos).long()
            out2 = backbone(
                input_ids=gen_ids, attention_mask=gen_attn,
                output_hidden_states=True, use_cache=False,
            )
            # last valid pos per row
            lengths = gen_attn.sum(dim=-1)
            last = (lengths - 1).clamp(min=0)
            h2 = out2.hidden_states[args.layer][torch.arange(B_, device=device), last]
            fve_sum += float(_fve_nrm(h2, v).sum().item())
            n += B_
        av.train()
        return fve_sum / max(n, 1)

    best_val = -1.0
    best_path = args.out / "best_av.pt"
    bad = 0
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
            v = pool[[r["target_idx"] for r in batch]].to(device)
            h_last, _gen = ste_rollout(
                av, backbone, v, args.rollout_len, args.layer,
                args.temperature, embed_matrix, dtype,
            )
            fve = _fve_nrm(h_last, v)
            loss = (1.0 - fve).mean()
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
                    "ep%d step %d  loss=%.4f  train_fve=%.4f  (%.1fs)",
                    epoch, step, log_loss / log_n, 1.0 - log_loss / log_n,
                    time.time() - t0,
                )
                log_loss = 0.0
                log_n = 0

            if step % args.val_every == 0:
                v_fve = run_val()
                logger.info("val @ %d  disc_fve=%.4f", step, v_fve)
                if v_fve > best_val:
                    best_val = v_fve
                    bad = 0
                    state = {n: p.detach().cpu() for n, p in av.named_parameters() if p.requires_grad}
                    torch.save({"trainable": state, "step": step, "val_disc": v_fve,
                                "cfg": vars(args)}, best_path)
                    logger.info("  new best %.4f  saved %s", v_fve, best_path)
                else:
                    bad += 1
                    logger.info("  no improvement (%d/%d)", bad, args.patience)
                    if args.patience > 0 and bad >= args.patience:
                        logger.info("early stop")
                        return

    v_fve = run_val()
    logger.info("final disc_fve=%.4f  best=%.4f", v_fve, best_val)


if __name__ == "__main__":
    main()
