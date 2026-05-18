"""Decoding-gap fix: best-of-K self-distillation + NN-hard-neg contrastive
   + cosine temperature schedule, for SRT-NLA verbalizer.

Rationale (see paper_nla.md):
    Adapter best-of-64 saturates the Qwen paraphrase ceiling at rho_cen ~ 0.99
    while greedy stays at rho_cen ~ 0.28 (worse than zero-training NN
    retrieval at 0.71). The capacity is there; the policy can't *pick*
    its best mode at decode time. Three concurrent fixes, each cheap and
    independently ablate-able via its loss weight:

      (a) BoK self-distillation (RFT / ReST style)
          Sample K rollouts at temperature T_now, score each by centered
          fve_nrm, pick the per-target argmax winner, teacher-force CE
          on the winner's tokens. Greedy slowly imitates the best sample.

      (b) InfoNCE on NN-retrieved hard negatives
          For each (v, gold_pos) pair, precompute the top-J nearest
          neighbours of v in centered cosine — these are the texts that
          NN-retrieval would confuse with v. The model must rank
          logp(gold_pos | v) above logp(gold_neg_j | v) for all J.

      (c) Cosine temperature schedule
          T anneals from --temp-start to --temp-end. Early training
          explores; late training matches the greedy mode the winner
          tokens were drawn from at low T.

Loss = alpha * L_bok + beta * L_ctr  +  (optional) gamma * L_act_on_winner

Usage (smoke):
    python scripts/train_nla_bok.py \\
        --targets artifacts/nla/targets_q7b_L20_seq64_30k_seed1.pt \\
        --pairs   artifacts/nla/curated/gold_pairs_seq64.jsonl \\
        --init-from artifacts/nla/ce_seq64_np16/best_av.pt \\
        --num-prefix-tokens 16 --num-inject-slots 1 \\
        --epochs 1 --batch-size 4 --samples-per-v 8 --hard-negs 4 \\
        --temp-start 1.0 --temp-end 0.3 \\
        --alpha-bok 1.0 --beta-ctr 0.5 --gamma-act 0.0 \\
        --val-every 100 --val-vectors 256 --pool-size 2000 \\
        --out artifacts/nla/bok_seq64_np16
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

logger = logging.getLogger("train_nla_bok")

_DTYPE = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


# ────────────────────────────── args ──────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # data / init
    p.add_argument("--targets", required=True, type=Path)
    p.add_argument("--pairs",   required=True, type=Path)
    p.add_argument("--init-from", type=Path, default=None)
    # AV layout
    p.add_argument("--num-prefix-tokens", type=int, default=16)
    p.add_argument("--prefix-mode", choices=["static", "mlp"], default="static")
    p.add_argument("--prefix-mlp-hidden", type=int, default=256)
    p.add_argument("--num-inject-slots", type=int, default=1)
    # backbone
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--dtype",    default="bfloat16")
    p.add_argument("--layer",    type=int, default=20)
    # optim
    p.add_argument("--epochs",      type=int, default=1)
    p.add_argument("--batch-size",  type=int, default=4)
    p.add_argument("--lr",          type=float, default=3e-5)
    p.add_argument("--weight-decay",type=float, default=0.0)
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    p.add_argument("--warmup-steps",type=int, default=100)
    # rollout
    p.add_argument("--samples-per-v", type=int, default=8,
                   help="K rollouts per target for best-of-K distillation")
    p.add_argument("--rollout-len",   type=int, default=64)
    p.add_argument("--temp-start",    type=float, default=1.0)
    p.add_argument("--temp-end",      type=float, default=0.3)
    # losses
    p.add_argument("--alpha-bok", type=float, default=1.0)
    p.add_argument("--beta-ctr",  type=float, default=0.5)
    p.add_argument("--gamma-act", type=float, default=0.0,
                   help="optional 1-fve_nrm on the BoK winner's h_last")
    p.add_argument("--hard-negs", type=int, default=4,
                   help="J nearest-neighbour hard-negative gold texts per target")
    p.add_argument("--neg-pool-cap", type=int, default=8192,
                   help="rows used to compute NN sims (centered cos) — cap for memory")
    # eval
    p.add_argument("--pool-size",    type=int, default=2000,
                   help="held-out pool for anisotropy-mean mu (centered fve_nrm)")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--val-every",    type=int, default=200)
    p.add_argument("--val-vectors",  type=int, default=256)
    p.add_argument("--log-every",    type=int, default=10)
    p.add_argument("--patience",     type=int, default=10)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--out",          required=True, type=Path)
    return p.parse_args()


# ────────────────────────────── helpers ──────────────────────────────

def _load_targets(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return torch.stack([a[-1] for a in obj["activations"]]).float()


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


def _last_h(hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    lengths = attn.sum(dim=-1)
    last_idx = (lengths - 1).clamp(min=0).long()
    rows = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[rows, last_idx]


def _pad_to(lst: list[list[int]], pad_id: int, max_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad list-of-int-lists to (B, T). Returns (ids, attn) on CPU."""
    if max_len is None:
        max_len = max(len(x) for x in lst)
    out = torch.full((len(lst), max_len), pad_id, dtype=torch.long)
    msk = torch.zeros((len(lst), max_len), dtype=torch.long)
    for i, x in enumerate(lst):
        x = x[:max_len]
        out[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        msk[i, :len(x)] = 1
    return out, msk


# ────────────────────────── NN hard-neg index ───────────────────────

def build_hardneg_index(
    pool_cen: torch.Tensor,         # (N, d), already centered
    pair_target_idx: torch.Tensor,  # (P,) which row of pool each pair points at
    J: int,
    cap: int,
    device: torch.device,
) -> torch.Tensor:
    """For each pair, return (J,) tensor of *pair-row* indices of nearest
    neighbour targets (excluding self). Done once on startup.

    Uses up to `cap` candidate rows to keep the sim matrix tractable.
    """
    P = pair_target_idx.numel()
    # candidate set = the same pair rows (we draw negs from the train set)
    cand = pair_target_idx.to(device)
    cand_cap = min(cap, P)
    cand = cand[:cand_cap]                       # (C,)
    Vc = pool_cen[pair_target_idx]               # (P, d)
    Cc = pool_cen[cand]                          # (C, d)
    Vn = F.normalize(Vc.to(device).float(), dim=-1)
    Cn = F.normalize(Cc.float(), dim=-1)
    # cosine (P, C); chunk over P for memory
    out = torch.empty((P, J), dtype=torch.long)
    chunk = 256
    for i in range(0, P, chunk):
        sim = Vn[i:i + chunk] @ Cn.t()           # (b, C)
        # mask self (pair i corresponds to candidate row i if i<cand_cap)
        b = sim.size(0)
        for k in range(b):
            gi = i + k
            if gi < cand_cap:
                sim[k, gi] = -1e9
        top = sim.topk(J, dim=1).indices         # (b, J) in [0, C)
        out[i:i + chunk] = top.cpu()
    return out  # (P, J) — values are indices into the pair list (0..cand_cap)


# ────────────────────────────── main ──────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _DTYPE[args.dtype]

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
    backbone = AutoModelForCausalLM.from_pretrained(args.backbone, torch_dtype=dtype)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    eos_id = int(tok.eos_token_id)
    pad_id = int(tok.pad_token_id)

    av = ActivationVerbalizer(cfg, backbone=backbone, tokenizer=tok).to(device)
    trainable = [p for p in av.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    logger.info("AV num_prefix=%d num_inject=%d trainable=%s",
                args.num_prefix_tokens, args.num_inject_slots, f"{n_trainable:,}")
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

    # targets and pairs
    pool_cpu = _load_targets(args.targets)        # (N, d) float32 cpu
    pairs    = _load_pairs(args.pairs)
    logger.info("loaded %d pairs; target pool N=%d", len(pairs), pool_cpu.size(0))

    # anisotropy mean from a held-out pool slice (NOT used as targets)
    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(pool_cpu.size(0), generator=rng)
    mu_idx = perm[: args.pool_size]
    mu = pool_cpu[mu_idx].mean(dim=0).to(device)  # (d,)
    pool_cen_cpu = pool_cpu - mu.cpu()            # (N, d) centered
    logger.info("anisotropy ||mu||=%.4f  (pool_size=%d)", float(mu.norm()), args.pool_size)

    # split
    perm2 = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(args.seed + 1))
    n_val = max(1, int(len(pairs) * args.val_fraction))
    val_idx   = perm2[:n_val].tolist()
    train_idx = perm2[n_val:].tolist()
    val_pairs = [pairs[i] for i in val_idx[:args.val_vectors]]
    logger.info("train=%d  val=%d", len(train_idx), len(val_pairs))

    # hard-neg index (over train pairs only)
    pair_tidx = torch.tensor([pairs[i]["target_idx"] for i in train_idx], dtype=torch.long)
    logger.info("building NN hard-neg index J=%d cap=%d ...", args.hard_negs, args.neg_pool_cap)
    hn = build_hardneg_index(pool_cen_cpu, pair_tidx, args.hard_negs,
                             args.neg_pool_cap, device)
    logger.info("hard-neg index: %s", tuple(hn.shape))

    embed_matrix = backbone.get_input_embeddings().weight  # (V, d) bf16

    # ────────────────── core forward (with grad) ──────────────────
    def _seq_logp_and_h(v_rep: torch.Tensor, gen_ids: torch.Tensor,
                        gen_attn: torch.Tensor, want_hidden: bool):
        """Run AV+backbone over [proj(v), gen[:-1]] and return:
              nll : (B, T)   token-wise -log p
              h_last : (B, d) hidden at layer L at last valid gen position
                       (None if want_hidden=False)
            Grad flows through proj/prefix only.
        """
        B_, T_ = gen_ids.shape
        prefix = av._inject_prefix(v_rep)                              # (B, P, d) bf16
        P_ = prefix.size(1)
        gen_in = gen_ids[:, :-1]
        tok_emb = F.embedding(gen_in, embed_matrix).to(prefix.dtype).detach()
        full_in = torch.cat([prefix, tok_emb], dim=1)
        prefix_attn = torch.ones(B_, P_, dtype=torch.long, device=device)
        full_attn   = torch.cat([prefix_attn, gen_attn[:, :-1]], dim=1)
        out = backbone(inputs_embeds=full_in, attention_mask=full_attn,
                       output_hidden_states=want_hidden, use_cache=False)
        logits = out.logits[:, P_ - 1 : P_ - 1 + T_, :]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        nll = -log_probs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)  # (B, T)
        h_last = None
        if want_hidden:
            full_h = out.hidden_states[args.layer]                      # (B, P+T-1, d)
            gen_h  = full_h[:, P_ - 1 : P_ - 1 + T_, :]                  # (B, T, d) — aligned w/ gen_ids
            h_last = _last_h(gen_h, gen_attn)
        return nll, h_last

    @torch.no_grad()
    def _rollout(v_rep: torch.Tensor, sample: bool, temp: float):
        gen_ids = av.generate(
            v_rep, max_new_tokens=args.rollout_len,
            do_sample=sample, temperature=max(temp, 1e-6), top_p=1.0,
        )
        B_, T_ = gen_ids.shape
        is_eos = gen_ids == eos_id
        pos = torch.arange(T_, device=device).unsqueeze(0).expand(B_, T_)
        first_eos = torch.where(
            is_eos.any(dim=-1, keepdim=True),
            is_eos.float().argmax(dim=-1, keepdim=True).long(),
            torch.full((B_, 1), T_ - 1, device=device, dtype=torch.long),
        )
        gen_attn = (pos <= first_eos).long()
        # score on centered fve
        out = backbone(input_ids=gen_ids, attention_mask=gen_attn,
                       output_hidden_states=True, use_cache=False)
        h = _last_h(out.hidden_states[args.layer], gen_attn).float()
        score_cen = _fve_nrm(h - mu, v_rep.float() - mu)
        score_raw = _fve_nrm(h,        v_rep.float())
        return gen_ids, gen_attn, score_cen, score_raw, h

    @torch.no_grad()
    def run_val() -> tuple[float, float]:
        """Returns (greedy_cen, greedy_raw) on val."""
        av.eval()
        tot_cen = tot_raw = 0.0; n = 0
        for i in range(0, len(val_pairs), args.batch_size):
            batch = val_pairs[i:i + args.batch_size]
            v = pool_cpu[[r["target_idx"] for r in batch]].to(device)
            _g, _a, sc, sr, _h = _rollout(v, sample=False, temp=1.0)
            tot_cen += float(sc.sum().item()); tot_raw += float(sr.sum().item())
            n += v.size(0)
        av.train()
        return tot_cen / max(n, 1), tot_raw / max(n, 1)

    # ────────────────── training loop ──────────────────
    K = args.samples_per_v
    J = args.hard_negs
    steps_per_epoch = max(1, (len(train_idx) + args.batch_size - 1) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs

    def temp_at(step: int) -> float:
        if total_steps <= 1: return args.temp_end
        f = step / max(1, total_steps - 1)
        return args.temp_start + (args.temp_end - args.temp_start) * 0.5 * (1 - math.cos(math.pi * f))

    sched = None
    if args.lr_schedule == "cosine":
        warmup = max(0, args.warmup_steps)
        def _lr_lambda(s):
            if warmup > 0 and s < warmup:
                return (s + 1) / float(warmup)
            prog = (s - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(max(prog, 0.0), 1.0)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        logger.info("cosine LR: total_steps=%d warmup=%d peak=%g", total_steps, warmup, args.lr)

    best_val = -1.0
    best_path = args.out / "best_av.pt"
    bad = 0; step = 0; t0 = time.time()
    log_bok = log_ctr = log_act = log_rew_cen = 0.0; log_n = 0

    for epoch in range(1, args.epochs + 1):
        torch.manual_seed(args.seed + epoch)
        order = torch.randperm(len(train_idx)).tolist()
        for i in range(0, len(order), args.batch_size):
            batch_local = order[i:i + args.batch_size]            # indices into train_idx
            pair_batch  = [pairs[train_idx[j]] for j in batch_local]
            B_ = len(pair_batch)
            v = pool_cpu[[r["target_idx"] for r in pair_batch]].to(device)  # (B, d)

            # ── 1. K rollouts at scheduled temperature, score, pick winner
            T_now = temp_at(step)
            v_rep = v.unsqueeze(1).expand(B_, K, -1).reshape(B_ * K, -1)
            gen_ids, gen_attn, sc_cen, sc_raw, _h = _rollout(v_rep, sample=True, temp=T_now)
            sc_cen_mat = sc_cen.view(B_, K)
            win = sc_cen_mat.argmax(dim=1)                         # (B,)
            row_off = torch.arange(B_, device=device) * K
            win_idx = (row_off + win)
            win_ids  = gen_ids[win_idx]                            # (B, T)
            win_attn = gen_attn[win_idx]                           # (B, T)
            win_score = sc_cen_mat[torch.arange(B_, device=device), win]

            # ── 2. BoK distillation: CE on winner's tokens
            nll_w, h_w = _seq_logp_and_h(v, win_ids, win_attn,
                                         want_hidden=(args.gamma_act > 0))
            mask_w = win_attn.float()
            tok_per = mask_w.sum(dim=-1).clamp(min=1.0)
            L_bok = (nll_w * mask_w).sum(dim=-1) / tok_per
            L_bok = L_bok.mean()

            # ── 3. InfoNCE on NN hard-negs (gold-text discrimination)
            L_ctr = torch.zeros((), device=device)
            if args.beta_ctr > 0 and J > 0:
                # gather negs: hn[train_idx[batch_local[k]]] → pair-list indices into train_idx
                neg_rows: list[list[int]] = []      # per-pos list of J neg gold_ids
                for k, bl in enumerate(batch_local):
                    neg_pair_ix = hn[bl].tolist()          # J indices into train_idx
                    for npix in neg_pair_ix:
                        neg_rows.append(pairs[train_idx[npix]]["gold_ids"])
                pos_rows = [r["gold_ids"] for r in pair_batch]
                # concat → (B*(1+J), T*)
                all_rows = []
                for k in range(B_):
                    all_rows.append(pos_rows[k])
                    all_rows.extend(neg_rows[k * J:(k + 1) * J])
                max_t = max(len(x) for x in all_rows)
                max_t = min(max_t, args.rollout_len)
                ids_cpu, msk_cpu = _pad_to(all_rows, pad_id, max_t)
                ctr_ids  = ids_cpu.to(device)
                ctr_attn = msk_cpu.to(device)
                # v_rep over (1+J) — same v repeated
                v_ctr = v.unsqueeze(1).expand(B_, 1 + J, -1).reshape(B_ * (1 + J), -1)
                nll_c, _ = _seq_logp_and_h(v_ctr, ctr_ids, ctr_attn, want_hidden=False)
                m = ctr_attn.float()
                logp = -(nll_c * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1.0)  # (B*(1+J),)
                logits = logp.view(B_, 1 + J)
                tgt = torch.zeros(B_, dtype=torch.long, device=device)
                L_ctr = F.cross_entropy(logits, tgt)

            # ── 4. (optional) direct activation match on winner
            L_act = torch.zeros((), device=device)
            if args.gamma_act > 0 and h_w is not None:
                L_act = (1.0 - _fve_nrm(h_w.float() - mu, v.float() - mu)).mean()

            loss = args.alpha_bok * L_bok + args.beta_ctr * L_ctr + args.gamma_act * L_act

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()
            if sched is not None:
                sched.step()
            step += 1
            log_bok += float(L_bok.item()) * B_
            log_ctr += float(L_ctr.item()) * B_
            log_act += float(L_act.item()) * B_
            log_rew_cen += float(win_score.mean().item()) * B_
            log_n += B_

            if step % args.log_every == 0:
                logger.info(
                    "ep%d step %d  T=%.3f  L_bok=%.4f  L_ctr=%.4f  L_act=%.4f  "
                    "win_cen=%.4f  (%.1fs)",
                    epoch, step, T_now,
                    log_bok / log_n, log_ctr / log_n, log_act / log_n,
                    log_rew_cen / log_n, time.time() - t0,
                )
                log_bok = log_ctr = log_act = log_rew_cen = 0.0; log_n = 0

            if step % args.val_every == 0:
                v_cen, v_raw = run_val()
                logger.info("val @ %d  greedy_cen=%.4f  greedy_raw=%.4f", step, v_cen, v_raw)
                if v_cen > best_val:
                    best_val = v_cen; bad = 0
                    state = {n: p.detach().cpu() for n, p in av.named_parameters() if p.requires_grad}
                    torch.save({"trainable": state, "step": step,
                                "val_greedy_cen": v_cen, "val_greedy_raw": v_raw,
                                "cfg": vars(args)}, best_path)
                    logger.info("  new best greedy_cen=%.4f  saved %s", v_cen, best_path)
                else:
                    bad += 1
                    logger.info("  no improvement (%d/%d)", bad, args.patience)
                    if args.patience > 0 and bad >= args.patience:
                        logger.info("early stop"); return

    v_cen, v_raw = run_val()
    logger.info("final greedy_cen=%.4f  greedy_raw=%.4f  best_cen=%.4f", v_cen, v_raw, best_val)


if __name__ == "__main__":
    main()
