"""SRT-NLA — Best-of-K self-distillation v2 (Lever B).

Rewrite of ``scripts/train_nla_bok.py`` informed by the May-17 K-curve
diagnosis and the Probe-I geometric findings:

    1. K=4 was undermatched. The K-curve is +0.10 ρ_norm per doubling;
       K=4 lands the teacher signal inside sampling noise of greedy.
       v2 defaults to **K=32** (handoff: "K=16 minimum, K=32 preferred").

    2. Logp has zero ranking power over centered-fve at the per-target
       level (Spearman 0.04 in rerank_eval). v2 therefore **never uses
       logp as a target** — neither as a reranker, value-head, nor self-
       distillation signal. The only logp use is the policy CE on the
       BoK winner's tokens (which is just standard distillation).

    3. The interpretability work flagged template-collapse / OOD
       attractor risk on the AV decoder. v2 logs **per-batch winner
       n-gram diversity** and at every val step writes the chosen-
       rollout text for the val-vectors so collapse is detectable
       qualitatively from the artifact log.

    4. Val tracks both greedy_cen AND oracle_cen at K=32 every
       ``--val-every`` steps. The headline diagnostic is the SHRINKING
       GAP between them as the policy improves.

    5. Memory budget knob: ``--profile`` runs 5 steps and prints peak
       GPU memory + step-time so the user can size the actual run on
       a fresh A6000 / Blackwell RTX 6000 instance.

Default loss:
    L = 1.0 * L_bok + 0.3 * L_ctr (+ 0.0 * L_act)

Default schedule:
    K=32, batch=8, rollout_len=64, temperature 1.5 → 0.7 cosine.

Smoke (intended on Blackwell 96GB):
    python scripts/train_nla_bok_v2.py \\
        --targets    artifacts/nla/targets_q7b_L20_seq64_30k_seed1.pt \\
        --pairs      artifacts/nla/curated/gold_pairs_seq64.jsonl \\
        --init-from  artifacts/nla/ce_seq64_np16/best_av.pt \\
        --num-prefix-tokens 16 --num-inject-slots 1 \\
        --epochs 1 --batch-size 8 --samples-per-v 32 --hard-negs 8 \\
        --temp-start 1.5 --temp-end 0.7 \\
        --alpha-bok 1.0 --beta-ctr 0.3 --gamma-act 0.0 \\
        --val-every 500 --val-vectors 200 --val-K 32 --pool-size 2000 \\
        --profile \\
        --out artifacts/nla/bok_v2_seq64_np16
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from srt.nla import ActivationVerbalizer, NLAConfig
from srt.nla.decoding import (
    _cos_pairs,
    _fve_from_cos,
    _h_last_prefix_free,
    _rollout as _decode_rollout,
    oracle_rerank_decode,
    rho_norm,
)

logger = logging.getLogger("train_nla_bok_v2")

_DTYPE = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


# ────────────────────────────── args ──────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # data / init
    p.add_argument("--targets",    required=True, type=Path)
    p.add_argument("--pairs",      required=True, type=Path)
    p.add_argument("--init-from",  type=Path, default=None)
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
    p.add_argument("--epochs",       type=int, default=1)
    p.add_argument("--batch-size",   type=int, default=8)
    p.add_argument("--lr",           type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--lr-schedule",  choices=["constant", "cosine"], default="cosine")
    p.add_argument("--warmup-steps", type=int, default=200)
    # rollout — v2 defaults raised per handoff
    p.add_argument("--samples-per-v", type=int, default=32,
                   help="K rollouts per target. K=32 preferred (K=16 minimum).")
    p.add_argument("--rollout-len",   type=int, default=64)
    p.add_argument("--temp-start",    type=float, default=1.5)
    p.add_argument("--temp-end",      type=float, default=0.7)
    p.add_argument("--score-batch-size", type=int, default=128,
                   help="cap on (B*K) chunked through the scoring forward")
    # losses
    p.add_argument("--alpha-bok", type=float, default=1.0)
    p.add_argument("--beta-ctr",  type=float, default=0.3)
    p.add_argument("--gamma-act", type=float, default=0.0)
    p.add_argument("--hard-negs", type=int, default=8)
    p.add_argument("--neg-pool-cap", type=int, default=8192)
    # eval
    p.add_argument("--pool-size",    type=int, default=2000)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--val-every",    type=int, default=500)
    p.add_argument("--val-vectors",  type=int, default=200)
    p.add_argument("--val-K",        type=int, default=32,
                   help="K used for the val-time oracle-rerank diagnostic")
    p.add_argument("--save-val-text", action="store_true", default=True,
                   help="dump greedy + oracle text at every val step (default on)")
    p.add_argument("--log-every",    type=int, default=10)
    p.add_argument("--patience",     type=int, default=10)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--out",          required=True, type=Path)
    p.add_argument("--profile", action="store_true",
                   help="run 5 steps and report peak GPU mem + step time, then exit")
    return p.parse_args()


# ────────────────────────────── helpers ──────────────────────────────

def _load_targets(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return torch.stack([a[-1] for a in obj["activations"]]).float()


def _load_pairs(path: Path) -> list[dict]:
    recs: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _last_h(hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    lengths = attn.sum(dim=-1)
    last_idx = (lengths - 1).clamp(min=0).long()
    rows = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[rows, last_idx]


def _pad_to(lst: list[list[int]], pad_id: int, max_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if max_len is None:
        max_len = max(len(x) for x in lst)
    out = torch.full((len(lst), max_len), pad_id, dtype=torch.long)
    msk = torch.zeros((len(lst), max_len), dtype=torch.long)
    for i, x in enumerate(lst):
        x = x[:max_len]
        out[i, : len(x)] = torch.tensor(x, dtype=torch.long)
        msk[i, : len(x)] = 1
    return out, msk


def build_hardneg_index(
    pool_cen: torch.Tensor,
    pair_target_idx: torch.Tensor,
    J: int,
    cap: int,
    device: torch.device,
) -> torch.Tensor:
    P = pair_target_idx.numel()
    cand_cap = min(cap, P)
    cand = pair_target_idx[:cand_cap]
    Vc = pool_cen[pair_target_idx]
    Cc = pool_cen[cand]
    Vn = F.normalize(Vc.to(device).float(), dim=-1)
    Cn = F.normalize(Cc.to(device).float(), dim=-1)
    out = torch.empty((P, J), dtype=torch.long)
    chunk = 256
    for i in range(0, P, chunk):
        sim = Vn[i : i + chunk] @ Cn.t()
        b = sim.size(0)
        for k in range(b):
            gi = i + k
            if gi < cand_cap:
                sim[k, gi] = -1e9
        out[i : i + chunk] = sim.topk(J, dim=1).indices.cpu()
    return out


def _ngram_dup_rate(ids: torch.Tensor, attn: torch.Tensor, n: int = 5) -> float:
    """Fraction of n-grams across the batch that are duplicates of another row.

    Cheap template-collapse detector. If the BoK winners all share a 5-gram,
    the policy is collapsing onto a canonical attractor (Probe I motif).
    """
    counts: Counter = Counter()
    rows = ids.tolist()
    msk = attn.tolist()
    total = 0
    for r, m in zip(rows, msk):
        L = sum(m)
        for i in range(0, max(0, L - n + 1)):
            counts[tuple(r[i : i + n])] += 1
            total += 1
    if total == 0:
        return 0.0
    dup = sum(c - 1 for c in counts.values() if c > 1)
    return dup / total


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
    logger.info("AV trainable=%s  K=%d  batch=%d  T:%.2f→%.2f",
                f"{n_trainable:,}", args.samples_per_v, args.batch_size,
                args.temp_start, args.temp_end)
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

    # ─── data ───
    pool_cpu = _load_targets(args.targets)
    pairs = _load_pairs(args.pairs)
    logger.info("loaded %d pairs; target pool N=%d", len(pairs), pool_cpu.size(0))

    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(pool_cpu.size(0), generator=rng)
    mu_idx = perm[: args.pool_size]
    mu = pool_cpu[mu_idx].mean(dim=0).to(device)
    pool_cen_cpu = pool_cpu - mu.cpu()
    logger.info("anisotropy ||mu||=%.4f  (pool_size=%d)", float(mu.norm()), args.pool_size)

    perm2 = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(args.seed + 1))
    n_val = max(1, int(len(pairs) * args.val_fraction))
    val_idx = perm2[:n_val].tolist()
    train_idx = perm2[n_val:].tolist()
    val_pairs = [pairs[i] for i in val_idx[: args.val_vectors]]
    logger.info("train=%d  val=%d", len(train_idx), len(val_pairs))

    pair_tidx = torch.tensor([pairs[i]["target_idx"] for i in train_idx], dtype=torch.long)
    logger.info("building NN hard-neg index J=%d cap=%d ...", args.hard_negs, args.neg_pool_cap)
    hn = build_hardneg_index(pool_cen_cpu, pair_tidx, args.hard_negs, args.neg_pool_cap, device)
    logger.info("hard-neg index: %s", tuple(hn.shape))

    embed_matrix = backbone.get_input_embeddings().weight

    # ────────────────── core forward (with grad on prefix only) ──────────────────
    def _seq_logp_and_h(v_rep, gen_ids, gen_attn, want_hidden):
        B_, T_ = gen_ids.shape
        prefix = av._inject_prefix(v_rep)
        P_ = prefix.size(1)
        gen_in = gen_ids[:, :-1]
        tok_emb = F.embedding(gen_in, embed_matrix).to(prefix.dtype).detach()
        full_in = torch.cat([prefix, tok_emb], dim=1)
        prefix_attn = torch.ones(B_, P_, dtype=torch.long, device=device)
        full_attn = torch.cat([prefix_attn, gen_attn[:, :-1]], dim=1)
        out = backbone(inputs_embeds=full_in, attention_mask=full_attn,
                       output_hidden_states=want_hidden, use_cache=False)
        logits = out.logits[:, P_ - 1 : P_ - 1 + T_, :]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        nll = -log_probs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)
        h_last = None
        if want_hidden:
            full_h = out.hidden_states[args.layer]
            gen_h = full_h[:, P_ - 1 : P_ - 1 + T_, :]
            h_last = _last_h(gen_h, gen_attn)
        return nll, h_last

    @torch.no_grad()
    def _sample_and_score(v_rep, sample, temp):
        gen_ids, gen_attn = _decode_rollout(
            av, v_rep,
            max_new_tokens=args.rollout_len,
            do_sample=sample,
            temperature=temp,
            eos_id=eos_id,
        )
        h = _h_last_prefix_free(backbone, gen_ids, gen_attn, args.layer)
        sc_cen = _fve_from_cos(_cos_pairs(h - mu, v_rep.float() - mu))
        return gen_ids, gen_attn, sc_cen, h

    # ────────────────── val pass: greedy + oracle K-curve diagnostic ──────────────────
    @torch.no_grad()
    def run_val(step: int) -> dict:
        av.eval()
        g_cen_sum = 0.0
        o_cen_sum = 0.0
        n = 0
        text_records: list[dict] = []
        for i in range(0, len(val_pairs), args.batch_size):
            batch = val_pairs[i : i + args.batch_size]
            v = pool_cpu[[r["target_idx"] for r in batch]].to(device)
            # greedy
            g_ids, g_attn = _decode_rollout(
                av, v,
                max_new_tokens=args.rollout_len,
                do_sample=False, temperature=1.0, eos_id=eos_id,
            )
            g_h = _h_last_prefix_free(backbone, g_ids, g_attn, args.layer)
            g_cen = _fve_from_cos(_cos_pairs(g_h - mu, v - mu))
            g_cen_sum += float(g_cen.sum().item())
            # oracle rerank
            res = oracle_rerank_decode(
                av, backbone, v,
                layer=args.layer, K=args.val_K,
                max_new_tokens=args.rollout_len,
                temperature=1.0, mu=mu,
                score_batch_size=args.score_batch_size,
                eos_id=eos_id, return_all=False,
            )
            o_cen_sum += float(res.best_cen.sum().item())
            n += v.size(0)

            if args.save_val_text:
                g_text = tok.batch_decode(g_ids, skip_special_tokens=True)
                o_text = tok.batch_decode(res.best_ids, skip_special_tokens=True)
                for b in range(v.size(0)):
                    text_records.append({
                        "step": step, "i": i + b,
                        "greedy_cen": float(g_cen[b].cpu()),
                        "greedy_text": g_text[b],
                        "oracle_cen": float(res.best_cen[b].cpu()),
                        "oracle_chosen_k": int(res.best_idx[b].cpu()),
                        "oracle_text": o_text[b],
                    })
        av.train()
        g_mean = g_cen_sum / max(n, 1)
        o_mean = o_cen_sum / max(n, 1)
        if args.save_val_text:
            txt_path = args.out / f"val_text_step{step:06d}.jsonl"
            with txt_path.open("w") as f:
                for rec in text_records:
                    f.write(json.dumps(rec) + "\n")
            logger.info("  wrote %s (%d records)", txt_path, len(text_records))
        return {
            "step": step, "n": n,
            "greedy_cen": g_mean, "oracle_cen": o_mean,
            "greedy_rho_norm": float(rho_norm(g_mean)),
            "oracle_rho_norm": float(rho_norm(o_mean)),
            "gap_cen": o_mean - g_mean,
        }

    # ────────────────── training loop ──────────────────
    K = args.samples_per_v
    J = args.hard_negs
    steps_per_epoch = max(1, (len(train_idx) + args.batch_size - 1) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs

    def temp_at(step: int) -> float:
        if total_steps <= 1:
            return args.temp_end
        f = step / max(1, total_steps - 1)
        return args.temp_start + (args.temp_end - args.temp_start) * 0.5 * (1 - math.cos(math.pi * f))

    sched = None
    if args.lr_schedule == "cosine":
        warmup = max(0, args.warmup_steps)

        def _lr_lambda(s: int) -> float:
            if warmup > 0 and s < warmup:
                return (s + 1) / float(warmup)
            prog = (s - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(max(prog, 0.0), 1.0)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        logger.info("cosine LR: total_steps=%d warmup=%d peak=%g", total_steps, warmup, args.lr)

    best_val = -1.0
    best_path = args.out / "best_av.pt"
    log_path = args.out / "train_log.jsonl"
    log_f = log_path.open("a")
    bad = 0
    step = 0
    t0 = time.time()
    log_bok = log_ctr = log_act = log_rew = log_dup = 0.0
    log_n = 0

    if args.profile:
        logger.info("--- PROFILE MODE: 5 steps then exit ---")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        torch.manual_seed(args.seed + epoch)
        order = torch.randperm(len(train_idx)).tolist()
        for i in range(0, len(order), args.batch_size):
            t_step = time.time()
            batch_local = order[i : i + args.batch_size]
            pair_batch = [pairs[train_idx[j]] for j in batch_local]
            B_ = len(pair_batch)
            v = pool_cpu[[r["target_idx"] for r in pair_batch]].to(device)

            # 1) K rollouts at scheduled temperature, score, pick winner
            T_now = temp_at(step)
            v_rep = v.unsqueeze(1).expand(B_, K, -1).reshape(B_ * K, -1)
            g_ids, g_attn, sc_cen, _h = _sample_and_score(v_rep, sample=True, temp=T_now)
            sc_mat = sc_cen.view(B_, K)
            win = sc_mat.argmax(dim=1)
            row_off = torch.arange(B_, device=device) * K
            win_idx = row_off + win
            win_ids = g_ids[win_idx]
            win_attn = g_attn[win_idx]
            win_score = sc_mat[torch.arange(B_, device=device), win]
            dup_rate = _ngram_dup_rate(win_ids.cpu(), win_attn.cpu(), n=5)

            # 2) BoK distillation: CE on winner tokens
            nll_w, h_w = _seq_logp_and_h(
                v, win_ids, win_attn, want_hidden=(args.gamma_act > 0)
            )
            mask_w = win_attn.float()
            tok_per = mask_w.sum(dim=-1).clamp(min=1.0)
            L_bok = ((nll_w * mask_w).sum(dim=-1) / tok_per).mean()

            # 3) InfoNCE on NN hard-neg gold texts
            #    NOTE: this is NOT logp-rerank — it's contrastive across v's,
            #    asking "does the model assign higher logp to MY gold than to
            #    a near-neighbour v's gold?" Per-target Spearman doesn't
            #    constrain this signal; keep it.
            L_ctr = torch.zeros((), device=device)
            if args.beta_ctr > 0 and J > 0:
                neg_rows: list[list[int]] = []
                for k, bl in enumerate(batch_local):
                    neg_pair_ix = hn[bl].tolist()
                    for npix in neg_pair_ix:
                        neg_rows.append(pairs[train_idx[npix]]["gold_ids"])
                pos_rows = [r["gold_ids"] for r in pair_batch]
                all_rows: list[list[int]] = []
                for k in range(B_):
                    all_rows.append(pos_rows[k])
                    all_rows.extend(neg_rows[k * J : (k + 1) * J])
                max_t = min(max(len(x) for x in all_rows), args.rollout_len)
                ids_cpu, msk_cpu = _pad_to(all_rows, pad_id, max_t)
                ctr_ids = ids_cpu.to(device)
                ctr_attn = msk_cpu.to(device)
                v_ctr = v.unsqueeze(1).expand(B_, 1 + J, -1).reshape(B_ * (1 + J), -1)
                nll_c, _ = _seq_logp_and_h(v_ctr, ctr_ids, ctr_attn, want_hidden=False)
                m = ctr_attn.float()
                logp = -(nll_c * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1.0)
                logits = logp.view(B_, 1 + J)
                tgt = torch.zeros(B_, dtype=torch.long, device=device)
                L_ctr = F.cross_entropy(logits, tgt)

            L_act = torch.zeros((), device=device)
            if args.gamma_act > 0 and h_w is not None:
                L_act = (1.0 - _fve_from_cos(_cos_pairs(h_w.float() - mu, v.float() - mu))).mean()

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
            log_rew += float(win_score.mean().item()) * B_
            log_dup += dup_rate * B_
            log_n += B_

            if step % args.log_every == 0:
                step_dt = time.time() - t_step
                logger.info(
                    "ep%d step %d  T=%.2f  L_bok=%.4f L_ctr=%.4f L_act=%.4f  "
                    "win_cen=%.4f  5gram_dup=%.3f  step_dt=%.2fs  total=%.0fs",
                    epoch, step, T_now,
                    log_bok / log_n, log_ctr / log_n, log_act / log_n,
                    log_rew / log_n, log_dup / log_n,
                    step_dt, time.time() - t0,
                )
                log_f.write(json.dumps({
                    "kind": "train", "step": step, "epoch": epoch,
                    "T": T_now,
                    "L_bok": log_bok / log_n, "L_ctr": log_ctr / log_n,
                    "L_act": log_act / log_n,
                    "win_cen": log_rew / log_n,
                    "ngram5_dup": log_dup / log_n,
                }) + "\n")
                log_f.flush()
                log_bok = log_ctr = log_act = log_rew = log_dup = 0.0
                log_n = 0

            if args.profile and step >= 5:
                if torch.cuda.is_available():
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    logger.info("PROFILE: peak GPU mem = %.2f GB after %d steps", peak, step)
                logger.info("PROFILE: %.2fs per step (avg over %d steps)",
                            (time.time() - t0) / step, step)
                logger.info("PROFILE: exiting (re-run without --profile to train).")
                return

            if step % args.val_every == 0:
                vres = run_val(step)
                logger.info(
                    "val @ %d  greedy_cen=%.4f (rho=%.3f)  oracle_cen=%.4f (rho=%.3f)  "
                    "gap=%.4f",
                    step, vres["greedy_cen"], vres["greedy_rho_norm"],
                    vres["oracle_cen"], vres["oracle_rho_norm"], vres["gap_cen"],
                )
                log_f.write(json.dumps({"kind": "val", **vres}) + "\n")
                log_f.flush()
                if vres["greedy_cen"] > best_val:
                    best_val = vres["greedy_cen"]
                    bad = 0
                    state = {n: p.detach().cpu()
                             for n, p in av.named_parameters() if p.requires_grad}
                    torch.save({
                        "trainable": state, "step": step,
                        "val": vres, "cfg": vars(args),
                    }, best_path)
                    logger.info("  new best greedy_cen=%.4f  saved %s", vres["greedy_cen"], best_path)
                else:
                    bad += 1
                    logger.info("  no improvement (%d/%d)", bad, args.patience)
                    if args.patience > 0 and bad >= args.patience:
                        logger.info("early stop")
                        log_f.close()
                        return

    vres = run_val(step)
    logger.info(
        "FINAL  greedy_cen=%.4f (rho=%.3f)  oracle_cen=%.4f (rho=%.3f)  best_greedy=%.4f",
        vres["greedy_cen"], vres["greedy_rho_norm"],
        vres["oracle_cen"], vres["oracle_rho_norm"], best_val,
    )
    log_f.write(json.dumps({"kind": "final", **vres}) + "\n")
    log_f.close()


if __name__ == "__main__":
    main()
