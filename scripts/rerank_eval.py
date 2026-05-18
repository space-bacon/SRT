"""Rerank-decoding evaluation for SRT-NLA verbalizer.

The paper showed adapter best-of-64 saturates the Qwen paraphrase ceiling
(rho_cen ~ 0.99) while greedy stays at ~0.28. This script makes the
*decoding-side* knobs explicit so we can decide whether anything beyond
"sample K, score, pick best" is needed.

For a single AV checkpoint and M targets we:

  1. Sample K_max rollouts per target at temperature T and 1 greedy rollout.
  2. Compute per-rollout features in one batched forward:
        - h_last  -> centered_fve(h_last, v)            (the "oracle" score)
        - mean per-token logp under the same AV          (the "policy" score)
  3. Report three things:

     (a) K-curve : top-1-of-K_oracle_cen for K in {1,2,4,8,16,32,K_max}
                   — the cost / quality trade-off.

     (b) logp-rerank : pick the candidate with highest mean-logp, score it
                       under the oracle. Reranking that needs NO v.
                       If this beats greedy, decoding is fixed for free
                       (modulo K-fold sampling cost).

     (c) NN-anchor-rerank : pick the candidate whose h_last has highest
                            centered-cos to the *nearest retrieval pool*
                            target (not v itself). Sanity baseline.

     Plus per-target Spearman correlation between (logp, oracle_cen)
     to tell us *why* logp-rerank works or doesn't.

Example:
    python scripts/rerank_eval.py \\
        --targets   artifacts/nla/targets_q7b_L20_seq64_30k_seed1.pt \\
        --av-ckpt   artifacts/nla/ce_seq64_np16/best_av.pt \\
        --backbone  Qwen/Qwen2.5-7B --layer 20 \\
        --num-prefix-tokens 16 --num-inject-slots 1 \\
        --num-vectors 200 --K 64 --temperature 1.0 \\
        --pool-size 2000 --rollout-len 64 \\
        --out artifacts/nla/rerank_eval_ce_seq64_np16.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srt.nla.config import NLAConfig
from srt.nla.verbalizer import ActivationVerbalizer
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s %(name)s %(message)s", level=logging.INFO)
log = logging.getLogger("rerank_eval")


def fve_from_cos(c: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + c)


def cos_pairs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    an = F.normalize(a.float(), dim=-1)
    bn = F.normalize(b.float(), dim=-1)
    return (an * bn).sum(-1)


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Per-target Spearman rank correlation, x and y are (K,)."""
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = (rx.norm() * ry.norm()).clamp(min=1e-8)
    return float((rx * ry).sum() / denom)


@torch.no_grad()
def _h_last_and_logp(
    av: ActivationVerbalizer,
    backbone,
    v_rep: torch.Tensor,        # (B, d)
    gen_ids: torch.Tensor,      # (B, T)
    gen_attn: torch.Tensor,     # (B, T)
    layer: int,
    embed_matrix: torch.Tensor,
):
    """Single batched forward over [proj(v), gen_ids[:-1]] → returns
    (h_last (B,d) at layer L on the *last gen token*, mean_logp (B,))."""
    B_, T_ = gen_ids.shape
    prefix = av._inject_prefix(v_rep)                              # (B, P, d) bf16
    P_ = prefix.size(1)
    gen_in = gen_ids[:, :-1]
    tok_emb = F.embedding(gen_in, embed_matrix).to(prefix.dtype)
    full_in = torch.cat([prefix, tok_emb], dim=1)
    prefix_attn = torch.ones(B_, P_, dtype=torch.long, device=prefix.device)
    full_attn   = torch.cat([prefix_attn, gen_attn[:, :-1]], dim=1)
    out = backbone(inputs_embeds=full_in, attention_mask=full_attn,
                   output_hidden_states=True, use_cache=False)
    # h_last at the last valid generated token. The hidden over gen positions
    # lives at indices [P-1 : P-1+T) of full_h (matches the gen_ids slice).
    full_h = out.hidden_states[layer]
    gen_h  = full_h[:, P_ - 1 : P_ - 1 + T_, :]                    # (B, T, d)
    lengths = gen_attn.sum(-1)
    last_idx = (lengths - 1).clamp(min=0).long()
    rows = torch.arange(B_, device=prefix.device)
    h_last = gen_h[rows, last_idx].float()                         # (B, d)
    # mean per-token logp under this AV
    logits = out.logits[:, P_ - 1 : P_ - 1 + T_, :]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    tok_logp = log_probs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)  # (B, T)
    m = gen_attn.float()
    mean_logp = (tok_logp * m).sum(-1) / m.sum(-1).clamp(min=1.0)
    return h_last, mean_logp


@torch.no_grad()
def _rollout(av, backbone, v_rep, T, sample, temperature, eos_id, device):
    gen_ids = av.generate(
        v_rep, max_new_tokens=T,
        do_sample=sample,
        temperature=max(temperature, 1e-6), top_p=1.0,
    )
    B_, Tt = gen_ids.shape
    is_eos = gen_ids == eos_id
    pos = torch.arange(Tt, device=device).unsqueeze(0).expand(B_, Tt)
    first_eos = torch.where(
        is_eos.any(dim=-1, keepdim=True),
        is_eos.float().argmax(dim=-1, keepdim=True).long(),
        torch.full((B_, 1), Tt - 1, device=device, dtype=torch.long),
    )
    gen_attn = (pos <= first_eos).long()
    return gen_ids, gen_attn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets",   required=True, type=Path)
    ap.add_argument("--av-ckpt",   required=True, type=Path)
    ap.add_argument("--backbone",  default="Qwen/Qwen2.5-7B")
    ap.add_argument("--layer",     type=int, default=20)
    ap.add_argument("--num-vectors", type=int, default=200)
    ap.add_argument("--K",           type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--rollout-len", type=int, default=64)
    ap.add_argument("--num-prefix-tokens", type=int, default=16)
    ap.add_argument("--prefix-mode", choices=["static", "mlp"], default="static")
    ap.add_argument("--prefix-mlp-hidden", type=int, default=256)
    ap.add_argument("--num-inject-slots", type=int, default=1)
    ap.add_argument("--batch-vectors", type=int, default=1,
                    help="how many targets to process at once (each spawns K rollouts)")
    ap.add_argument("--score-batch-size", type=int, default=64,
                    help="how many (candidate, prefix) pairs to score per backbone forward")
    ap.add_argument("--pool-size", type=int, default=2000,
                    help="pool slice used for mu (centered metric) and NN-anchor baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out",  required=True, type=Path)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("loading backbone %s", args.backbone)
    tok = AutoTokenizer.from_pretrained(args.backbone)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.backbone, torch_dtype=torch.bfloat16
    ).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    if backbone.config.pad_token_id is None:
        backbone.config.pad_token_id = tok.pad_token_id
    eos_id = int(tok.eos_token_id)

    cfg = NLAConfig(
        backbone_id=args.backbone,
        extraction_layer=args.layer,
        num_prefix_tokens=args.num_prefix_tokens,
        prefix_mode=args.prefix_mode,
        prefix_mlp_hidden=args.prefix_mlp_hidden,
        num_inject_slots=args.num_inject_slots,
    )
    av = ActivationVerbalizer(cfg, backbone, tok).to(device).eval()

    log.info("warm-start from %s", args.av_ckpt)
    sd = torch.load(args.av_ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("trainable", "av", "av_state_dict", "model", "state_dict"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]; break
    missing, unexpected = av.load_state_dict(sd, strict=False)
    log.info("warm-start: missing=%d unexpected=%d", len(missing), len(unexpected))

    log.info("loading targets %s", args.targets)
    obj = torch.load(args.targets, map_location="cpu", weights_only=False)
    pool_all = torch.stack([a[-1].float() for a in obj["activations"]], 0)
    N_all = pool_all.size(0)
    M = min(args.num_vectors, N_all)
    P = min(args.pool_size, N_all - M)
    V_cpu = pool_all[:M]
    pool_cpu = pool_all[M:M + P]
    V = V_cpu.to(device)
    pool = pool_cpu.to(device)
    mu = pool.mean(dim=0, keepdim=True)                              # (1, d)
    mu_norm = float(mu.norm())
    log.info("M=%d P=%d  anisotropy ||mu||=%.4f", M, P, mu_norm)

    embed_matrix = backbone.get_input_embeddings().weight

    K = args.K
    T = args.rollout_len
    K_grid = sorted({k for k in [1, 2, 4, 8, 16, 32, K] if 1 <= k <= K})
    log.info("rollouts: M=%d K=%d T=%d  K_grid=%s", M, K, T, K_grid)

    greedy_cen_per = torch.zeros(M)
    greedy_raw_per = torch.zeros(M)
    sampled_cen_all = torch.zeros(M, K)
    sampled_raw_all = torch.zeros(M, K)
    sampled_logp_all = torch.zeros(M, K)
    sampled_h_all   = torch.zeros(M, K, V.size(1))                   # for NN-anchor rerank

    # ---------------- main loop: one target at a time (K rollouts batched) ----------------
    B = args.batch_vectors
    SB = args.score_batch_size
    t0 = time.time()
    for i in range(0, M, B):
        v = V[i:i + B]
        bs = v.size(0)
        # 1) greedy rollout (bs targets, 1 rollout each)
        g_ids, g_attn = _rollout(av, backbone, v, T, sample=False,
                                 temperature=1.0, eos_id=eos_id, device=device)
        g_h, _g_logp = _h_last_and_logp(av, backbone, v, g_ids, g_attn,
                                        args.layer, embed_matrix)
        g_cen = fve_from_cos(cos_pairs(g_h - mu, v - mu))
        g_raw = fve_from_cos(cos_pairs(g_h,      v))
        greedy_cen_per[i:i + bs] = g_cen.cpu()
        greedy_raw_per[i:i + bs] = g_raw.cpu()

        # 2) K sampled rollouts per target — reshape to (bs*K, ...)
        v_rep = v.unsqueeze(1).expand(bs, K, -1).reshape(bs * K, -1)
        s_ids, s_attn = _rollout(av, backbone, v_rep, T, sample=True,
                                 temperature=args.temperature,
                                 eos_id=eos_id, device=device)
        # 3) score K rollouts in score-batches to bound memory
        h_chunks = []; lp_chunks = []
        for j in range(0, s_ids.size(0), SB):
            jh, jl = _h_last_and_logp(
                av, backbone, v_rep[j:j + SB],
                s_ids[j:j + SB], s_attn[j:j + SB],
                args.layer, embed_matrix,
            )
            h_chunks.append(jh); lp_chunks.append(jl)
        h_s = torch.cat(h_chunks, 0)                                 # (bs*K, d)
        lp  = torch.cat(lp_chunks, 0)                                # (bs*K,)
        # centered & raw scores
        sc_cen = fve_from_cos(cos_pairs(h_s - mu, v_rep - mu))
        sc_raw = fve_from_cos(cos_pairs(h_s,      v_rep))
        sampled_cen_all[i:i + bs] = sc_cen.view(bs, K).cpu()
        sampled_raw_all[i:i + bs] = sc_raw.view(bs, K).cpu()
        sampled_logp_all[i:i + bs] = lp.view(bs, K).cpu()
        sampled_h_all[i:i + bs]   = h_s.view(bs, K, -1).cpu()

        if (i // B) % 5 == 0:
            log.info("vec %d/%d  (%.1fs elapsed)", i + bs, M, time.time() - t0)

    # ────────────────── analyses ──────────────────
    # K-curve (oracle): top-1 of first K' samples for each K' in grid
    k_curve_cen: dict[str, float] = {}
    k_curve_raw: dict[str, float] = {}
    for Kp in K_grid:
        # to be invariant to ordering, take top1 over a *random* subset of K'
        # — but since the K rollouts are exchangeable we can just use the
        # first K' columns (sampled iid).
        top1_cen = sampled_cen_all[:, :Kp].max(dim=1).values
        top1_raw = sampled_raw_all[:, :Kp].max(dim=1).values
        k_curve_cen[str(Kp)] = float(top1_cen.mean())
        k_curve_raw[str(Kp)] = float(top1_raw.mean())

    # logp-rerank: pick argmax_k mean_logp, report its oracle score
    logp_argmax = sampled_logp_all.argmax(dim=1)                    # (M,)
    rows = torch.arange(M)
    logp_pick_cen = sampled_cen_all[rows, logp_argmax]
    logp_pick_raw = sampled_raw_all[rows, logp_argmax]

    # NN-anchor rerank: for each target, find nearest pool target (excluding v),
    # then pick rollout whose h_last is closest (centered cos) to that anchor.
    Vn_c = F.normalize(V - mu, dim=-1)
    Pn_c = F.normalize(pool - mu, dim=-1)
    sim_anchor = Vn_c @ Pn_c.t()                                     # (M, P)
    nn_idx = sim_anchor.argmax(dim=1)                                # (M,)
    nn_target = pool[nn_idx]                                         # (M, d)
    nn_anchor_cen = fve_from_cos(cos_pairs(nn_target, V))            # baseline "what NN-retrieval scores" cen
    # rerank: pick rollout closest to nn_target in centered cos
    nn_a_cen = nn_target - mu                                         # (M, d)
    # for each m, score each of K candidates by centered cos to nn_a_cen[m]
    h_centered = sampled_h_all - mu.cpu()                            # (M, K, d) cpu
    h_n = F.normalize(h_centered, dim=-1)
    a_n = F.normalize(nn_a_cen.cpu(), dim=-1).unsqueeze(1)            # (M, 1, d)
    cos_anchor = (h_n * a_n).sum(-1)                                 # (M, K)
    anchor_pick = cos_anchor.argmax(dim=1)
    anchor_pick_cen = sampled_cen_all[rows, anchor_pick]

    # per-target Spearman(logp, oracle_cen) -- tells us "does logp rank candidates"
    spear = torch.tensor([
        _spearman(sampled_logp_all[m], sampled_cen_all[m]) for m in range(M)
    ])

    out: dict = {
        "backbone": args.backbone, "layer": args.layer,
        "av_ckpt": str(args.av_ckpt),
        "M": M, "K": K, "T": T, "temperature": args.temperature,
        "pool_size": int(pool.size(0)), "anisotropy_mu_norm": mu_norm,

        # baselines
        "greedy_cen_mean": float(greedy_cen_per.mean()),
        "greedy_raw_mean": float(greedy_raw_per.mean()),
        "sampled_cen_mean": float(sampled_cen_all.mean()),
        "sampled_raw_mean": float(sampled_raw_all.mean()),

        # K-curve (oracle rerank)
        "kcurve_oracle_top1_cen": k_curve_cen,
        "kcurve_oracle_top1_raw": k_curve_raw,

        # logp rerank (cheap, NO v required at rerank time)
        "logp_rerank_cen_mean": float(logp_pick_cen.mean()),
        "logp_rerank_raw_mean": float(logp_pick_raw.mean()),
        "logp_rerank_vs_greedy": float(logp_pick_cen.mean() - greedy_cen_per.mean()),

        # NN-anchor rerank (uses pool but NOT v)
        "nn_anchor_rerank_cen_mean": float(anchor_pick_cen.mean()),
        "nn_anchor_baseline_cen_mean": float(nn_anchor_cen.mean()),

        # Spearman(logp, oracle_cen) per target (does logp rank candidates?)
        "spearman_logp_vs_oracle_cen_mean": float(spear.mean()),
        "spearman_logp_vs_oracle_cen_p50":  float(spear.median()),
        "spearman_logp_vs_oracle_cen_p05":  float(torch.quantile(spear, 0.05)),
        "spearman_logp_vs_oracle_cen_p95":  float(torch.quantile(spear, 0.95)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)

    log.info("======== Anchors ========")
    log.info("anisotropy ||mu||      = %.3f", mu_norm)
    log.info("greedy_cen             = %.4f", out["greedy_cen_mean"])
    log.info("sampled_cen (T=%.2f)    = %.4f", args.temperature, out["sampled_cen_mean"])
    log.info("======== K-curve (oracle top-1 of K, centered) ========")
    for Kp in K_grid:
        log.info("  K=%4d   top1_cen = %.4f", Kp, k_curve_cen[str(Kp)])
    log.info("======== Cheap reranks ========")
    log.info("logp_rerank_cen       = %.4f   (delta vs greedy = %+.4f)",
             out["logp_rerank_cen_mean"], out["logp_rerank_vs_greedy"])
    log.info("nn_anchor_rerank_cen  = %.4f", out["nn_anchor_rerank_cen_mean"])
    log.info("nn_anchor_baseline    = %.4f", out["nn_anchor_baseline_cen_mean"])
    log.info("======== Why ========")
    log.info("Spearman(logp, oracle_cen)  mean=%.3f  p50=%.3f  p05=%.3f  p95=%.3f",
             out["spearman_logp_vs_oracle_cen_mean"],
             out["spearman_logp_vs_oracle_cen_p50"],
             out["spearman_logp_vs_oracle_cen_p05"],
             out["spearman_logp_vs_oracle_cen_p95"])
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
