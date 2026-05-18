#!/usr/bin/env python
"""Centered (anisotropy-corrected) re-evaluation of the NLA verbalizer.

The raw fve_nrm metric collapses to 0.5*(1+cos(h,v)). Two unrelated Qwen
L20 last-token hiddens already have cos ≈ 0.24 (fve_nrm ≈ 0.62) due to a
shared anisotropic mean. The "0.689" best-of-64 we kept hitting was
therefore only ~0.07 above the anisotropy floor.

This script re-evaluates an adapter checkpoint under two metrics in one
pass, and adds a strong non-trained baseline (NN-retrieval from a pool of
real Qwen samples):

    raw_fve(h,v)      = 0.5 * (1 + cos(h, v))
    centered_fve(h,v) = 0.5 * (1 + cos(h - mu, v - mu))

where mu is the per-coordinate mean of a held-out pool of N real Qwen
L20 last-token hiddens.

For each of M target vectors v we report:

  adapter:
    - greedy raw / centered
    - sampled-mean raw / centered
    - best-of-K raw / centered  (sample-wise argmax under each metric)
  retrieval baseline (no training):
    - nearest other real Qwen sample in the pool, raw / centered

Example:
    python scripts/centered_eval.py \
        --targets artifacts/nla/targets_q7b_L20_seq64_30k_seed1.pt \
        --av-ckpt artifacts/nla/ce_seq64_np16/best_av.pt \
        --backbone Qwen/Qwen2.5-7B --layer 20 \
        --num-prefix-tokens 16 --num-inject-slots 1 \
        --num-vectors 32 --samples-per-v 64 --pool-size 2000 \
        --out artifacts/nla/centered_eval_10k.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srt.nla.config import NLAConfig
from srt.nla.verbalizer import ActivationVerbalizer
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s %(name)s %(message)s", level=logging.INFO)
log = logging.getLogger("centered_eval")


def fve_from_cos(c: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + c)


def cos_pairs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """row-wise cos for matching rows of a and b."""
    an = F.normalize(a.float(), dim=-1)
    bn = F.normalize(b.float(), dim=-1)
    return (an * bn).sum(-1)


def _percentile(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.float(), q).item())


def stats(name: str, x: torch.Tensor) -> dict:
    x = x.float().flatten()
    return {
        f"{name}_mean": float(x.mean()),
        f"{name}_p50": _percentile(x, 0.5),
        f"{name}_p95": _percentile(x, 0.95),
    }


@torch.no_grad()
def extract_last_hidden(backbone, gen_ids, gen_attn, layer):
    out = backbone(
        input_ids=gen_ids,
        attention_mask=gen_attn,
        output_hidden_states=True,
        use_cache=False,
    )
    h = out.hidden_states[layer]
    last = gen_attn.sum(-1) - 1
    idx = last.clamp(min=0).long()
    rows = torch.arange(h.size(0), device=h.device)
    return h[rows, idx, :].detach().to(torch.float32)  # keep on device


@torch.no_grad()
def rollout(av, backbone, v_batch, K, T, temperature, layer, eos_id):
    B = v_batch.size(0)
    v_rep = v_batch.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
    gen_ids = av.generate(
        v_rep, max_new_tokens=T,
        do_sample=(temperature > 0),
        temperature=max(temperature, 1e-6), top_p=1.0,
    )
    eos_mask = (gen_ids == eos_id)
    first_eos = torch.where(
        eos_mask.any(-1),
        eos_mask.float().argmax(-1),
        torch.full((gen_ids.size(0),), gen_ids.size(1) - 1, device=gen_ids.device),
    )
    arange = torch.arange(gen_ids.size(1), device=gen_ids.device).unsqueeze(0)
    attn = (arange <= first_eos.unsqueeze(1)).long()
    h_last = extract_last_hidden(backbone, gen_ids, attn, layer)  # (B*K, d)
    return h_last.view(B, K, -1), v_rep.view(B, K, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, type=Path)
    ap.add_argument("--av-ckpt", required=True, type=Path)
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--num-vectors", type=int, default=32)
    ap.add_argument("--samples-per-v", type=int, default=64)
    ap.add_argument("--rollout-len", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--num-prefix-tokens", type=int, default=16)
    ap.add_argument("--prefix-mode", choices=["static", "mlp"], default="static")
    ap.add_argument("--prefix-mlp-hidden", type=int, default=256)
    ap.add_argument("--num-inject-slots", type=int, default=1)
    ap.add_argument("--batch-vectors", type=int, default=2)
    ap.add_argument("--pool-size", type=int, default=2000,
                    help="number of real Qwen samples used for anisotropy mean + NN retrieval pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"

    log.info("loading backbone %s", args.backbone)
    tok = AutoTokenizer.from_pretrained(args.backbone)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.backbone, torch_dtype=torch.bfloat16
    ).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id or eos_id
    if backbone.config.pad_token_id is None:
        backbone.config.pad_token_id = pad_id

    cfg = NLAConfig(
        backbone_id=args.backbone,
        num_prefix_tokens=args.num_prefix_tokens,
        extraction_layer=args.layer,
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
                sd = sd[k]
                break
    missing, unexpected = av.load_state_dict(sd, strict=False)
    log.info("warm-start: missing=%d unexpected=%d", len(missing), len(unexpected))

    log.info("loading targets %s", args.targets)
    tgt = torch.load(args.targets, map_location="cpu", weights_only=False)
    acts = tgt["activations"]
    pool_all = torch.stack([a[-1].float() for a in acts], 0)  # (N_all, d)
    N_all = pool_all.size(0)
    log.info("pool_all=%s", tuple(pool_all.shape))

    M = args.num_vectors
    P = min(args.pool_size, N_all)
    # use first M as target set, next P as anisotropy-mean / NN-retrieval pool
    V = pool_all[:M].to(device)                                  # (M, d)
    pool = pool_all[M:M + P].to(device)                          # (P, d)
    log.info("M=%d targets   pool=%d for anisotropy mean & NN", M, pool.size(0))

    mu = pool.mean(dim=0, keepdim=True)                          # (1, d)
    mu_norm = mu.norm().item()
    log.info("anisotropy mean ||mu||=%.4f", mu_norm)

    # ---------------- ADAPTER rollouts ----------------
    K = args.samples_per_v
    T = args.rollout_len
    B = args.batch_vectors

    log.info("adapter rollouts M=%d K=%d T=%d", M, K, T)
    h_sampled = []      # (M, K, d)
    h_greedy = []       # (M, 1, d)
    for i in range(0, M, B):
        v = V[i:i + B]
        g_h, _ = rollout(av, backbone, v, 1, T, 0.0, args.layer, eos_id)
        s_h, _ = rollout(av, backbone, v, K, T, args.temperature, args.layer, eos_id)
        h_greedy.append(g_h)
        h_sampled.append(s_h)
        log.info("  vec %d-%d done", i, i + v.size(0))
    h_greedy = torch.cat(h_greedy, 0)        # (M, 1, d)
    h_sampled = torch.cat(h_sampled, 0)      # (M, K, d)

    # broadcast v for scoring
    V_expand_g = V.unsqueeze(1).expand_as(h_greedy)
    V_expand_s = V.unsqueeze(1).expand_as(h_sampled)

    # ---------------- RAW metric ----------------
    def score(h, v):
        return fve_from_cos(cos_pairs(h.reshape(-1, h.size(-1)),
                                      v.reshape(-1, v.size(-1)))).view(h.shape[:-1])

    g_raw = score(h_greedy, V_expand_g).squeeze(1)               # (M,)
    s_raw = score(h_sampled, V_expand_s)                          # (M, K)
    boK_raw = s_raw.max(dim=1).values                             # (M,)
    top5_raw = s_raw.topk(min(5, K), dim=1).values.mean(dim=1)

    # ---------------- CENTERED metric ----------------
    h_greedy_c = h_greedy - mu
    h_sampled_c = h_sampled - mu
    V_c = V - mu
    Ve_g_c = V_c.unsqueeze(1).expand_as(h_greedy_c)
    Ve_s_c = V_c.unsqueeze(1).expand_as(h_sampled_c)
    g_cen = score(h_greedy_c, Ve_g_c).squeeze(1)
    s_cen = score(h_sampled_c, Ve_s_c)
    boK_cen = s_cen.max(dim=1).values
    top5_cen = s_cen.topk(min(5, K), dim=1).values.mean(dim=1)

    # ---------------- NN-RETRIEVAL baseline ----------------
    # for each target v, find max sim to any pool element (raw + centered)
    Vn = F.normalize(V.float(), dim=-1)
    Pn = F.normalize(pool.float(), dim=-1)
    sim_raw = Vn @ Pn.t()                                         # (M, P)
    nn_raw = sim_raw.max(dim=1).values
    nn_raw_fve = fve_from_cos(nn_raw)

    Vcn = F.normalize(V_c, dim=-1)
    Pcn = F.normalize(pool - mu, dim=-1)
    sim_cen = Vcn @ Pcn.t()
    nn_cen = sim_cen.max(dim=1).values
    nn_cen_fve = fve_from_cos(nn_cen)

    # ---------------- RANDOM floor ----------------
    # off-diagonal pairwise within pool
    Psub = Pn[:512]
    off_raw = (Psub @ Psub.t())[~torch.eye(Psub.size(0), dtype=torch.bool, device=device)]
    rand_raw_fve = fve_from_cos(off_raw)
    Pcs = Pcn[:512]
    off_cen = (Pcs @ Pcs.t())[~torch.eye(Pcs.size(0), dtype=torch.bool, device=device)]
    rand_cen_fve = fve_from_cos(off_cen)

    # ---------------- assemble ----------------
    out: dict = {
        "backbone": args.backbone, "layer": args.layer,
        "av_ckpt": str(args.av_ckpt),
        "M": M, "K": K, "T": T, "pool_size": int(pool.size(0)),
        "anisotropy_mu_norm": mu_norm,

        # adapter raw
        "adapter_raw_greedy_mean": float(g_raw.mean()),
        "adapter_raw_sampled_mean": float(s_raw.mean()),
        "adapter_raw_bestofK_mean": float(boK_raw.mean()),
        "adapter_raw_top5ofK_mean": float(top5_raw.mean()),

        # adapter centered
        "adapter_cen_greedy_mean": float(g_cen.mean()),
        "adapter_cen_sampled_mean": float(s_cen.mean()),
        "adapter_cen_bestofK_mean": float(boK_cen.mean()),
        "adapter_cen_top5ofK_mean": float(top5_cen.mean()),

        # NN retrieval baseline
        "nn_raw_mean": float(nn_raw_fve.mean()),
        "nn_raw_p50": _percentile(nn_raw_fve, 0.5),
        "nn_cen_mean": float(nn_cen_fve.mean()),
        "nn_cen_p50": _percentile(nn_cen_fve, 0.5),

        # random floor
        "random_raw_mean": float(rand_raw_fve.mean()),
        "random_cen_mean": float(rand_cen_fve.mean()),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)

    log.info("======== RAW (anisotropy-uncorrected) ========")
    log.info("random_floor   = %.4f", out["random_raw_mean"])
    log.info("adapter greedy = %.4f", out["adapter_raw_greedy_mean"])
    log.info("adapter sampl  = %.4f", out["adapter_raw_sampled_mean"])
    log.info("adapter boK    = %.4f", out["adapter_raw_bestofK_mean"])
    log.info("NN-retrieval   = %.4f", out["nn_raw_mean"])
    log.info("======== CENTERED (anisotropy-corrected) ========")
    log.info("random_floor   = %.4f", out["random_cen_mean"])
    log.info("adapter greedy = %.4f", out["adapter_cen_greedy_mean"])
    log.info("adapter sampl  = %.4f", out["adapter_cen_sampled_mean"])
    log.info("adapter boK    = %.4f", out["adapter_cen_bestofK_mean"])
    log.info("NN-retrieval   = %.4f", out["nn_cen_mean"])
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
