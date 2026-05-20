"""Deployable oracle-rerank decoding for SRT-NLA — CLI driver.

This script is the production-shaped wrapper around
``srt.nla.decoding.oracle_rerank_decode``. It loads an AV checkpoint,
samples K rollouts per target, scores each rollout with the centered
fve metric, picks the argmax, and reports:

  - greedy_cen_mean    (T=0, K=1)        baseline
  - oracle_cen_mean    (oracle-rerank)   the deployable method
  - greedy_rho_norm, oracle_rho_norm     in paper units
  - per-target Δ (oracle − greedy)       headroom distribution

Per-target chosen text is also written so the decoded outputs can be
inspected after the run.

Example:
    python scripts/oracle_rerank_decode.py \\
        --targets   artifacts/nla/targets_q7b_L20_seq64_30k_seed1.pt \\
        --av-ckpt   artifacts/nla/ce_seq64_np16/best_av.pt \\
        --backbone  Qwen/Qwen2.5-7B --layer 20 \\
        --num-prefix-tokens 16 --num-inject-slots 1 \\
        --num-vectors 200 --K 64 --temperature 1.0 \\
        --rollout-len 64 --pool-size 2000 \\
        --out artifacts/nla/oracle_rerank_decode_seq64_np16.json

Notes
-----
- Uses prefix-FREE re-forward to extract h_last (deployment-realistic;
  matches centered_eval.py and the paper's metric).
- Single-GPU. ~K× the wall-clock of greedy decoding; with K=64 and
  rollout-len=64 this is ~1–2 minutes for 200 targets on an A6000.
- Targets file must be the same `.pt` produced by the training
  pipeline (a dict with key "activations" — list of (T, d) tensors;
  the last token's activation is used as v).
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srt.nla.config import NLAConfig
from srt.nla.decoding import (
    PARAPHRASE_CEILING_CEN,
    RANDOM_FLOOR_CEN,
    _cos_pairs,
    _fve_from_cos,
    _h_last_prefix_free,
    _rollout,
    oracle_rerank_decode,
    rho_norm,
)
from srt.nla.verbalizer import ActivationVerbalizer
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s %(name)s %(message)s", level=logging.INFO)
log = logging.getLogger("oracle_rerank_decode")


def _load_av_state_dict(path: Path) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("trainable", "av", "av_state_dict", "model", "state_dict"):
            if k in sd and isinstance(sd[k], dict):
                return sd[k]
    return sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, type=Path)
    ap.add_argument("--av-ckpt", required=True, type=Path)
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--num-vectors", type=int, default=200)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--rollout-len", type=int, default=64)
    ap.add_argument("--num-prefix-tokens", type=int, default=16)
    ap.add_argument("--prefix-mode", choices=["static", "mlp"], default="static")
    ap.add_argument("--prefix-mlp-hidden", type=int, default=256)
    ap.add_argument("--num-inject-slots", type=int, default=1)
    ap.add_argument("--pool-size", type=int, default=2000)
    ap.add_argument("--score-batch-size", type=int, default=64)
    ap.add_argument("--batch-vectors", type=int, default=1,
                    help="targets processed concurrently (each spawns K rollouts)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--save-text", action="store_true",
                    help="also save chosen-rollout text per target into <out>.text.jsonl")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    log.info("loading backbone %s", args.backbone)
    tok = AutoTokenizer.from_pretrained(args.backbone)
    backbone = (
        AutoModelForCausalLM.from_pretrained(args.backbone, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
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
    sd = _load_av_state_dict(args.av_ckpt)
    missing, unexpected = av.load_state_dict(sd, strict=False)
    log.info("warm-start: missing=%d unexpected=%d", len(missing), len(unexpected))

    log.info("loading targets %s", args.targets)
    obj = torch.load(args.targets, map_location="cpu", weights_only=False)
    pool_all = torch.stack([a[-1].float() for a in obj["activations"]], 0)
    N_all = pool_all.size(0)
    M = min(args.num_vectors, N_all)
    P = min(args.pool_size, N_all - M)
    V_cpu = pool_all[:M]
    pool_cpu = pool_all[M : M + P]
    V = V_cpu.to(device)
    pool = pool_cpu.to(device)
    mu = pool.mean(dim=0, keepdim=True)
    log.info(
        "M=%d P=%d  anisotropy ||mu||=%.4f  K=%d  T=%d  temp=%.2f",
        M, P, float(mu.norm()), args.K, args.rollout_len, args.temperature,
    )

    greedy_cen = torch.zeros(M)
    oracle_cen = torch.zeros(M)
    chosen_idx = torch.zeros(M, dtype=torch.long)
    text_records: list[dict] = []

    t0 = time.time()
    B = args.batch_vectors
    for i in range(0, M, B):
        v = V[i : i + B]
        bs = v.size(0)

        # --- greedy baseline (T=0, K=1) ---
        g_ids, g_attn = _rollout(
            av, v,
            max_new_tokens=args.rollout_len,
            do_sample=False,
            temperature=1.0,
            eos_id=eos_id,
        )
        g_h = _h_last_prefix_free(backbone, g_ids, g_attn, args.layer)
        g_cen = _fve_from_cos(_cos_pairs(g_h - mu, v - mu))
        greedy_cen[i : i + bs] = g_cen.cpu()

        # --- oracle rerank (the deployable method) ---
        res = oracle_rerank_decode(
            av, backbone, v,
            layer=args.layer,
            K=args.K,
            max_new_tokens=args.rollout_len,
            temperature=args.temperature,
            mu=mu,
            score_batch_size=args.score_batch_size,
            eos_id=eos_id,
            return_all=False,
        )
        oracle_cen[i : i + bs] = res.best_cen.cpu()
        chosen_idx[i : i + bs] = res.best_idx.cpu()

        if args.save_text:
            g_text = tok.batch_decode(g_ids, skip_special_tokens=True)
            o_text = tok.batch_decode(res.best_ids, skip_special_tokens=True)
            for b in range(bs):
                text_records.append({
                    "i": i + b,
                    "greedy_text": g_text[b],
                    "greedy_cen": float(g_cen[b].cpu()),
                    "oracle_text": o_text[b],
                    "oracle_cen": float(res.best_cen[b].cpu()),
                    "oracle_chosen_k": int(res.best_idx[b].cpu()),
                })

        if (i // B) % 10 == 0:
            log.info("  %d/%d  (%.1fs)", i + bs, M, time.time() - t0)

    delta = oracle_cen - greedy_cen
    out = {
        "backbone": args.backbone,
        "layer": args.layer,
        "av_ckpt": str(args.av_ckpt),
        "M": M, "K": args.K, "T": args.rollout_len,
        "temperature": args.temperature,
        "pool_size": int(pool.size(0)),
        "anisotropy_mu_norm": float(mu.norm()),
        "anchors": {
            "random_floor_cen": RANDOM_FLOOR_CEN,
            "paraphrase_ceiling_cen": PARAPHRASE_CEILING_CEN,
        },
        # core numbers
        "greedy_cen_mean": float(greedy_cen.mean()),
        "oracle_cen_mean": float(oracle_cen.mean()),
        "greedy_rho_norm": float(rho_norm(greedy_cen.mean())),
        "oracle_rho_norm": float(rho_norm(oracle_cen.mean())),
        # per-target headroom distribution
        "delta_mean": float(delta.mean()),
        "delta_p50":  float(delta.median()),
        "delta_p05":  float(torch.quantile(delta, 0.05)),
        "delta_p95":  float(torch.quantile(delta, 0.95)),
        "delta_min":  float(delta.min()),
        "delta_max":  float(delta.max()),
        # diagnostic: which K-position got picked
        "chosen_idx_mean": float(chosen_idx.float().mean()),
        "chosen_idx_p50":  float(chosen_idx.float().median()),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)
    log.info("wrote %s", args.out)

    if args.save_text:
        text_path = args.out.with_suffix(args.out.suffix + ".text.jsonl")
        with text_path.open("w") as f:
            for rec in text_records:
                f.write(json.dumps(rec) + "\n")
        log.info("wrote %s", text_path)

    log.info("======== Anchors ========")
    log.info("random_floor_cen      = %.4f", RANDOM_FLOOR_CEN)
    log.info("paraphrase_ceiling_cen= %.4f", PARAPHRASE_CEILING_CEN)
    log.info("======== Core results ========")
    log.info("greedy_cen            = %.4f   (rho_norm = %.3f)",
             out["greedy_cen_mean"], out["greedy_rho_norm"])
    log.info("oracle_cen            = %.4f   (rho_norm = %.3f)",
             out["oracle_cen_mean"], out["oracle_rho_norm"])
    log.info("======== Headroom (oracle - greedy) per target ========")
    log.info("  mean=%.4f  p50=%.4f  p05=%.4f  p95=%.4f  min=%.4f  max=%.4f",
             out["delta_mean"], out["delta_p50"], out["delta_p05"],
             out["delta_p95"], out["delta_min"], out["delta_max"])


if __name__ == "__main__":
    main()
