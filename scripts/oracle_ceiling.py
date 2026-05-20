#!/usr/bin/env python
"""Oracle / ceiling / floor baselines for the NLA verbalization metric.

The single-vector verbalization metric used throughout the NLA pipeline is

    fve_nrm(h, v) = 1 - 1/2 * || normalize(h) - normalize(v) ||^2
                  = 1/2 * (1 + cos(h, v))

so fve_nrm is just a linear remap of cosine. The best-of-64 probe reports
≈ 0.689 for our trained adapter, but we have no reference frame: we never
measured what fve_nrm looks like for

    (a) the *true* source text replayed through the backbone   (oracle, ~1.0)
    (b) two unrelated Qwen-sampled continuations                (random floor)
    (c) the nearest *other* real Qwen sample in the pool        (NN-in-pool)
    (d) explicit paraphrases of the source via Qwen             (semantic ceiling)

This script computes all four on a held-out subset of the existing
`sample_targets.py` .pt shard, so the headline result can be expressed as

    rho = (adapter - random_floor) / (paraphrase_ceiling - random_floor)

without spending another training cycle.

Example:
    python scripts/oracle_ceiling.py \
        --targets artifacts/nla/targets_q7b_L20_seq64_30k_seed1.pt \
        --backbone Qwen/Qwen2.5-7B --layer 20 \
        --n-targets 200 --paraphrase-k 8 --paraphrase-max-new 64 \
        --out artifacts/nla/oracle_ceiling_30k.json
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

from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s %(name)s %(message)s", level=logging.INFO)
log = logging.getLogger("oracle_ceiling")


def fve_nrm_from_cos(c: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + c)


def _pct(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.float(), q).item())


def summary_stats(name: str, x: torch.Tensor) -> dict:
    x = x.float().flatten()
    fve = fve_nrm_from_cos(x)
    return {
        f"{name}_cos_mean": float(x.mean()),
        f"{name}_cos_p05": _pct(x, 0.05),
        f"{name}_cos_p50": _pct(x, 0.50),
        f"{name}_cos_p95": _pct(x, 0.95),
        f"{name}_fve_mean": float(fve.mean()),
        f"{name}_fve_p50": _pct(fve, 0.50),
        f"{name}_n": int(x.numel()),
    }


def cos_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise cosine for matching rows of a, b."""
    an = F.normalize(a.float(), dim=-1)
    bn = F.normalize(b.float(), dim=-1)
    return (an * bn).sum(-1)


@torch.no_grad()
def encode_last_hidden(
    backbone, tok, texts: list[str], layer: int, device: str, max_len: int = 128
) -> torch.Tensor:
    """Tokenize each text and return the last-valid-token hidden at `layer` (fp32 cpu)."""
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id or eos_id
    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    # ensure non-empty rows
    valid_len = attn.sum(-1)
    if (valid_len == 0).any():
        # replace empties with a single eos to avoid a zero-length forward
        bad = valid_len == 0
        input_ids[bad, 0] = eos_id
        attn[bad, 0] = 1
        valid_len = attn.sum(-1)
    out = backbone(
        input_ids=input_ids,
        attention_mask=attn,
        output_hidden_states=True,
        use_cache=False,
    )
    h = out.hidden_states[layer]  # (B, T, d)
    idx = (valid_len - 1).clamp(min=0).long()
    rows = torch.arange(h.size(0), device=h.device)
    return h[rows, idx, :].detach().to(torch.float32).cpu()


@torch.no_grad()
def sample_paraphrases(
    backbone,
    tok,
    sources: list[str],
    k: int,
    max_new: int,
    temperature: float,
    device: str,
) -> list[list[str]]:
    """For each source, sample k Qwen paraphrases via a simple instruction prompt."""
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id or eos_id
    prompts = [
        f"Paraphrase the following text using different words but the same meaning.\n\nText: {s}\n\nParaphrase:"
        for s in sources
    ]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    B = input_ids.size(0)
    # replicate K times
    input_ids = input_ids.unsqueeze(1).expand(B, k, -1).reshape(B * k, -1)
    attn = attn.unsqueeze(1).expand(B, k, -1).reshape(B * k, -1)
    out = backbone.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
    )
    gen = out[:, input_ids.size(1):]  # only new tokens
    texts = tok.batch_decode(gen, skip_special_tokens=True)
    # group back into per-source lists
    return [texts[i * k:(i + 1) * k] for i in range(B)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, type=Path)
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n-targets", type=int, default=200,
                    help="how many of the saved targets to evaluate")
    ap.add_argument("--encode-batch", type=int, default=16)
    ap.add_argument("--paraphrase-k", type=int, default=8,
                    help="paraphrase samples per source; 0 = skip semantic ceiling")
    ap.add_argument("--paraphrase-max-new", type=int, default=64)
    ap.add_argument("--paraphrase-batch", type=int, default=4,
                    help="number of sources processed per paraphrase generation call")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-encode-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("loading backbone %s", args.backbone)
    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        args.backbone, torch_dtype=torch.bfloat16
    ).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    if backbone.config.pad_token_id is None:
        backbone.config.pad_token_id = tok.pad_token_id

    log.info("loading targets %s", args.targets)
    tgt = torch.load(args.targets, map_location="cpu", weights_only=False)
    seqs_all: list[str] = tgt["sequences"]
    acts_all = tgt["activations"]  # list of (T, d)

    N = min(args.n_targets, len(seqs_all))
    log.info("evaluating first %d / %d targets", N, len(seqs_all))
    sources = seqs_all[:N]
    h_true = torch.stack([acts_all[i][-1].float() for i in range(N)], 0)  # (N, d)
    h_true_n = F.normalize(h_true, dim=-1)

    # anisotropy mean for centered metric (use h_true as the pool itself)
    mu = h_true.mean(dim=0, keepdim=True)  # (1, d)
    h_true_c = h_true - mu
    h_true_cn = F.normalize(h_true_c, dim=-1)
    log.info("anisotropy ||mu||=%.4f over %d targets", mu.norm().item(), N)

    results: dict = {
        "backbone": args.backbone,
        "layer": args.layer,
        "n_targets": N,
        "targets_path": str(args.targets),
        "anisotropy_mu_norm": float(mu.norm().item()),
    }

    # ----- (a) REPLAY: re-encode the source text and compare to h_true -----
    log.info("computing REPLAY (oracle re-encoding) ...")
    h_replay_chunks = []
    for i in range(0, N, args.encode_batch):
        batch = sources[i:i + args.encode_batch]
        h_replay_chunks.append(
            encode_last_hidden(backbone, tok, batch, args.layer, device,
                               max_len=args.max_encode_len)
        )
        log.info("  replay encode %d/%d", min(i + args.encode_batch, N), N)
    h_replay = torch.cat(h_replay_chunks, 0)  # (N, d)
    h_replay_n = F.normalize(h_replay, dim=-1)
    replay_cos = (h_true_n * h_replay_n).sum(-1)  # (N,)
    results.update(summary_stats("replay", replay_cos))
    replay_cen_cos = cos_rows(h_true - mu, h_replay - mu)
    results.update(summary_stats("replay_cen", replay_cen_cos))
    log.info("REPLAY raw fve=%.4f   centered fve=%.4f",
             results["replay_fve_mean"], results["replay_cen_fve_mean"])

    # ----- (b) RANDOM-FLOOR: pairwise cos between unrelated targets -----
    log.info("computing RANDOM-FLOOR (off-diagonal pairwise cos in pool) ...")
    cos_mat = h_true_n @ h_true_n.t()  # (N, N)
    eye = torch.eye(N, dtype=torch.bool)
    off = cos_mat[~eye]  # (N*(N-1),)
    results.update(summary_stats("random_floor", off))
    cos_mat_c = h_true_cn @ h_true_cn.t()
    off_c = cos_mat_c[~eye]
    results.update(summary_stats("random_floor_cen", off_c))
    log.info("RANDOM-FLOOR raw fve=%.4f   centered fve=%.4f",
             results["random_floor_fve_mean"], results["random_floor_cen_fve_mean"])

    # ----- (c) NN-IN-POOL: nearest other real Qwen text for each target -----
    log.info("computing NN-IN-POOL ...")
    cos_mat_masked = cos_mat.clone()
    cos_mat_masked.fill_diagonal_(-1.0)
    nn_cos = cos_mat_masked.max(dim=1).values
    results.update(summary_stats("nn_in_pool", nn_cos))
    cos_mat_c_masked = cos_mat_c.clone()
    cos_mat_c_masked.fill_diagonal_(-1.0)
    nn_cos_c = cos_mat_c_masked.max(dim=1).values
    results.update(summary_stats("nn_in_pool_cen", nn_cos_c))
    log.info("NN-IN-POOL raw fve=%.4f   centered fve=%.4f",
             results["nn_in_pool_fve_mean"], results["nn_in_pool_cen_fve_mean"])

    # ----- (d) PARAPHRASE CEILING: semantic-equivalent regenerations -----
    if args.paraphrase_k > 0:
        log.info("computing PARAPHRASE ceiling (k=%d, max_new=%d) ...",
                 args.paraphrase_k, args.paraphrase_max_new)
        per_para_cos: list[torch.Tensor] = []      # raw, one (k,) per source
        per_para_cos_c: list[torch.Tensor] = []    # centered, one (k,) per source
        for i in range(0, N, args.paraphrase_batch):
            batch_srcs = sources[i:i + args.paraphrase_batch]
            paras = sample_paraphrases(
                backbone, tok, batch_srcs,
                k=args.paraphrase_k, max_new=args.paraphrase_max_new,
                temperature=args.temperature, device=device,
            )
            # encode all k*|batch| paraphrases
            flat = [p for row in paras for p in row]
            # guard against empty strings
            flat = [t if t.strip() else " " for t in flat]
            h_para_chunks = []
            for j in range(0, len(flat), args.encode_batch):
                h_para_chunks.append(
                    encode_last_hidden(backbone, tok, flat[j:j + args.encode_batch],
                                       args.layer, device, max_len=args.max_encode_len)
                )
            h_para = torch.cat(h_para_chunks, 0)  # (|batch|*k, d)
            h_para_n = F.normalize(h_para, dim=-1)
            h_para_cn = F.normalize(h_para - mu, dim=-1)
            # per-source cos against the original h_true
            for bi, src_idx in enumerate(range(i, i + len(batch_srcs))):
                rows = h_para_n[bi * args.paraphrase_k:(bi + 1) * args.paraphrase_k]
                cos = (rows * h_true_n[src_idx].unsqueeze(0)).sum(-1)  # (k,)
                per_para_cos.append(cos)
                rows_c = h_para_cn[bi * args.paraphrase_k:(bi + 1) * args.paraphrase_k]
                cos_c = (rows_c * h_true_cn[src_idx].unsqueeze(0)).sum(-1)
                per_para_cos_c.append(cos_c)
            log.info("  paraphrase %d/%d", min(i + args.paraphrase_batch, N), N)
        para_mat = torch.stack(per_para_cos, 0)        # (N, k) raw
        para_mat_c = torch.stack(per_para_cos_c, 0)    # (N, k) centered
        results.update(summary_stats("paraphrase_all", para_mat))
        results.update(summary_stats("paraphrase_all_cen", para_mat_c))
        # "best paraphrase per source" — the ceiling we actually care about
        para_best = para_mat.max(dim=1).values
        para_best_c = para_mat_c.max(dim=1).values
        para_mean_per_src = para_mat.mean(dim=1)
        para_mean_per_src_c = para_mat_c.mean(dim=1)
        results.update(summary_stats("paraphrase_best", para_best))
        results.update(summary_stats("paraphrase_best_cen", para_best_c))
        results.update(summary_stats("paraphrase_mean_per_src", para_mean_per_src))
        results.update(summary_stats("paraphrase_mean_per_src_cen", para_mean_per_src_c))
        log.info("PARAPHRASE all   raw fve=%.4f  centered fve=%.4f",
                 results["paraphrase_all_fve_mean"], results["paraphrase_all_cen_fve_mean"])
        log.info("PARAPHRASE best  raw fve=%.4f  centered fve=%.4f",
                 results["paraphrase_best_fve_mean"], results["paraphrase_best_cen_fve_mean"])

    # ----- derived: normalised score template -----
    # rho = (adapter_fve - random_floor_fve) / (paraphrase_best_fve - random_floor_fve)
    # we don't have the adapter number in this script; just write the anchors.
    if args.paraphrase_k > 0:
        denom = results["paraphrase_best_fve_mean"] - results["random_floor_fve_mean"]
        denom_c = (results["paraphrase_best_cen_fve_mean"]
                   - results["random_floor_cen_fve_mean"])
        results["normalisation_denom_fve"] = denom
        results["normalisation_denom_fve_cen"] = denom_c
        results["formula"] = (
            "rho_raw = (adapter_best_of_N_fve - random_floor_fve_mean) / "
            "(paraphrase_best_fve_mean - random_floor_fve_mean); "
            "rho_cen analogous with *_cen_* fields."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(results, f, indent=2)
    log.info("wrote %s", args.out)
    log.info("=== headline (RAW fve_nrm) ===")
    log.info("replay      fve_mean=%.4f  (oracle, should be ~1.0)", results["replay_fve_mean"])
    log.info("random_flr  fve_mean=%.4f  (cos between unrelated samples)",
             results["random_floor_fve_mean"])
    log.info("nn_in_pool  fve_mean=%.4f  (best other real Qwen text)",
             results["nn_in_pool_fve_mean"])
    if args.paraphrase_k > 0:
        log.info("paraphrase  fve_mean=%.4f (best-of-%d Qwen paraphrase)",
                 results["paraphrase_best_fve_mean"], args.paraphrase_k)
    log.info("=== headline (CENTERED fve_nrm) ===")
    log.info("replay      fve_mean=%.4f", results["replay_cen_fve_mean"])
    log.info("random_flr  fve_mean=%.4f", results["random_floor_cen_fve_mean"])
    log.info("nn_in_pool  fve_mean=%.4f", results["nn_in_pool_cen_fve_mean"])
    if args.paraphrase_k > 0:
        log.info("paraphrase  fve_mean=%.4f (best-of-%d Qwen paraphrase)",
                 results["paraphrase_best_cen_fve_mean"], args.paraphrase_k)
        log.info("adapter best-of-64 from prior probe was 0.6750 (30k) / 0.6890 (10k ceiling); see centered_eval_*.json for centered numbers.")


if __name__ == "__main__":
    main()
