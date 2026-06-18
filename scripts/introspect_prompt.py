#!/usr/bin/env python3
"""Generate from the frozen backbone + SRT adapter and report internal states.

Streams a single greedy generation while capturing the SRT read-out signals per
token (entropy, metapragmatic divergence, BEN regime P(supercritical), reflexivity
r_hat, chain residual), then summarizes where the model's internal state moved
most. This is SRT's validated use: observing the model think, not steering it.

Run on a CUDA box with transformers==4.53.3 (5.x breaks generation):
    python scripts/introspect_prompt.py --prompt "Prove whether P = NP." --max-new-tokens 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srt.adapter import SRTAdapter  # noqa: E402
from srt.config import SRTConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-repo", default="RiverRider/srt-adapter-v1.0")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--inject", action="store_true", help="enable FiLM injection")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--top-moments", type=int, default=12)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = SRTConfig.from_json(hf_hub_download(args.adapter_repo, "config.json"))
    cfg.backbone_dtype = args.dtype
    model = SRTAdapter(cfg)
    model.load_state_dict(
        load_safetensors(hf_hub_download(args.adapter_repo, "adapter.safetensors"),
                         device="cpu"), strict=False)
    model = model.to(args.device).eval()
    tok = AutoTokenizer.from_pretrained(cfg.backbone_id)

    ids = tok(args.prompt, return_tensors="pt").input_ids.to(args.device)
    out_ids, sig = model.generate(
        ids, max_new_tokens=args.max_new_tokens, eos_token_ids={tok.eos_token_id},
        temperature=args.temperature, disable_injectors=(not args.inject),
        return_signals=True)
    text = tok.decode(out_ids[0], skip_special_tokens=True)

    # Per-token decode aligned to signal_log (one entry per generated token).
    toks = [tok.decode([t]) for t in out_ids[0].tolist()]

    def col(key):
        return [s.get(key) for s in sig]
    ent = col("ent"); div = col("div_norm"); reg = col("regime_super")
    rhat = col("r_hat"); chain = col("chain")

    def stats(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return (0.0, 0.0, 0.0)
        return (sum(xs) / len(xs), max(xs), min(xs))

    print("\n" + "=" * 70)
    print(f"PROMPT: {args.prompt}")
    print(f"INJECT: {args.inject}   tokens: {len(sig)}")
    print("=" * 70)
    print("\n--- GENERATED TEXT ---")
    print(text)

    print("\n--- INTERNAL STATE (whole generation) ---")
    for name, xs in [("entropy (nats)", ent), ("divergence |Δ|", div),
                     ("regime P(super)", reg), ("reflexivity r_hat", rhat),
                     ("chain residual", chain)]:
        m, hi, lo = stats(xs)
        print(f"  {name:20s} mean={m:7.3f}  peak={hi:7.3f}  min={lo:7.3f}")
    rsup = [r for r in reg if r is not None]
    if rsup:
        frac = sum(1 for r in rsup if r > 0.5) / len(rsup)
        print(f"  supercritical fraction: {frac:.1%} of tokens (regime 'bifurcating')")

    # Where did the internal state move most? Rank tokens by divergence.
    scored = [(i, div[i], ent[i], reg[i]) for i in range(len(sig))
              if div[i] is not None]
    scored.sort(key=lambda r: r[1], reverse=True)
    print(f"\n--- TOP {args.top_moments} HIGH-DIVERGENCE TOKENS (internal 'struggle') ---")
    print(f"  {'tok#':>4}  {'token':<16} {'div':>7} {'entropy':>8} {'P(super)':>9}")
    for i, d, e, r in scored[:args.top_moments]:
        t = toks[i].replace("\n", "\\n")[:16]
        print(f"  {i:>4}  {t:<16} {d:7.3f} {(e or 0):8.3f} {(r or 0):9.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
