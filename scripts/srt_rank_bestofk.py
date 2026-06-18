#!/usr/bin/env python3
"""SRT-ranked best-of-K on GSM8K (CUDA box).

Tests the real hypothesis: can a BLIND SRT read-out signal rank K sampled
solutions better than majority vote? For each problem we draw K samples from the
bare backbone (inject OFF — the n=30 A/B showed injection hurts) while capturing
per-token SRT signals via adapter.generate(return_signals=True). We then score
several selectors and compare:

  greedy (K=1)              baseline
  majority vote             self-consistency, the bar to beat
  SRT-ranked (per signal)   pick the sample optimizing an SRT aggregate
  logp-ranked               pick highest mean seq-logprob (DEAD in NLA; control)
  oracle pass@K / best      ceiling

Win condition: an SRT-ranked selector beats majority vote (and logp).

Run on a CUDA box (the KV-cached generate() path is correct on CUDA; broken on
MPS/CPU — see /memories/repo/cached-step-generate-bug.md):

    python scripts/srt_rank_bestofk.py --n 200 --k 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srt.adapter import SRTAdapter  # noqa: E402
from srt.config import SRTConfig  # noqa: E402
from scripts.inject_ab_eval import (  # noqa: E402
    extract_final_number, gold_number, truncate_continuation, FEWSHOT,
)

# Each entry: (key, direction). direction=-1 means "lower is better" selector.
# Hypotheses: smooth/confident/subcritical/self-consistent reasoning = correct.
SRT_RANKERS = [
    ("mean_entropy", -1),     # most confident
    ("mean_div_norm", -1),    # smoothest internal trajectory
    ("regime_super_frac", -1),  # most subcritical
    ("mean_r_hat", -1),       # least reflexive bifurcation
    ("mean_chain", -1),       # most self-consistent chain residual
    ("mean_nll", -1),         # logp control (highest seq prob)
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-repo", default="RiverRider/srt-adapter-v1.0")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="artifacts/inject_ab/srt_rank_bestofk.json")
    return p.parse_args()


def aggregate_signals(signal_log: list[dict]) -> dict:
    """Mean SRT read-out signals over the generated tokens (read-only taps)."""
    def col(key):
        return [s[key] for s in signal_log if key in s and s[key] is not None]
    ent = col("ent"); div = col("div_norm"); rsup = col("regime_super")
    rhat = col("r_hat"); chain = col("chain"); nll = col("nll")
    mean = lambda xs: float(sum(xs) / len(xs)) if xs else float("inf")
    return {
        "mean_entropy": mean(ent),
        "mean_div_norm": mean(div),
        "regime_super_frac": mean(rsup),  # mean P(supercritical)
        "mean_r_hat": mean(rhat),
        "mean_chain": mean(chain),
        "mean_nll": mean(nll),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    from huggingface_hub import hf_hub_download
    cfg = SRTConfig.from_json(hf_hub_download(args.adapter_repo, "config.json"))
    cfg.backbone_dtype = args.dtype
    print(f"== SRT-ranked best-of-K | {cfg.backbone_id} | K={args.k} n={args.n} "
          f"device={args.device} ==")

    t0 = time.time()
    model = SRTAdapter(cfg)
    w = load_safetensors(hf_hub_download(args.adapter_repo, "adapter.safetensors"), device="cpu")
    model.load_state_dict(w, strict=False)
    model = model.to(args.device).eval()
    tok = AutoTokenizer.from_pretrained(cfg.backbone_id)
    eos_ids = {tok.eos_token_id}
    print(f"   ready in {time.time() - t0:.0f}s")

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")

    def one_gen(ids, temperature, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                             eos_token_ids=eos_ids, temperature=temperature,
                             top_p=args.top_p, disable_injectors=True,
                             return_signals=True)
        new_ids, sig = out
        text = truncate_continuation(tok.decode(new_ids[0], skip_special_tokens=True))
        return extract_final_number(text), aggregate_signals(sig)

    methods = ["greedy", "majority"] + [f"srt_{k}" for k, _ in SRT_RANKERS] + \
              ["oracle_passk", "oracle_best"]
    correct = {m: 0 for m in methods}
    rows = []
    for i in range(min(args.n, len(ds))):
        gold = gold_number(ds[i]["answer"])
        ids = tok(FEWSHOT + f"Question: {ds[i]['question']}\nAnswer:",
                  return_tensors="pt").input_ids.to(args.device)
        t = time.time()
        greedy_ans, _ = one_gen(ids, 0.0)
        samples = []  # (answer, aggregates)
        for k in range(args.k):
            a, agg = one_gen(ids, args.temperature, seed=args.seed + i * 1000 + k)
            samples.append((a, agg))

        answers = [a for a, _ in samples]
        valid = [a for a in answers if a is not None]
        majority = Counter(valid).most_common(1)[0][0] if valid else None

        sel = {"greedy": greedy_ans, "majority": majority}
        for key, direction in SRT_RANKERS:
            best_idx = min(range(len(samples)),
                           key=lambda j: direction * samples[j][1][key])
            sel[f"srt_{key}"] = samples[best_idx][0]
        passk = gold in answers
        sel["oracle_passk"] = gold if passk else None
        sel["oracle_best"] = gold if passk else (majority)

        for m in methods:
            if m == "oracle_passk":
                correct[m] += passk
            else:
                correct[m] += (sel.get(m) == gold)
        rows.append({"i": i, "gold": gold, "answers": answers,
                     "sel": {k: sel.get(k) for k in sel}})
        n = len(rows)
        flags = " ".join(f"{m.replace('srt_','S:')}={'✓' if (sel.get(m)==gold or (m=='oracle_passk' and passk)) else '✗'}"
                         for m in ["greedy", "majority", "srt_mean_div_norm", "oracle_passk"])
        print(f"  [{i + 1}/{args.n}] gold={gold:>7} {flags} ({time.time() - t:.0f}s)")

    n = len(rows)
    result = {"adapter": args.adapter_repo, "n": n, "k": args.k,
              "temperature": args.temperature,
              "accuracy": {m: correct[m] / n if n else 0.0 for m in methods}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**result, "rows": rows}, indent=2))

    print("\n" + "=" * 64)
    print(f"SRT-RANKED BEST-OF-K — GSM8K (K={args.k}, n={n})")
    print("=" * 64)
    acc = result["accuracy"]
    mv = acc["majority"]
    for m in methods:
        tag = ""
        if m.startswith("srt_") or m == "logp":
            tag = "  <-- BEATS majority" if acc[m] > mv else ""
        print(f"  {m:22s} {acc[m]:.3f}{tag}")
    print("=" * 64)
    print(f"  win condition: any srt_* > majority ({mv:.3f})")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
