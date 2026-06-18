#!/usr/bin/env python3
"""Corrected-direction best-of-K selection on GSM8K.

The held-out AUROC test found that LOW SRT divergence / chain predicts a WRONG
answer (AUROC ~0.71). The earlier selector picked argMIN divergence — i.e. the
most wrong-prone sample. This script flips to the CORRECTED direction and tests
whether selecting by the SRT signals can now beat majority vote (0.905):

  greedy (K=1)         baseline
  majority vote        self-consistency, the bar to beat
  sel_div_high         pick the sample with the HIGHEST mean divergence
  sel_chain_high       pick the HIGHEST mean chain residual
  sel_entropy_low      pick the LOWEST mean entropy (confidence baseline)
  sel_combo            pick argmax of z(div)+z(chain)-z(entropy)
  oracle pass@K        ceiling

Per-sample aggregate signals are saved for offline re-analysis.

Run on a CUDA box (transformers==4.53.3):
    python scripts/srt_select_corrected.py --n 200 --k 8
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srt.adapter import SRTAdapter  # noqa: E402
from srt.config import SRTConfig  # noqa: E402
from scripts.inject_ab_eval import (  # noqa: E402
    extract_final_number, gold_number, truncate_continuation, FEWSHOT,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-repo", default="RiverRider/srt-adapter-v1.0")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="artifacts/inject_ab/srt_select_corrected.json")
    return p.parse_args()


def agg(sig):
    def col(k):
        return [s[k] for s in sig if k in s and s[k] is not None]
    m = lambda x: sum(x) / len(x) if x else 0.0
    return {"entropy": m(col("ent")), "div": m(col("div_norm")),
            "chain": m(col("chain")), "rhat": m(col("r_hat"))}


def zscores(vals):
    n = len(vals)
    mu = sum(vals) / n
    var = sum((v - mu) ** 2 for v in vals) / n
    sd = var ** 0.5 or 1.0
    return [(v - mu) / sd for v in vals]


METHODS = ["greedy", "majority", "sel_div_high", "sel_chain_high",
           "sel_entropy_low", "sel_combo", "oracle_passk"]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = SRTConfig.from_json(hf_hub_download(args.adapter_repo, "config.json"))
    cfg.backbone_dtype = args.dtype
    model = SRTAdapter(cfg)
    model.load_state_dict(load_safetensors(
        hf_hub_download(args.adapter_repo, "adapter.safetensors"), device="cpu"),
        strict=False)
    model = model.to(args.device).eval()
    tok = AutoTokenizer.from_pretrained(cfg.backbone_id)
    eos = {tok.eos_token_id}

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")

    def gen(ids, temperature, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        out_ids, sig = model.generate(
            ids, max_new_tokens=args.max_new_tokens, eos_token_ids=eos,
            temperature=temperature, top_p=args.top_p,
            disable_injectors=True, return_signals=True)
        ans = extract_final_number(truncate_continuation(
            tok.decode(out_ids[0], skip_special_tokens=True)))
        return ans, agg(sig)

    correct = {m: 0 for m in METHODS}
    rows = []
    for i in range(args.start, min(args.start + args.n, len(ds))):
        gold = gold_number(ds[i]["answer"])
        ids = tok(FEWSHOT + f"Question: {ds[i]['question']}\nAnswer:",
                  return_tensors="pt").input_ids.to(args.device)
        greedy_ans, _ = gen(ids, 0.0)
        samples = [gen(ids, args.temperature, seed=args.seed + i * 1000 + k)
                   for k in range(args.k)]
        answers = [a for a, _ in samples]
        sigs = [s for _, s in samples]
        valid = [a for a in answers if a is not None]
        majority = Counter(valid).most_common(1)[0][0] if valid else None

        def pick(key, hi=True):
            idx = max(range(len(samples)), key=lambda j: (sigs[j][key] if hi else -sigs[j][key]))
            return answers[idx]
        zd = zscores([s["div"] for s in sigs])
        zc = zscores([s["chain"] for s in sigs])
        ze = zscores([s["entropy"] for s in sigs])
        combo_idx = max(range(len(samples)), key=lambda j: zd[j] + zc[j] - ze[j])

        sel = {"greedy": greedy_ans, "majority": majority,
               "sel_div_high": pick("div", True),
               "sel_chain_high": pick("chain", True),
               "sel_entropy_low": pick("entropy", False),
               "sel_combo": answers[combo_idx]}
        passk = gold in answers
        for m in METHODS:
            if m == "oracle_passk":
                correct[m] += passk
            else:
                correct[m] += (sel.get(m) == gold)
        rows.append({"i": i, "gold": gold, "answers": answers, "sigs": sigs,
                     "sel": sel, "passk": passk})
        n = len(rows)
        if n % 10 == 0:
            acc = {m: round(correct[m] / n, 3) for m in METHODS}
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(
                {"n": n, "k": args.k, "partial": True, "accuracy": acc, "rows": rows}, indent=2))
            print(f"  [{n}] maj={acc['majority']} div_hi={acc['sel_div_high']} "
                  f"chain_hi={acc['sel_chain_high']} ent_lo={acc['sel_entropy_low']} "
                  f"combo={acc['sel_combo']} pass@k={acc['oracle_passk']}")

    n = len(rows)
    acc = {m: correct[m] / n if n else 0.0 for m in METHODS}
    Path(args.out).write_text(json.dumps(
        {"n": n, "k": args.k, "temperature": args.temperature,
         "accuracy": acc, "rows": rows}, indent=2))

    print("\n" + "=" * 60)
    print(f"CORRECTED-DIRECTION SELECTION — GSM8K (K={args.k}, n={n})")
    print("=" * 60)
    mv = acc["majority"]
    for m in METHODS:
        tag = "  <-- BEATS majority" if (m.startswith("sel_") and acc[m] > mv) else ""
        print(f"  {m:<18} {acc[m]:.3f}{tag}")
    print("=" * 60)
    print(f"  bar to beat: majority = {mv:.3f}")


if __name__ == "__main__":
    main()
