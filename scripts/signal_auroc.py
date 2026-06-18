#!/usr/bin/env python3
"""Can any SRT read-out signal predict GSM8K correctness?

For each problem we greedily generate one solution, capture the per-token SRT
read-out signals (entropy, divergence, regime P(super), reflexivity r_hat, chain
residual), aggregate them over the generation, and record whether the final
answer was correct. We then compute the AUROC of each signal for predicting
correctness (probability that a correct generation scores lower/higher than a
wrong one).

This directly tests "can SRT-introspect tell when the model is wrong?". Prior:
entropy is the validated uncertainty signal; the SRT side-channels were ~chance
for hallucination. AUROC=0.5 means no predictive power.

Run on a CUDA box (transformers==4.53.3):
    python scripts/signal_auroc.py --n 200
"""
from __future__ import annotations

import argparse
import json
import sys
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

# Signal -> True if "higher value should indicate WRONG" (uncertainty-like).
# We orient every AUROC so >0.5 means the signal is predictive of being WRONG.
SIGNALS = ["mean_entropy", "peak_entropy", "mean_div_norm", "peak_div_norm",
           "regime_super_frac", "mean_r_hat", "mean_chain"]

# Directions PRE-REGISTERED from the first n=200 run (problems 0-199), to be
# tested on a disjoint held-out slice. value = the direction in which the signal
# was claimed to predict a WRONG answer. "high" => high value predicts wrong.
PREREG = {
    "mean_entropy": "high",      # validated uncertainty: high entropy -> wrong
    "peak_entropy": "low",       # run1: high peak entropy -> correct
    "mean_div_norm": "low",      # HYPOTHESIS: confidently-wrong = smooth trajectory
    "peak_div_norm": "high",
    "regime_super_frac": "low",
    "mean_r_hat": "low",
    "mean_chain": "low",         # HYPOTHESIS: low chain residual -> wrong
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-repo", default="RiverRider/srt-adapter-v1.0")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--start", type=int, default=0,
                   help="offset into the GSM8K test set (for disjoint held-out slices)")
    p.add_argument("--start", type=int, default=0,
                   help="skip the first --start problems (for a disjoint held-out slice)")
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="artifacts/inject_ab/signal_auroc.json")
    return p.parse_args()


def aggregate(sig: list[dict]) -> dict:
    def col(k):
        return [s[k] for s in sig if k in s and s[k] is not None]
    ent = col("ent"); div = col("div_norm"); reg = col("regime_super")
    rhat = col("r_hat"); chain = col("chain")
    mean = lambda x: sum(x) / len(x) if x else 0.0
    mx = lambda x: max(x) if x else 0.0
    return {
        "mean_entropy": mean(ent), "peak_entropy": mx(ent),
        "mean_div_norm": mean(div), "peak_div_norm": mx(div),
        "regime_super_frac": mean(reg), "mean_r_hat": mean(rhat),
        "mean_chain": mean(chain),
    }


def auroc(scores: list[float], labels: list[int]) -> float:
    """AUROC via Mann-Whitney U. labels: 1 = positive class (WRONG)."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    u = rank_pos - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


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

    feats: list[dict] = []
    wrong: list[int] = []
    n_correct = 0
    idxs = range(args.start, min(args.start + args.n, len(ds)))
    for i in idxs:
        gold = gold_number(ds[i]["answer"])
        ids = tok(FEWSHOT + f"Question: {ds[i]['question']}\nAnswer:",
                  return_tensors="pt").input_ids.to(args.device)
        out_ids, sig = model.generate(
            ids, max_new_tokens=args.max_new_tokens, eos_token_ids=eos,
            temperature=0.0, disable_injectors=True, return_signals=True)
        ans = extract_final_number(truncate_continuation(
            tok.decode(out_ids[0], skip_special_tokens=True)))
        ok = ans == gold
        n_correct += ok
        feats.append(aggregate(sig))
        wrong.append(0 if ok else 1)
        if len(feats) % 20 == 0:
            print(f"  [{len(feats)}/{args.n}] acc so far {n_correct}/{len(feats)}")

    n = len(feats)
    aurocs = {}
    for s in SIGNALS:
        scores = [f[s] for f in feats]
        a = auroc(scores, wrong)
        prereg_dir = PREREG[s]
        prereg = a if prereg_dir == "high" else (1 - a)
        aurocs[s] = {"auroc_wrong_if_high": round(a, 4),
                     "auroc_oriented": round(max(a, 1 - a), 4),
                     "prereg_direction": prereg_dir,
                     "auroc_prereg": round(prereg, 4),
                     "direction": "high=wrong" if a >= 0.5 else "low=wrong"}

    result = {"adapter": args.adapter_repo, "n": n,
              "accuracy": n_correct / n if n else 0.0,
              "base_rate_wrong": sum(wrong) / n if n else 0.0,
              "auroc": aurocs}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**result, "feats": feats, "wrong": wrong}, indent=2))

    print("\n" + "=" * 64)
    print(f"SRT SIGNAL → GSM8K CORRECTNESS  (problems {args.start}–{args.start + n - 1}, "
          f"acc={result['accuracy']:.3f})")
    print("AUROC for predicting a WRONG answer (0.5 = no predictive power)")
    print("pre-reg = AUROC in the direction fixed BEFORE seeing this slice")
    print("=" * 64)
    print(f"  {'signal':<20} {'pre-reg':>8} {'(dir)':>6}  {'post-hoc best':>13}")
    for s in SIGNALS:
        a = aurocs[s]
        print(f"  {s:<20} {a['auroc_prereg']:8.3f} {a['prereg_direction']:>6}  "
              f"{a['auroc_oriented']:13.3f}")
    print("=" * 64)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
