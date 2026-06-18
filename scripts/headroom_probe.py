#!/usr/bin/env python3
"""Best-of-K headroom probe on GSM8K, driven through the SRT-Showcase Space.

For each problem we draw K sampled solutions (bare backbone, temperature>0) and
measure three numbers that bound any selection method:

  * greedy (K=1)         : the deterministic baseline.
  * self-consistency     : majority vote over the K answers (blind, standard
                           test-time scaling) — the bar an SRT selector must beat.
  * pass@K (oracle)      : did ANY of the K samples match the gold answer — the
                           ceiling / headroom available to a perfect selector.

If pass@K >> self-consistency there is room for a smarter (e.g. SRT-signal)
selector. If self-consistency ~ pass@K, no blind selector can help and we stop.

Uses the bare backbone (inject OFF) since the n=30 A/B showed injection hurts.

Usage:
    python scripts/headroom_probe.py --n 25 --k 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.inject_ab_eval import extract_final_number, gold_number, truncate_continuation  # noqa: E402
from scripts.inject_ab_space import FEWSHOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--space", default="RiverRider/srt-showcase")
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--out", default="artifacts/inject_ab/gsm8k_headroom.json")
    return p.parse_args()


def hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"')
    return None


def main() -> None:
    args = parse_args()
    from gradio_client import Client
    client = Client(args.space, token=hf_token())

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")

    def gen(prompt: str, temperature: float) -> str | None:
        # cb_generate(prompt, mode, max_new, budget, k, temperature, top_p,
        #             rep, tint, inject). inject=False (bare backbone).
        r = client.predict(prompt, "Completion", args.max_new, 2, 4,
                           float(temperature), float(args.top_p), 1.0,
                           "entropy", False, api_name="/cb_generate")
        fo = r[4] if isinstance(r, (list, tuple)) else r
        return truncate_continuation(fo) if fo else None

    greedy_correct = sc_correct = passk_correct = 0
    rows = []
    quota_hit = False
    for i in range(min(args.n, len(ds))):
        gold = gold_number(ds[i]["answer"])
        prompt = FEWSHOT + f"Question: {ds[i]['question']}\nAnswer:"
        try:
            t = time.time()
            greedy_ans = extract_final_number(gen(prompt, 0.0) or "")
            samples = []
            for _ in range(args.k):
                cont = gen(prompt, args.temperature)
                samples.append(extract_final_number(cont or ""))
        except Exception as e:  # noqa: BLE001
            if "quota" in str(e).lower():
                print(f"  quota exhausted at problem {i + 1}")
                quota_hit = True
                break
            print(f"  [{i + 1}] error: {str(e)[:100]}")
            continue

        valid = [s for s in samples if s is not None]
        # self-consistency: majority vote over the K sampled answers.
        sc_ans = Counter(valid).most_common(1)[0][0] if valid else None
        passk = gold in samples  # oracle: any sample correct
        ok_greedy = greedy_ans == gold
        ok_sc = sc_ans == gold
        greedy_correct += ok_greedy
        sc_correct += ok_sc
        passk_correct += passk
        rows.append({"i": i, "gold": gold, "greedy": greedy_ans,
                     "sc": sc_ans, "samples": samples, "passk": passk})
        m = len(rows)
        print(f"  [{i + 1}/{args.n}] gold={gold:>6} greedy={'✓' if ok_greedy else '✗'} "
              f"sc={'✓' if ok_sc else '✗'} pass@{args.k}={'✓' if passk else '✗'} | "
              f"greedy={greedy_correct}/{m} sc={sc_correct}/{m} pass={passk_correct}/{m} "
              f"({time.time() - t:.0f}s)")

    n = len(rows)
    result = {"space": args.space, "n": n, "k": args.k,
              "temperature": args.temperature, "quota_exhausted": quota_hit,
              "acc_greedy": greedy_correct / n if n else 0.0,
              "acc_self_consistency": sc_correct / n if n else 0.0,
              "acc_passk_oracle": passk_correct / n if n else 0.0,
              "rows": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print(f"HEADROOM PROBE — GSM8K (K={args.k}, n={n}"
          f"{', quota-limited' if quota_hit else ''})")
    print("=" * 60)
    if n:
        print(f"  greedy (K=1)         : {greedy_correct}/{n} = {result['acc_greedy']:.3f}")
        print(f"  self-consistency vote: {sc_correct}/{n} = {result['acc_self_consistency']:.3f}")
        print(f"  pass@{args.k} (oracle)    : {passk_correct}/{n} = {result['acc_passk_oracle']:.3f}")
        headroom = result["acc_passk_oracle"] - result["acc_self_consistency"]
        print(f"  HEADROOM (pass@K - SC): {headroom:+.3f}  "
              f"<- room for a smarter selector (e.g. SRT)")
    print("=" * 60)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
