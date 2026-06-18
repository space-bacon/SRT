#!/usr/bin/env python3
"""Inject-ON vs inject-OFF A/B on GSM8K, driven through the running SRT-Showcase
Space (CUDA / ZeroGPU) via gradio_client.

This avoids renting a box: it reuses the already-deployed adapter on the exact
backend (CUDA) where the KV-cached generate() path is correct. Greedy decoding
(temperature=0) makes `inject` the only variable. Scoring is objective exact-match
on the GSM8K gold number.

ZeroGPU has a per-account daily quota, so this is a *probe*, not a full benchmark:
the script saves partial results and stops cleanly when quota is exhausted. The
same driver works at scale if the Space is temporarily moved to a dedicated GPU.

Usage:
    python scripts/inject_ab_space.py --n 30 --space RiverRider/srt-showcase
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.inject_ab_eval import (  # noqa: E402
    extract_final_number, gold_number, truncate_continuation,
)

# Compact 2-shot CoT to stay under the Space's MAX_PROMPT_CHARS=1500 guard.
FEWSHOT = (
    "Question: Natalia sold clips to 48 of her friends in April, and then she "
    "sold half as many clips in May. How many clips did she sell altogether?\n"
    "Answer: In April she sold 48. In May she sold 48 / 2 = 24. Altogether "
    "48 + 24 = 72. The answer is 72.\n\n"
    "Question: Weng earns $12 an hour for babysitting. Yesterday she did 50 "
    "minutes. How much did she earn?\n"
    "Answer: Per minute she earns 12 / 60 = 0.2. For 50 minutes that is "
    "50 * 0.2 = 10 dollars. The answer is 10.\n\n"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--space", default="RiverRider/srt-showcase")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--out", default="artifacts/inject_ab/gsm8k_space.json")
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

    token = hf_token()
    print(f"== inject A/B via Space {args.space} | token={'yes' if token else 'NO'} "
          f"| n={args.n} (greedy) ==")
    client = Client(args.space, token=token)

    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")

    def gen(prompt: str, inject: bool) -> str | None:
        # cb_generate(prompt, mode, max_new, budget, k, temperature, top_p,
        #             repetition_penalty, tint, inject) -> (..., final_output)
        r = client.predict(prompt, "Completion", args.max_new, 2, 4,
                           0.0, 1.0, 1.0, "entropy", inject,
                           api_name="/cb_generate")
        fo = r[4] if isinstance(r, (list, tuple)) else r
        return truncate_continuation(fo)

    both = on_only = off_only = neither = 0
    on_correct = off_correct = 0
    rows = []
    quota_hit = False
    for i in range(min(args.n, len(ds))):
        q = ds[i]["question"]
        gold = gold_number(ds[i]["answer"])
        prompt = FEWSHOT + f"Question: {q}\nAnswer:"
        try:
            t = time.time()
            cont_on = gen(prompt, True)
            cont_off = gen(prompt, False)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "quota" in msg.lower():
                print(f"  quota exhausted at problem {i + 1}: {msg[:120]}")
                quota_hit = True
                break
            print(f"  [{i + 1}] error: {msg[:120]}")
            continue
        a_on = extract_final_number(cont_on) if cont_on else None
        a_off = extract_final_number(cont_off) if cont_off else None
        ok_on, ok_off = a_on == gold, a_off == gold
        on_correct += ok_on
        off_correct += ok_off
        both += ok_on and ok_off
        on_only += ok_on and not ok_off
        off_only += ok_off and not ok_on
        neither += (not ok_on) and (not ok_off)
        rows.append({"i": i, "gold": gold, "on": a_on, "off": a_off,
                     "ok_on": ok_on, "ok_off": ok_off})
        print(f"  [{i + 1}/{args.n}] gold={gold:>6} on={str(a_on):>6}{'✓' if ok_on else '✗'} "
              f"off={str(a_off):>6}{'✓' if ok_off else '✗'} | "
              f"acc on={on_correct}/{len(rows)} off={off_correct}/{len(rows)} "
              f"({time.time() - t:.0f}s)")

    n = len(rows)
    result = {"space": args.space, "n": n, "decoding": "greedy",
              "quota_exhausted": quota_hit,
              "acc_on": on_correct / n if n else 0.0,
              "acc_off": off_correct / n if n else 0.0,
              "delta": (on_correct - off_correct) / n if n else 0.0,
              "mcnemar": {"both_right": both, "on_only": on_only,
                          "off_only": off_only, "neither": neither},
              "rows": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print(f"INJECT A/B via Space — GSM8K (greedy, n={n}"
          f"{', quota-limited' if quota_hit else ''})")
    print("=" * 60)
    if n:
        print(f"  acc inject-ON : {on_correct}/{n} = {result['acc_on']:.3f}")
        print(f"  acc inject-OFF: {off_correct}/{n} = {result['acc_off']:.3f}")
        print(f"  delta (on-off): {result['delta']:+.3f}")
        print(f"  disagreements : on-only={on_only}  off-only={off_only}  "
              f"(both={both}, neither={neither})")
    print("=" * 60)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
