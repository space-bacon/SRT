#!/usr/bin/env python3
"""What does K actually cost in wall-clock, per answer served.

I described selection as costing "8x the compute". That is wrong, and the reason
matters for deployment. Autoregressive decode is memory-bandwidth bound: every
step streams the weights out of HBM once, and that cost is identical whether the
step advances 1 sequence or 8. Prefill is paid once and shared across the K
samples. So K samples in ONE batch cost far less than K single samples, and the
marginal sample is nearly free until K exceeds what fits in a batch.

That predicts a curve with a knee, not a line: flat while the batch absorbs K,
then stepping each time K spills into another batch. Where the knee sits is a
property of the card and the model, so it has to be measured on the hardware you
serve from, not argued from theory.

Reports wall-clock per turn and the multiplier over K=1, which is the number that
decides whether selection is deployable in an interactive chat.

    python scripts/k_latency.py --model Qwen/Qwen2.5-Coder-14B-Instruct
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))

HUBS = [h for h in (os.environ.get("HF_HOME", "") and
                    os.path.join(os.environ["HF_HOME"], "hub"),
                    "/workspace/.hf_home/hub", "/root/.hf_home/hub") if h]


def snap(repo):
    stem = f"models--{repo.replace('/', '--')}/snapshots/*/"
    for hub in HUBS:
        g = glob.glob(f"{hub}/{stem}")
        if g:
            return g[0]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--probs", default="data/mbpp.json")
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--out", default="artifacts/nla/k_latency.json")
    a = ap.parse_args()

    path = snap(a.model)
    if path is None:
        print(f"MISSING {a.model}", flush=True)
        return 1

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    t0 = time.time()
    mod = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)

    probs = json.load(open(os.path.join(HERE, a.probs)))[:a.n_prompts]
    instr = "Write a Python function for the task below. Reply with the full function only.\n\n"

    res = {"question": "what does K cost in wall-clock per answer served",
           "model": a.model, "max_new": a.max_new, "n_prompts": a.n_prompts,
           "repeats": a.repeats, "device": torch.cuda.get_device_name(0), "k": {}}

    for k in a.ks:
        times = []
        for p in probs:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": instr + p["prompt"]}],
                tokenize=True, add_generation_prompt=True)
            ids = ids["input_ids"] if hasattr(ids, "keys") else ids
            b = tok.pad({"input_ids": [ids]}, return_tensors="pt").to("cuda")
            for r in range(a.repeats):
                torch.cuda.synchronize()
                s = time.time()
                with torch.no_grad():
                    mod.generate(**b, max_new_tokens=a.max_new, do_sample=True,
                                 top_p=0.9, temperature=1.0, num_return_sequences=k,
                                 pad_token_id=tok.pad_token_id)
                torch.cuda.synchronize()
                # First repeat pays kernel warmup, so it is not part of the estimate.
                if r:
                    times.append(time.time() - s)
        res["k"][str(k)] = {
            "median_s": round(statistics.median(times), 3),
            "min_s": round(min(times), 3),
            "n": len(times),
        }
        print(f"  K={k:3d}  median {res['k'][str(k)]['median_s']:6.2f}s", flush=True)

    base = res["k"][str(a.ks[0])]["median_s"]
    for k in a.ks:
        v = res["k"][str(k)]
        v["vs_k1"] = round(v["median_s"] / base, 2)
        v["naive_expectation"] = k
    res["reading"] = ("vs_k1 far below k means batching absorbs the extra samples, "
                      "so selection does not cost k times a single answer")
    print("\n  K    wall    vs K=1   naive")
    for k in a.ks:
        v = res["k"][str(k)]
        print(f"  {k:3d}  {v['median_s']:6.2f}s  {v['vs_k1']:5.2f}x   {k}x", flush=True)

    op = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(res, open(op, "w"), indent=2)
    print(f"\nwrote {op}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
