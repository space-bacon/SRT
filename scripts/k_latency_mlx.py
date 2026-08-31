#!/usr/bin/env python3
"""What K costs on the machine that actually serves, not on the research box.

The batching argument is a CUDA/PyTorch fact, and the lab serves from an M2 Ultra
through MLX. Apple Silicon has unified memory and different bandwidth
characteristics, so the knee could sit somewhere else entirely, or the curve
could be a straight line. Assuming it transfers is exactly the class of mistake
that produced "8x the compute" in the first place.

`sunstone_server.py` calls generate/stream_generate, neither of which batches, so
K samples there really would cost K sequential turns. `mlx_lm.batch_generate`
takes a list of prompts, so K copies of one prompt can share a step. This
measures whether that helps enough to be worth wiring in.

Run it when the lab is quiet: it competes with the live demo for the same GPU.

    python scripts/k_latency_mlx.py --model mlx-community/Qwen2.5-Coder-14B-Instruct-4bit
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Write a Python function that merges two sorted lists.",
    "Write a Python function that counts vowels in a string.",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mlx-community/Qwen2.5-Coder-14B-Instruct-4bit")
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="artifacts/nla/k_latency_mlx.json")
    a = ap.parse_args()

    from mlx_lm import load
    from mlx_lm.generate import batch_generate

    t0 = time.time()
    model, tok = load(a.model)
    print(f"loaded {a.model} in {time.time() - t0:.0f}s", flush=True)

    res = {"question": "what does K cost on the MLX serving path",
           "model": a.model, "max_tokens": a.max_tokens, "repeats": a.repeats,
           "note": "sunstone_server uses generate/stream_generate, which do not "
                   "batch; this measures mlx_lm.batch_generate instead",
           "k": {}}

    for k in a.ks:
        times = []
        for p in PROMPTS:
            ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                          add_generation_prompt=True)
            for r in range(a.repeats):
                s = time.time()
                batch_generate(model, tok, prompts=[ids] * k,
                               max_tokens=a.max_tokens, verbose=False)
                # First repeat pays kernel compilation, so it is excluded.
                if r:
                    times.append(time.time() - s)
        res["k"][str(k)] = {"median_s": round(statistics.median(times), 3),
                            "min_s": round(min(times), 3), "n": len(times)}
        print(f"  K={k:3d}  median {res['k'][str(k)]['median_s']:6.2f}s", flush=True)

    base = res["k"][str(a.ks[0])]["median_s"]
    for k in a.ks:
        res["k"][str(k)]["vs_k1"] = round(res["k"][str(k)]["median_s"] / base, 2)
    res["reading"] = ("vs_k1 well under k means Apple Silicon amortises the batch "
                      "the way CUDA does, and selection is deployable on the Mac")

    print("\n  K    wall     vs K=1   naive")
    for k in a.ks:
        v = res["k"][str(k)]
        print(f"  {k:3d}  {v['median_s']:6.2f}s  {v['vs_k1']:6.2f}x   {k}x", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
