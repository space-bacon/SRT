"""Demo: generate a continuation with an adaptive-density reasoning trace.

    python scripts/demos/trace_demo.py
    python scripts/demos/trace_demo.py --prompt "..." --max-new-tokens 300 --budget 16 --k 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from srt_introspect import Trace


DEFAULT_PROMPT = (
    "Q: A retired plumber from Ohio claims he invented a self-cooling beer "
    "can in 1973. Is this likely true, and what would have to be different "
    "about thermodynamics for it to work?\nA:"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--budget", type=int, default=12, help="adaptive verbalization slots")
    p.add_argument("--k", type=int, default=8, help="AV samples per slot")
    p.add_argument("--device", default=None)
    p.add_argument("--out-json", default=None, help="optional path to dump full trace")
    args = p.parse_args()

    print(f"loading Trace (device={args.device or 'auto'})…")
    t0 = time.perf_counter()
    t = Trace.load(device=args.device)
    print(f"  ready in {time.perf_counter()-t0:.1f}s")

    print(f"\nprompt: {args.prompt!r}")
    print(f"  max_new_tokens={args.max_new_tokens} budget={args.budget} k={args.k}")
    t0 = time.perf_counter()
    result = t.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        budget=args.budget,
        k=args.k,
    )
    elapsed = time.perf_counter() - t0
    print(f"  generated {len(result.steps)} tokens in {elapsed:.1f}s")

    print("\n--- continuation ---")
    print(result.text)

    print("\n--- divergence summary ---")
    if result.steps:
        divs = [s.divergence for s in result.steps]
        print(f"  min={min(divs):.2f}  max={max(divs):.2f}  mean={sum(divs)/len(divs):.2f}")
        regimes = [s.regime for s in result.steps]
        print(f"  regime counts: 0={regimes.count(0)} 1={regimes.count(1)}")

    print("\n--- selected verbalizations ---")
    for s in result.selected():
        print(f"  [{s.token_idx:>4}] tok={s.token!r:>14}  d={s.divergence:>5.1f}  r̂={s.r_hat:.2f}  reg={s.regime}")
        print(f"           → {s.verbalization[:200]}")

    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump(
                {
                    "prompt": args.prompt,
                    "text": result.text,
                    "elapsed_s": elapsed,
                    "steps": [
                        {
                            "i": s.token_idx,
                            "token": s.token,
                            "divergence": s.divergence,
                            "regime": s.regime,
                            "r_hat": s.r_hat,
                            "verbalization": s.verbalization,
                        }
                        for s in result.steps
                    ],
                },
                f,
                indent=2,
            )
        print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
