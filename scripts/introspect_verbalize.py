#!/usr/bin/env python3
"""Full SRT introspection: generate + verbalize the hidden state with the AV.

Loads the Trace stack (SRT adapter `RiverRider/srt-adapter-v1.0` + Activation
Verbalizer `RiverRider/srt-nla-av-v1`) and produces a continuation plus
natural-language verbalizations of the model's internal state at the
highest-effort token positions, each with a round-trip fidelity cosine.

Run on a CUDA box (transformers==4.53.3). Loads TWO backbone copies (~30 GB bf16).
    python scripts/introspect_verbalize.py --prompt "Prove whether P = NP." --max-new-tokens 300 --budget 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srt_introspect import Trace  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading Trace (adapter + activation verbalizer)...")
    trace = Trace.load(device=args.device)
    print(f"Trace ready on {trace.device}\n")

    result = trace.generate(
        args.prompt, max_new_tokens=args.max_new_tokens,
        budget=args.budget, temperature=args.temperature,
        repetition_penalty=1.15,
    )

    print("=" * 72)
    print(f"PROMPT: {args.prompt}")
    print("=" * 72)
    print("\n--- GENERATED TEXT ---")
    print(result.text)

    sel = result.selected()
    print(f"\n--- AV VERBALIZATIONS ({len(sel)} of {len(result.steps)} tokens) ---")
    print("Each card: what the AV decodes from the hidden state at that token,")
    print("with round-trip cosine (re-encode the verbalization, compare to original).\n")
    for s in sel:
        rt = "" if s.roundtrip_cos is None else f"  [round-trip cos={s.roundtrip_cos:.3f}]"
        tok = s.token.replace("\n", "\\n")
        print(f"  @tok#{s.token_idx} {tok!r}  (div={s.divergence:.2f} "
              f"regime={'super' if s.regime else 'sub'} r̂={s.r_hat:.2f} "
              f"H={s.entropy:.2f}){rt}")
        print(f"      “{s.verbalization}”\n")
    print("=" * 72)


if __name__ == "__main__":
    main()
