"""Mini benchmark: run `srt_introspect.Trace.generate` over a small
prompt set and emit per-prompt JSON traces + an aggregate summary.

This is *not* the per-token detector eval (see scripts/instrument_eval.py
for that). It validates the **product** surface: latency, divergence
distribution, regime balance, and that verbalizations are non-empty.

    python scripts/evals/trace_bench.py --out-dir artifacts/trace_bench
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from srt_introspect import Trace


DEFAULT_PROMPTS = [
    "Q: What killed the dinosaurs?\nA:",
    "Q: Is the speed of light constant in all reference frames?\nA:",
    "Q: Why does pasta water boil over so easily?\nA:",
    "Q: A retired plumber from Ohio claims he invented a self-cooling beer can in 1973. Is this likely true?\nA:",
    "Q: Briefly summarize the plot of Hamlet.\nA:",
    "Q: What is gradient descent, in two sentences?\nA:",
    "Q: Why is the sky blue at noon but red at sunset?\nA:",
    "Q: Translate to French: 'The trains are on strike again.'\nA:",
]


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    xs_sorted = sorted(xs)
    n = len(xs)
    return {
        "n": n,
        "min": xs_sorted[0],
        "max": xs_sorted[-1],
        "mean": sum(xs) / n,
        "p50": xs_sorted[n // 2],
        "p90": xs_sorted[min(n - 1, int(0.9 * n))],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts-file", type=Path, default=None,
                   help="JSONL with {'prompt': ...} or plain TXT (one per line)")
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/trace_bench"))
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.prompts_file:
        text = args.prompts_file.read_text()
        if args.prompts_file.suffix == ".jsonl":
            prompts = [json.loads(l)["prompt"] for l in text.splitlines() if l.strip()]
        else:
            prompts = [l for l in text.splitlines() if l.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading Trace (device={args.device or 'auto'})…")
    t = Trace.load(device=args.device)

    runs = []
    all_divs: list[float] = []
    all_regimes: list[int] = []
    all_rhats: list[float] = []
    total_verbalizations = 0
    nonempty_verbalizations = 0

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt[:80]!r}")
        t0 = time.perf_counter()
        r = t.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            budget=args.budget,
            k=args.k,
        )
        elapsed = time.perf_counter() - t0
        divs = [s.divergence for s in r.steps]
        regimes = [s.regime for s in r.steps]
        rhats = [s.r_hat for s in r.steps]
        sel = r.selected()
        ne = sum(1 for s in sel if s.verbalization and s.verbalization.strip())
        print(f"  {len(r.steps)} tok in {elapsed:.1f}s · div mean={sum(divs)/max(1,len(divs)):.2f}"
              f" · regime[0]={regimes.count(0)} regime[1]={regimes.count(1)}"
              f" · verb {ne}/{len(sel)} nonempty")

        all_divs.extend(divs)
        all_regimes.extend(regimes)
        all_rhats.extend(rhats)
        total_verbalizations += len(sel)
        nonempty_verbalizations += ne

        # dump per-prompt trace
        out = args.out_dir / f"trace_{i:02d}.json"
        with open(out, "w") as f:
            json.dump(
                {
                    "prompt": prompt,
                    "text": r.text,
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
                        for s in r.steps
                    ],
                },
                f,
                indent=2,
            )
        runs.append({
            "i": i,
            "prompt": prompt,
            "n_tokens": len(r.steps),
            "elapsed_s": elapsed,
            "tok_per_s": len(r.steps) / max(1e-6, elapsed),
            "div_mean": sum(divs) / max(1, len(divs)),
            "regime_1_frac": regimes.count(1) / max(1, len(regimes)),
            "n_verbalizations": len(sel),
            "n_nonempty": ne,
        })

    summary = {
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "budget": args.budget,
            "k": args.k,
            "device": args.device or "auto",
            "n_prompts": len(prompts),
        },
        "divergence": _stats(all_divs),
        "r_hat": _stats(all_rhats),
        "regime_1_frac": (sum(r for r in all_regimes) / max(1, len(all_regimes))),
        "verbalization_nonempty_rate": (nonempty_verbalizations / max(1, total_verbalizations)),
        "per_run": runs,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {len(prompts)} traces + summary.json to {args.out_dir}")
    print(f"  div mean={summary['divergence']['mean']:.2f}"
          f"  regime[1] frac={summary['regime_1_frac']:.2f}"
          f"  verbalization nonempty={summary['verbalization_nonempty_rate']:.2%}")


if __name__ == "__main__":
    main()
