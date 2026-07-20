"""Base-vs-instruction-tuned K-curve figure (for Erik / paper §11.7).

Publication-style (white bg), not the marketing brand: this is a figure
for a peer scientist. Plots best-of-K centered-fve for gemma-4-31B base
vs -it against the shared anchor frame, with fitted log-linear slopes.
The point is that the two lineages are indistinguishable, which is the
within-family control that refutes the base-vs-tuned conjecture.

    python scripts/plot_base_vs_it_kcurve.py \
        --base artifacts/nla/gemma4_base/kcurve_ce.jsonl \
        --it   artifacts/nla/gemma4/kcurve_ce.jsonl \
        --out  artifacts/nla/figures/base_vs_it_kcurve.png
"""
from __future__ import annotations

import argparse
import json
import math
import random


def curve(path: str, seed: int = 0) -> dict[int, float]:
    rows = [json.loads(l) for l in open(path)]
    random.seed(seed)
    out = {}
    for K in (1, 2, 4, 8, 16, 32):
        tot = 0.0
        for r in rows:
            s = [c["cen"] for c in r["sampled"]]
            if K >= len(s):
                tot += max(s)
                continue
            tot += sum(max(random.sample(s, K)) for _ in range(200)) / 200
        out[K] = tot / len(rows)
    return out


def slope(c: dict[int, float]) -> float:
    return (c[32] - c[2]) / (math.log2(32) - math.log2(2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="artifacts/nla/gemma4_base/kcurve_ce.jsonl")
    ap.add_argument("--it", default="artifacts/nla/gemma4/kcurve_ce.jsonl")
    ap.add_argument("--floor", type=float, default=0.502)
    ap.add_argument("--nn", type=float, default=0.674)
    ap.add_argument("--out", default="artifacts/nla/figures/base_vs_it_kcurve.png")
    args = ap.parse_args()

    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = curve(args.base)
    it = curve(args.it)
    Ks = [1, 2, 4, 8, 16, 32]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
        "font.size": 12,
    })
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=200)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # reference bands
    ax.axhline(args.nn, ls="--", lw=1.2, color="#555",
               label=f"nearest-neighbour retrieval ({args.nn:.2f})")
    ax.axhline(args.floor, ls=":", lw=1.2, color="#999",
               label=f"random-text floor ({args.floor:.2f})")

    ax.plot(Ks, [b[k] for k in Ks], "o-", color="#7A5CFF", lw=2.2, ms=6,
            label=f"gemma-4-31B  base   (slope {slope(b):+.3f}/oct)")
    ax.plot(Ks, [it[k] for k in Ks], "s-", color="#1F9E8A", lw=2.2, ms=6,
            label=f"gemma-4-31B  instruct (slope {slope(it):+.3f}/oct)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel("best-of-K samples (oracle rerank)")
    ax.set_ylabel("centered fidelity  ½(1+cos), L47")
    ax.set_ylim(0.48, 0.70)
    ax.set_title("Instruction tuning does not change verbalizability\n"
                 "within a model family (within-lineage control)",
                 fontsize=13, loc="left")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.grid(axis="y", ls="-", lw=0.4, color="#eee")
    fig.text(0.012, 0.01,
             "gemma-4-31B base vs instruct, matched layer / corpus / recipe; "
             "n=50 targets. Curves indistinguishable; base marginally lower. "
             "The steeper cross-family (Qwen) slope is confounded with "
             "architecture and scale, not tuning.",
             fontsize=7.2, color="#666", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)
    print(f"base  : " + "  ".join(f"K{k}={b[k]:.3f}" for k in Ks) + f"  slope {slope(b):+.4f}")
    print(f"it    : " + "  ".join(f"K{k}={it[k]:.3f}" for k in Ks) + f"  slope {slope(it):+.4f}")


if __name__ == "__main__":
    main()
