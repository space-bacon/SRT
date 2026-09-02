"""Does the format effect track baseline disorder WITHIN one benchmark?

`paper_format_susceptibility.md` finds corr(baseline order, format gain) = -0.870
across two benchmarks, and reads the template as a symmetry-breaking field whose
effect is largest where the system is least committed. Two benchmarks are two
points, so the relation could be a between-domain artifact.

This tests it per problem, with model, family, tokenizer and budget all fixed.

Two traps, both handled:

Naive `gain = chat - raw` is mechanically anti-correlated with `raw`, so
measurement noise alone manufactures a negative correlation. Every problem-level
number here is split-half: the eight raw samples are cut into halves A and B,
half A supplies the stratifying variable and half B supplies the outcome, so the
noise in one is independent of the noise in the other. The naive number is
printed alongside purely to show how much of it is that artifact.

Raw cosine on an anisotropic encoder is not interpretable, so every similarity is
reported against a different-problem floor computed through the same path.

    python scripts/format_susceptibility_perproblem.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
import torch

SCORER = "sentence-transformers/all-MiniLM-L6-v2"
SIZES = ["0.5B", "1.5B", "3B", "7B", "14B", "32B"]


def embed(texts, batch=256):
    from transformers import AutoModel, AutoTokenizer
    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(SCORER)
    mod = AutoModel.from_pretrained(SCORER).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            b = tok(texts[i:i + batch], padding=True, truncation=True,
                    max_length=256, return_tensors="pt").to(dev)
            h = mod(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu())
    X = torch.cat(out).double()
    return torch.nn.functional.normalize(X, dim=1).numpy()


def intra(E, idx):
    """Mean pairwise cosine within the given sample columns, per problem."""
    iu, ju = zip(*itertools.combinations(idx, 2))
    return (E[:, iu, :] * E[:, ju, :]).sum(-1).mean(1)


def floor_of(E, rng, n=20000):
    """Different-problem pairs through the identical path."""
    n_prob, k, _ = E.shape
    F = E.reshape(-1, E.shape[-1])
    a = rng.integers(0, len(F), n)
    b = rng.integers(0, len(F), n)
    keep = (a // k) != (b // k)
    return float((F[a[keep]] * F[b[keep]]).sum(-1).mean())


def load(gen_dir, tag, arm, n_prob, k):
    p = os.path.join(gen_dir, f"{tag}__{arm}.json")
    rows = json.load(open(p))[:n_prob]
    return [c[:k] for c in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-dir", default="artifacts/nla/coder_matrix1024")
    ap.add_argument("--prefix", default="coder")
    ap.add_argument("--sizes", nargs="*", default=SIZES)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--out", default="artifacts/nla/format_susceptibility_perproblem.json")
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    n_prob = len(json.load(open(os.path.join(a.gen_dir, f"{a.prefix}0.5B_inst__chat.json"))))
    half_a, half_b = list(range(a.k // 2)), list(range(a.k // 2, a.k))

    per_size, pooled = {}, {"raw_a": [], "raw_b": [], "chat": [], "size": []}
    for s in a.sizes:
        tag = f"{a.prefix}{s}_inst"
        raw = load(a.gen_dir, tag, "raw", n_prob, a.k)
        cht = load(a.gen_dir, tag, "chat", n_prob, a.k)
        E = embed([c for r in raw for c in r]).reshape(n_prob, a.k, -1)
        C = embed([c for r in cht for c in r]).reshape(n_prob, a.k, -1)

        ra, rb = intra(E, half_a), intra(E, half_b)
        ch = intra(C, list(range(a.k)))
        fl_r, fl_c = floor_of(E, rng), floor_of(C, rng)

        # Independent halves, so this correlation cannot be a subtraction artifact.
        r_clean = float(np.corrcoef(ra, ch - rb)[0, 1])
        r_naive = float(np.corrcoef(rb, ch - rb)[0, 1])
        r_direct = float(np.corrcoef(ra, ch)[0, 1])

        per_size[s] = {
            "n_problems": int(n_prob),
            "raw_intra": float(rb.mean()), "chat_intra": float(ch.mean()),
            "raw_floor": fl_r, "chat_floor": fl_c,
            "raw_above_floor": float(rb.mean() - fl_r),
            "chat_above_floor": float(ch.mean() - fl_c),
            "r_splithalf": r_clean, "r_naive_inflated": r_naive,
            "r_direct_raw_vs_chat": r_direct,
        }
        pooled["raw_a"] += list(ra); pooled["raw_b"] += list(rb)
        pooled["chat"] += list(ch); pooled["size"] += [s] * n_prob
        print(f"{s:6s} raw {rb.mean():.4f} (floor {fl_r:.4f})  "
              f"chat {ch.mean():.4f} (floor {fl_c:.4f})  "
              f"r_splithalf {r_clean:+.3f}  r_naive {r_naive:+.3f}", flush=True)

    ra = np.array(pooled["raw_a"]); rb = np.array(pooled["raw_b"])
    ch = np.array(pooled["chat"])
    gain = ch - rb
    r_clean = float(np.corrcoef(ra, gain)[0, 1])
    r_naive = float(np.corrcoef(rb, gain)[0, 1])

    print(f"\npooled over {len(ra)} problem-model pairs")
    print(f"  split-half  corr(raw_A, chat - raw_B) = {r_clean:+.3f}   <- the honest one")
    print(f"  naive       corr(raw_B, chat - raw_B) = {r_naive:+.3f}   <- inflated by shared noise")
    print(f"  direct      corr(raw_A, chat)         = {float(np.corrcoef(ra, ch)[0, 1]):+.3f}")

    # A field that saturates should show the gain flattening in the ordered bins,
    # not falling linearly, so bin rather than trusting the correlation alone.
    order = np.argsort(ra)
    edges = np.array_split(order, a.bins)
    print(f"\nstratified by half-A raw order, {a.bins} equal bins")
    print(f"  {'bin':>4s} {'n':>6s} {'raw_A':>8s} {'raw_B':>8s} {'chat':>8s} {'gain':>8s}")
    strata = []
    for i, ix in enumerate(edges):
        row = {"bin": i, "n": len(ix), "raw_a": float(ra[ix].mean()),
               "raw_b": float(rb[ix].mean()), "chat": float(ch[ix].mean()),
               "gain": float(gain[ix].mean())}
        strata.append(row)
        print(f"  {i:>4d} {len(ix):>6d} {row['raw_a']:8.4f} {row['raw_b']:8.4f} "
              f"{row['chat']:8.4f} {row['gain']:+8.4f}")

    lo, hi = strata[0]["gain"], strata[-1]["gain"]
    print(f"\n  most-disordered bin gains {lo:+.4f}, most-ordered bin gains {hi:+.4f}, "
          f"difference {lo - hi:+.4f}")

    out = {"gen_dir": a.gen_dir, "k": a.k, "n_problems": int(n_prob),
           "per_size": per_size, "strata": strata,
           "pooled": {"r_splithalf": r_clean, "r_naive_inflated": r_naive,
                      "n_pairs": len(ra)}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
