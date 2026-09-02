"""Is the field response a moving mean, or two branches with a moving mixture?

`paper_format_susceptibility.md` finds order rising steeply to alpha ~= 0.4 and
then flat, with no dead zone at low field. Read as a mean, that is a saturating
field and not a bifurcation.

But a pitchfork does not predict a shifted mean. It predicts two branches
separated by a separatrix, with the mixture weight moving as the control rises.
A population splitting between a low branch and a high branch produces a
perfectly smooth mean. So "smooth in the mean" does not settle it, and the papers
that read best-of-K as a pitchfork (paper_nla.md section 12) are making a claim
about branch structure that a mean cannot test.

This looks at the distribution of per-problem order at each alpha and asks
whether it is bimodal, using three independent readings:

  Hartigan-style dip     a nonparametric test for departure from unimodality
  bimodality coefficient moment-based, BC > 0.555 is the usual flag
  GMM BIC                does a two-component fit beat one component

A pitchfork wants bimodality near the knee, unimodality at both ends, and a
mixture weight that moves monotonically with alpha. A single mode sliding
rightward is a saturating field and closes the question.

    python scripts/format_field_bimodality.py
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os

import numpy as np
import torch

SCORER = "sentence-transformers/all-MiniLM-L6-v2"


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


def dip(x):
    """Hartigan's dip: sup distance from the ECDF to its closest unimodal fit.

    Approximated by the gap between the ECDF and the greatest convex minorant /
    least concave majorant, which is what the exact statistic measures.
    """
    x = np.sort(np.asarray(x, float))
    n = len(x)
    ecdf = np.arange(1, n + 1) / n
    lo = np.maximum.accumulate(ecdf - np.linspace(0, 1, n))
    hi = np.minimum.accumulate((ecdf - np.linspace(0, 1, n))[::-1])[::-1]
    return float(np.max(np.abs(lo - hi)) / 2)


def bimodality_coefficient(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    s = x.std(ddof=1)
    g = ((x - m) ** 3).mean() / s ** 3
    k = ((x - m) ** 4).mean() / s ** 4 - 3.0
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def gmm_bic(x, seed=0):
    """BIC for a one- vs two-component Gaussian mixture. Negative favours two."""
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return None, None, None
    X = np.asarray(x, float).reshape(-1, 1)
    g1 = GaussianMixture(1, random_state=seed).fit(X)
    g2 = GaussianMixture(2, random_state=seed, n_init=5).fit(X)
    w = float(np.sort(g2.weights_)[0])
    return float(g1.bic(X)), float(g2.bic(X)), w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-dir", default="artifacts/nla/field_sweep")
    ap.add_argument("--pattern", default="coder3B_a*__chat.json")
    ap.add_argument("--out", default="artifacts/nla/field_sweep/bimodality.json")
    a = ap.parse_args()

    rows_out = []
    print("Per-problem order at each field strength: one branch or two?")
    print(f"{'alpha':>6s} {'mean':>7s} {'sd':>6s} {'dip':>7s} {'BC':>6s} "
          f"{'BIC1-BIC2':>10s} {'minor w':>8s}  reading")
    for f in sorted(glob.glob(os.path.join(a.gen_dir, a.pattern))):
        al = float(os.path.basename(f).split("_a")[1].split("__")[0])
        rows = json.load(open(f))
        n, k = len(rows), len(rows[0])
        E = embed([c for r in rows for c in r]).reshape(n, k, -1)
        iu, ju = zip(*itertools.combinations(range(k), 2))
        per = (E[:, iu, :] * E[:, ju, :]).sum(-1).mean(1)

        d = dip(per)
        bc = bimodality_coefficient(per)
        b1, b2, w = gmm_bic(per)
        delta = (b1 - b2) if b1 is not None else float("nan")
        flag = "TWO" if (bc > 0.555 and delta > 10) else "one"
        rows_out.append({"alpha": al, "mean": float(per.mean()),
                         "sd": float(per.std()), "dip": d, "bc": bc,
                         "bic1_minus_bic2": delta, "minor_weight": w,
                         "reading": flag})
        print(f"{al:6.2f} {per.mean():7.4f} {per.std():6.4f} {d:7.4f} {bc:6.3f} "
              f"{delta:10.1f} {w if w is not None else float('nan'):8.3f}  {flag}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows_out, open(a.out, "w"), indent=1)

    two = [r for r in rows_out if r["reading"] == "TWO"]
    print()
    print(f"  bimodal at {len(two)}/{len(rows_out)} field strengths")
    if two:
        print("  alphas flagged:", ", ".join(f"{r['alpha']:.2f}" for r in two))
        ws = [r["minor_weight"] for r in two if r["minor_weight"] is not None]
        if len(ws) > 1:
            print(f"  minority weight across those: {min(ws):.3f} to {max(ws):.3f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
