"""Score the pre-registered NLA best-of-K bimodality predictions N1 to N4.

Registered in paper_format_susceptibility.md (commit a44e57bc) before any
per-target data existed. The criteria below are copied from there and are not to
be adjusted after seeing the data.

  N1  per-target best-of-K centred fve is bimodal (Hartigan dip p < 0.05, and
      BC > 0.555, and GMM BIC1-BIC2 > 10) for at least two contiguous K in
      {2, 4, 8, 16, 32}
  N2  the two GMM means sit within 0.05 of 0.586 (greedy) and 0.799
      (paraphrase-best); not near 0.99
  N3  the weight in the upper mode increases monotonically with K
  N4  unimodal at K = 1

The confound N4 exists to catch: target heterogeneity (some targets are easy to
verbalise, some are not) gives bimodality already at K = 1 and a distribution that
merely translates rightward. Best-of-K max is monotone in K by construction, so a
rightward shift is not evidence of anything. Only splitting is.

    python scripts/nla_bimodality.py artifacts/nla/rerank_eval_X.pertarget.npz
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

LOW, HIGH, TOL = 0.586, 0.799, 0.05
KS = [1, 2, 4, 8, 16, 32, 64]


def bc(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); s = x.std(ddof=1)
    g = ((x - m) ** 3).mean() / s ** 3
    k = ((x - m) ** 4).mean() / s ** 4 - 3.0
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def gmm(x, seed=0):
    from sklearn.mixture import GaussianMixture
    X = np.asarray(x, float).reshape(-1, 1)
    g1 = GaussianMixture(1, random_state=seed).fit(X)
    g2 = GaussianMixture(2, random_state=seed, n_init=5).fit(X)
    order = np.argsort(g2.means_.ravel())
    means = g2.means_.ravel()[order]
    w = g2.weights_[order]
    return float(g1.bic(X) - g2.bic(X)), means, w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import diptest
    z = np.load(a.npz)
    S = z["sampled_cen"]                      # (M, K)
    M, Kmax = S.shape
    print(f"{M} targets, K up to {Kmax}\n")
    print(f"{'K':>3s} {'mean':>7s} {'sd':>7s} {'dip p':>7s} {'BC':>6s} {'BICgap':>8s} "
          f"{'lo mode':>8s} {'hi mode':>8s} {'hi w':>6s}  bimodal")

    rows = []
    for K in KS:
        if K > Kmax:
            break
        x = S[:, :K].max(axis=1)
        d, p = diptest.diptest(x)
        b = bc(x)
        gap, means, w = gmm(x)
        bim = bool(p < 0.05 and b > 0.555 and gap > 10)
        rows.append({"K": K, "mean": float(x.mean()), "sd": float(x.std()),
                     "dip_p": float(p), "bc": b, "bic_gap": gap,
                     "lo_mode": float(means[0]), "hi_mode": float(means[-1]),
                     "hi_weight": float(w[-1]), "bimodal": bim})
        print(f"{K:3d} {x.mean():7.4f} {x.std():7.4f} {p:7.3f} {b:6.3f} {gap:8.1f} "
              f"{means[0]:8.4f} {means[-1]:8.4f} {w[-1]:6.3f}  {'TWO' if bim else 'one'}")

    byk = {r["K"]: r for r in rows}
    bim_ks = [r["K"] for r in rows if r["bimodal"] and r["K"] in (2, 4, 8, 16, 32)]
    contiguous = any(k in bim_ks and 2 * k in bim_ks for k in bim_ks)
    n1 = contiguous
    n2 = any(abs(r["lo_mode"] - LOW) < TOL and abs(r["hi_mode"] - HIGH) < TOL
             for r in rows if r["bimodal"])
    wts = [r["hi_weight"] for r in rows if r["bimodal"]]
    n3 = len(wts) >= 2 and all(wts[i] <= wts[i + 1] for i in range(len(wts) - 1))
    n4 = 1 in byk and not byk[1]["bimodal"]

    print()
    print(f"N1 bimodal at >=2 contiguous K in {{2..32}}: {bim_ks}  -> {'PASS' if n1 else 'FAIL'}")
    print(f"N2 modes near {LOW} and {HIGH} (+/-{TOL})            -> {'PASS' if n2 else 'FAIL'}")
    print(f"N3 upper-mode weight monotone in K: {[round(w, 3) for w in wts]}  -> {'PASS' if n3 else 'FAIL'}")
    print(f"N4 unimodal at K=1 (dip p={byk.get(1, {}).get('dip_p', float('nan')):.3f})  -> {'PASS' if n4 else 'FAIL'}")
    if 1 in byk and byk[1]["bimodal"]:
        print("   K=1 is bimodal: this is target heterogeneity, not a transition.")

    if a.out:
        json.dump({"rows": rows, "N1": n1, "N2": n2, "N3": n3, "N4": n4,
                   "criteria": {"low": LOW, "high": HIGH, "tol": TOL}},
                  open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
