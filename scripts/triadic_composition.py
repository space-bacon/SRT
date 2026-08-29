"""Do interpretants compose? The direct test of whether the dyadic reduction holds.

Retention, as we have been computing it, divides the vendor out. cross/within
treats the encoder as a nuisance to normalise away, which is exactly the move
paper.md section 2.7 says is not available: the third term is constitutive, not
ornamental. Worse, xvendor_fit_n builds a single shared d-dim space for all
vendors before it measures anything, so the estimator presupposes the common
code whose existence is the open question. Damaging that space and watching both
terms fall together is not evidence about sharing. It is evidence that we forced
a shared space and then broke it.

This asks the question without a shared space anywhere. Fit a map directly
between two vendors' image states, A -> C, and score it by whether vendor A's
reading of a picture lands on vendor C's reading of the same picture. Then route
the same journey through a third vendor, A -> B -> C, and compare.

  composed ~= direct              a single global code exists, every vendor is a
                                  lossy view of it, the dyadic reduction is fine
                                  and retention was measuring something real

  composed << direct              each pair shares something the others do not
                                  carry. relations are pair-specific, and a
                                  common-space estimator cannot see it

  spread across intermediates     if the choice of B matters, B is participating
                                  in the relation rather than passing it through

The last one is the cleanest signal because it needs no baseline: under a global
code every intermediate is equally good, so any systematic spread across B is
thirdness showing up in a quantity that has no dyadic definition.

Everything is centred on the train mean before scoring, and raw anisotropy is
printed per vendor, because cosine between uncentred hidden states is not
interpretable and a dominant direction would manufacture agreement here.

  python scripts/triadic_composition.py --vendor tag:states_glob:manifest_glob ...
"""
import argparse
import itertools
import json

import numpy as np
import torch

from xvendor_fit_n import load_vendor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vendor", action="append", required=True,
                   help="tag:states_glob:manifest_glob, repeatable")
    p.add_argument("--holdout-frac", type=float, default=0.1)
    p.add_argument("--ridge", type=float, default=1e-2,
                   help="ridge as a fraction of the mean Gram eigenvalue")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="/root/triadic_composition.json")
    return p.parse_args()


def fit(X, Y, alpha):
    """Ridge map from one vendor's space to another, train rows only."""
    G = X.T @ X
    lam = alpha * torch.diagonal(G).mean()
    return torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=G.device,
                                                  dtype=G.dtype), X.T @ Y)


def r_at_1(P, T):
    """Does each mapped row retrieve the same item in the target vendor's space?"""
    P = torch.nn.functional.normalize(P, dim=1)
    T = torch.nn.functional.normalize(T, dim=1)
    S = P @ T.T
    return float((S.argmax(1) == torch.arange(len(P), device=S.device)).float().mean())


def anisotropy(M):
    N = torch.nn.functional.normalize(M, dim=1)
    S = N @ N.T
    n = len(N)
    return float((S.sum() - torch.diagonal(S).sum()) / (n * (n - 1)))


def main():
    a = parse_args()
    print("loading vendors:")
    V = [load_vendor(s) for s in a.vendor]
    tags = [v["tag"] for v in V]

    shared = set(V[0]["key"])
    for v in V[1:]:
        shared &= set(v["key"])
    order = [k for k in V[0]["key"] if k in shared]
    n = len(order)
    print(f"\naligned on {n} keys shared by all {len(V)} vendors")

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(n)
    n_te = int(a.holdout_frac * n)
    te, tr = perm[:n_te], perm[n_te:]
    tr_t = torch.tensor(tr).cuda()
    te_t = torch.tensor(te).cuda()

    X = {}
    aniso = {}
    print()
    for v in V:
        pos = {k: i for i, k in enumerate(v["key"])}
        idx = np.array([pos[k] for k in order])
        M = torch.tensor(v["item"][idx]).cuda().double()
        raw = anisotropy(M[te_t])
        M = M - M[tr_t].mean(0, keepdim=True)
        cen = anisotropy(M[te_t])
        X[v["tag"]] = M
        aniso[v["tag"]] = {"raw": raw, "centred": cen, "d": int(M.shape[1])}
        print(f"  {v['tag']:<10} d={M.shape[1]:5d}  anisotropy raw {raw:6.3f} "
              f"-> centred {cen:6.3f}")

    res = {"question": "do interpretants compose, or is each pairing constitutive",
           "vendors": tags, "n_aligned": n, "n_train": len(tr), "n_holdout": n_te,
           "centred": True, "anisotropy": aniso, "direct": {}, "composed": {},
           "self_hop": {}, "floor": {}}

    print(f"\nfitting {len(tags) * (len(tags) - 1)} direct maps, "
          f"train {len(tr)} rows, holdout {n_te}")
    M = {}
    for A, C in itertools.permutations(tags, 2):
        M[(A, C)] = fit(X[A][tr_t], X[C][tr_t], a.ridge)
        res["direct"][f"{A}->{C}"] = r_at_1(X[A][te_t] @ M[(A, C)], X[C][te_t])
    for A in tags:
        M[(A, A)] = fit(X[A][tr_t], X[A][tr_t], a.ridge)

    g = torch.Generator(device="cuda").manual_seed(99)
    for A, C in itertools.permutations(tags, 2):
        sh = torch.randperm(n_te, generator=g, device="cuda")
        res["floor"][f"{A}->{C}"] = r_at_1(X[A][te_t][sh] @ M[(A, C)], X[C][te_t])

    print(f"\n{'A -> C':<26} {'direct':>8s} {'via self':>9s}  routed through each "
          f"third vendor")
    rows = []
    for A, C in itertools.permutations(tags, 2):
        direct = res["direct"][f"{A}->{C}"]
        selfhop = r_at_1(X[A][te_t] @ M[(A, A)] @ M[(A, C)], X[C][te_t])
        res["self_hop"][f"{A}->{C}"] = selfhop
        vias = {}
        for B in tags:
            if B in (A, C):
                continue
            vias[B] = r_at_1(X[A][te_t] @ M[(A, B)] @ M[(B, C)], X[C][te_t])
            res["composed"][f"{A}->{B}->{C}"] = vias[B]
        spread = max(vias.values()) - min(vias.values())
        rows.append({"pair": f"{A}->{C}", "direct": direct, "self_hop": selfhop,
                     "via": vias, "best_via": max(vias.values()), "spread": spread})
        via_s = "  ".join(f"{B[:7]} {v:.4f}" for B, v in vias.items())
        print(f"{A} -> {C:<14} {direct:8.4f} {selfhop:9.4f}  {via_s}")

    dr = [r["direct"] for r in rows]
    sh = [r["self_hop"] for r in rows]
    bv = [r["best_via"] for r in rows]
    sp = [r["spread"] for r in rows]
    fl = list(res["floor"].values())
    summ = {"mean_direct": float(np.mean(dr)),
            "mean_self_hop": float(np.mean(sh)),
            "mean_best_via": float(np.mean(bv)),
            "mean_spread_across_intermediates": float(np.mean(sp)),
            "max_spread_across_intermediates": float(np.max(sp)),
            "mean_shuffled_floor": float(np.mean(fl)),
            "hop_cost_self": float(np.mean(dr) - np.mean(sh)),
            "hop_cost_third_vendor": float(np.mean(dr) - np.mean(bv))}
    res["summary"] = summ

    print(f"\nmean direct                        {summ['mean_direct']:.4f}")
    print(f"mean routed through self (control) {summ['mean_self_hop']:.4f}   "
          f"cost of one extra fitted hop {summ['hop_cost_self']:+.4f}")
    print(f"mean routed through a third vendor {summ['mean_best_via']:.4f}   "
          f"cost beyond that                {summ['hop_cost_third_vendor']:+.4f}")
    print(f"mean spread across intermediates   "
          f"{summ['mean_spread_across_intermediates']:.4f}")
    print(f"shuffled floor                     {summ['mean_shuffled_floor']:.4f}")

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
