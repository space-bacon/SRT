#!/usr/bin/env python3
"""Is the edge-count ladder about the count, or about which edges the route uses?

Dipankar Sarkar's objection to the ladder in the cross-vendor paper: the rungs do
not hold edge identity fixed. The 1-edge rung uses only the qwen3omni/gemma4
boundary, the 2-edge rung adds mistral, the 3-edge rung adds aria. Our own
anisotropy numbers say the vendors are not interchangeable, so the boundaries are
not either. A route crossing three distinct boundaries is more likely to contain
the worst boundary than a route crossing one, and that alone produces a monotone
ladder with no route-level effect in it.

Two tests, run at matched hop counts.

1. SPREAD. Measure every one of the six boundaries on its own, as a there-and-back
   at the same hop counts. If the spread across single boundaries covers the
   ladder's range, edge identity is sufficient on its own.

2. MULTIPLICATIVITY, the test he proposed. Under composition bookkeeping with no
   route-level term, a route's retention is the product of the retentions of the
   boundaries it uses, each measured at the number of round trips that route
   actually gives it. Measured at the product means nothing lives above the
   edges. Measured below the product means the count is doing work.

   At h hops the 2-edge route repeats h/4 times, so each of its two boundaries
   gets h/4 round trips, which is a single-boundary route of h/2 hops. The 3-edge
   route repeats h/6 times, so each of its three boundaries gets h/3 hops.

Same fit, centering, ridge and holdout as holonomy_palindrome.py, so the numbers
line up with the published ladder rather than merely resembling it.

    python scripts/holonomy_edge_identity.py --domain roco
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAGS = ["qwen3omni", "gemma4", "mistral", "aria"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="artifacts/nla/omni")
    p.add_argument("--domain", default="roco")
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--hops", type=int, nargs="*", default=[12, 24, 36])
    p.add_argument("--out", default=None)
    return p.parse_args()


def fit(X, Y, alpha):
    G = X.T @ X
    lam = alpha * torch.diagonal(G).mean()
    return torch.linalg.solve(
        G + lam * torch.eye(G.shape[0], device=G.device, dtype=G.dtype), X.T @ Y)


def returns(P, T):
    Pn = torch.nn.functional.normalize(P, dim=1)
    Tn = torch.nn.functional.normalize(T, dim=1)
    S = Pn @ Tn.T
    d = torch.arange(len(P), device=S.device)
    return float((S.argmax(1) == d).float().mean()), float(S[d, d].mean())


def walk(X, M, te, route, hops):
    if hops % (len(route) - 1):
        raise SystemExit(f"{hops} hops does not divide route {route}")
    P = X[route[0]][te]
    for _ in range(hops // (len(route) - 1)):
        for i in range(len(route) - 1):
            P = P @ M[(route[i], route[i + 1])]
    return returns(P, X[route[0]][te])


def load(a):
    rows = json.load(open(os.path.join(a.dir, f"{a.domain}_manifest.json")))["rows"]
    V = {}
    for t in TAGS:
        z = np.load(os.path.join(a.dir, "states", f"{a.domain}_{t}.npz"))
        ok = z["ok"]
        V[t] = {"img": z["item"][ok],
                "key": np.array([r["key"] for r, k in zip(rows, ok) if k])}
    shared = set(V[TAGS[0]]["key"])
    for t in TAGS[1:]:
        shared &= set(V[t]["key"])
    order = [k for k in V[TAGS[0]]["key"] if k in shared]
    n = len(order)
    perm = np.random.default_rng(0).permutation(n)
    n_te = int(a.holdout_frac * n)
    te = torch.tensor(perm[:n_te]).to(DEV)
    tr = torch.tensor(perm[n_te:]).to(DEV)
    X = {}
    for t in TAGS:
        pos = {k: i for i, k in enumerate(V[t]["key"])}
        M0 = torch.tensor(V[t]["img"][np.array([pos[k] for k in order])]).to(DEV).double()
        X[t] = M0 - M0[tr].mean(0, keepdim=True)
    M = {(x, y): fit(X[x][tr], X[y][tr], a.ridge) for x in TAGS for y in TAGS}
    return X, M, te, n, n_te


def main():
    a = parse_args()
    a.out = a.out or os.path.join(a.dir, f"holonomy_edge_identity_{a.domain}.json")
    X, M, te, n, n_te = load(a)
    A, B, C, D = TAGS
    print(f"device {DEV}   domain {a.domain}   aligned {n}   holdout {n_te}")

    res = {"question": "does the edge-count ladder survive holding edge identity fixed",
           "credit": "objection and multiplicativity test proposed by Dipankar Sarkar",
           "domain": a.domain, "vendors": TAGS, "n_holdout": n_te}

    # --- 1. every boundary on its own -------------------------------------
    pairs = list(itertools.combinations(TAGS, 2))
    need = sorted({h for h in a.hops} | {h // 2 for h in a.hops} | {h // 3 for h in a.hops})
    need = [h for h in need if h % 2 == 0]
    single = {}
    print(f"\nsingle boundary, there-and-back{'':<10}" + "".join(f"{h:>9d}" for h in need))
    for x, y in pairs:
        row = {}
        for h in need:
            r, c = walk(X, M, te, [x, y, x], h)
            row[h] = {"r@1": round(r, 4), "cos": round(c, 4)}
        single[f"{x}|{y}"] = row
        print(f"  {x} <-> {y:<22}" + "".join(f"{row[h]['r@1']:9.4f}" for h in need))
    res["single_boundary"] = single

    at36 = {k: v[36]["r@1"] for k, v in single.items() if 36 in v}
    lo, hi = min(at36, key=at36.get), max(at36, key=at36.get)
    res["spread_at_36"] = {"best_edge": hi, "best": at36[hi],
                           "worst_edge": lo, "worst": at36[lo],
                           "range": round(at36[hi] - at36[lo], 4)}
    print(f"\n  spread across the six boundaries at 36 hops: "
          f"{at36[lo]:.4f} ({lo}) .. {at36[hi]:.4f} ({hi})   range {at36[hi]-at36[lo]:.4f}")

    # --- 2. the ladder, and the product that would explain it -------------
    ladder = {1: [A, B, A], 2: [A, B, C, B, A], 3: [A, B, C, D, C, B, A]}
    uses = {1: [(A, B)], 2: [(A, B), (B, C)], 3: [(A, B), (B, C), (C, D)]}
    print(f"\n{'route':<16}{'hops':>6}{'measured':>10}{'product':>10}{'meas-prod':>11}")
    lad = {}
    for k, route in ladder.items():
        for h in a.hops:
            r, c = walk(X, M, te, route, h)
            per = h // k          # hops each boundary gets in this route
            prod = 1.0
            ok = per % 2 == 0
            if ok:
                for x, y in uses[k]:
                    key = f"{x}|{y}" if f"{x}|{y}" in single else f"{y}|{x}"
                    prod *= single[key][per]["r@1"]
            lad[f"{k} edges, {h} hops"] = {
                "measured_r@1": round(r, 4), "cos": round(c, 4),
                "per_edge_hops": per,
                "product_r@1": round(prod, 4) if ok else None,
                "measured_minus_product": round(r - prod, 4) if ok else None}
            print(f"{k} distinct{'':<6}{h:>6}{r:>10.4f}"
                  + (f"{prod:>10.4f}{r - prod:>11.4f}" if ok else f"{'-':>10}{'-':>11}"))
    res["ladder_vs_product"] = lad

    # --- 3. the check with no model in it ---------------------------------
    worst = at36[lo]
    three = lad["3 edges, 36 hops"]["measured_r@1"]
    used = [f"{x}|{y}" if f"{x}|{y}" in single else f"{y}|{x}" for x, y in uses[3]]
    res["worst_single_vs_three_edges_at_36"] = {
        "worst_single_boundary": lo, "worst_single_r@1": worst,
        "three_edge_r@1": three, "delta": round(worst - three, 4),
        "boundaries_the_three_edge_route_uses": {k: at36[k] for k in used},
        "reading": ("a single boundary already at or below the 3-edge rung would mean "
                    "edge identity is sufficient and the count is not needed")}
    print(f"\nworst single boundary at 36 hops {worst:.4f} ({lo})"
          f"  vs 3-edge route {three:.4f}   delta {worst - three:+.4f}")
    print(f"  the 3-edge route uses: " + ", ".join(f"{k} {at36[k]:.4f}" for k in used))

    # --- 4. scramble edge identity, hold the count ------------------------
    # Every ordering of the three non-anchor vendors puts different boundaries in
    # the same three slots. If the ladder is monotone under all of them, which
    # edges a rung happens to use cannot be what orders the rungs.
    print(f"\nladder under every ordering of the non-anchor vendors, r@1 at 36 hops")
    print(f"{'B, C, D':<34}{'1 edge':>9}{'2 edges':>9}{'3 edges':>9}   monotone")
    perms = {}
    n_mono = 0
    for p in itertools.permutations([B, C, D]):
        b, c, d = p
        vals = [walk(X, M, te, r, 36)[0] for r in
                ([A, b, A], [A, b, c, b, A], [A, b, c, d, c, b, A])]
        mono = vals[0] > vals[1] > vals[2]
        n_mono += mono
        perms[", ".join(p)] = {"1": round(vals[0], 4), "2": round(vals[1], 4),
                               "3": round(vals[2], 4), "monotone": mono}
        print(f"{', '.join(p):<34}" + "".join(f"{v:9.4f}" for v in vals)
              + f"   {'yes' if mono else 'NO'}")
    res["ladder_under_vendor_permutations_36"] = perms
    res["monotone_permutations"] = f"{n_mono} of {len(perms)}"
    print(f"\nmonotone in {n_mono} of {len(perms)} orderings")

    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
