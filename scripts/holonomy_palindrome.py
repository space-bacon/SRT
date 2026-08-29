#!/usr/bin/env python3
"""Does the four-cycle lose more because it encloses area, or because it never retraces?

I read a three-way ordering as evidence about enclosed area: a self-loop held at
1.0000, a there-and-back at 0.7162, a four-cycle at 0.3830. Dipankar Sarkar's
objection is that there-and-back differs from the four-cycle in a second way I
did not control. A -> B -> A composes a map with its own approximate inverse, so
the errors cancel pairwise. The four-cycle never retraces a single edge.

His control, and it is the right one: A -> B -> C -> D -> C -> B -> A. All four
vendors, every edge traversed and then retraced, zero enclosed area.

  palindrome near the there-and-back number  -> I was measuring retracing
  palindrome near the four-cycle number      -> the area term is real

Hop counts are matched exactly rather than approximately. The three path types
have periods 2, 4 and 6, so they are compared at 12, 24 and 36 hops, where all
three divide evenly and no path gets credit for a shorter journey.

Runs on CPU in float64. These are large ridge solves composed dozens of times,
which is where float32 error would masquerade as the effect being measured.

  python scripts/holonomy_palindrome.py --domain roco
"""
import argparse
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
    """Carry the holdout along `route`, repeated until `hops` hops are used."""
    if hops % (len(route) - 1):
        raise SystemExit(f"{hops} hops does not divide route of length "
                         f"{len(route) - 1}")
    P = X[route[0]][te]
    for _ in range(hops // (len(route) - 1)):
        for i in range(len(route) - 1):
            P = P @ M[(route[i], route[i + 1])]
    return returns(P, X[route[0]][te])


def main():
    a = parse_args()
    a.out = a.out or os.path.join(a.dir, f"holonomy_palindrome_{a.domain}.json")
    rows = json.load(open(os.path.join(a.dir, f"{a.domain}_manifest.json")))["rows"]

    print(f"device {DEV}   domain {a.domain}")
    V = {}
    for t in TAGS:
        f = os.path.join(a.dir, "states", f"{a.domain}_{t}.npz")
        if not os.path.exists(f):
            print(f"  {t}: missing, skipping")
            continue
        z = np.load(f)
        ok = z["ok"]
        if len(rows) != len(ok):
            raise SystemExit(f"{t}: manifest {len(rows)} rows, states {len(ok)}")
        V[t] = {"img": z["item"][ok],
                "key": np.array([r["key"] for r, k in zip(rows, ok) if k])}
    tags = [t for t in TAGS if t in V]
    if len(tags) < 4:
        raise SystemExit(f"need four vendors for this control, have {tags}")

    shared = set(V[tags[0]]["key"])
    for t in tags[1:]:
        shared &= set(V[t]["key"])
    order = [k for k in V[tags[0]]["key"] if k in shared]
    n = len(order)
    perm = np.random.default_rng(0).permutation(n)
    n_te = int(a.holdout_frac * n)
    te = torch.tensor(perm[:n_te]).to(DEV)
    tr = torch.tensor(perm[n_te:]).to(DEV)
    print(f"  aligned {n}, train {n - n_te}, holdout {n_te}")

    X = {}
    for t in tags:
        pos = {k: i for i, k in enumerate(V[t]["key"])}
        idx = np.array([pos[k] for k in order])
        M0 = torch.tensor(V[t]["img"][idx]).to(DEV).double()
        X[t] = M0 - M0[tr].mean(0, keepdim=True)

    M = {}
    for x in tags:
        for y in tags:
            M[(x, y)] = fit(X[x][tr], X[y][tr], a.ridge)

    A, B, C, D = tags
    routes = {
        "self-loop, no boundary": [A, A],
        "there-and-back, 1 edge retraced": [A, B, A],
        "palindrome, 4 vendors, retraced, zero area": [A, B, C, D, C, B, A],
        "four-cycle, 4 vendors, never retraces": [A, B, C, D, A],
    }
    res = {"question": "is the four-cycle penalty about enclosed area or about "
                       "never retracing an edge",
           "credit": "control proposed by Dipankar Sarkar",
           "domain": a.domain, "vendors": tags, "n_holdout": n_te,
           "routes": {k: " -> ".join(v) for k, v in routes.items()},
           "by_hops": {}}

    print(f"\n{'route':<44}" + "".join(f"{h:>10d}" for h in a.hops))
    for name, route in routes.items():
        row, cells = {}, []
        for h in a.hops:
            r, c = walk(X, M, te, route, h)
            row[h] = {"r@1": round(r, 4), "cos": round(c, 4)}
            cells.append(f"{r:10.4f}")
        res["by_hops"][name] = row
        print(f"{name:<44}" + "".join(cells))

    pal = "palindrome, 4 vendors, retraced, zero area"
    tab = "there-and-back, 1 edge retraced"
    cyc = "four-cycle, 4 vendors, never retraces"
    print()
    verdicts = {}
    for h in a.hops:
        p, t_, c = (res["by_hops"][k][h]["r@1"] for k in (pal, tab, cyc))
        near = "retracing" if abs(p - t_) < abs(p - c) else "area"
        verdicts[h] = near
        print(f"  {h:2d} hops: palindrome {p:.4f} sits "
              f"{'closer to there-and-back ' + format(t_, '.4f') if near == 'retracing' else 'closer to the four-cycle ' + format(c, '.4f')}"
              f"  -> {near}")
    res["verdict_by_hops"] = verdicts
    res["verdict"] = max(set(verdicts.values()), key=list(verdicts.values()).count)
    res["reading"] = (
        "retracing means the three-way ordering reported earlier was partly an "
        "artifact of pairwise error cancellation and the enclosed-area reading "
        "should be withdrawn. area means the four-cycle penalty survives a "
        "route that retraces every edge and visits every vendor.")
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nverdict: {res['verdict']}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
