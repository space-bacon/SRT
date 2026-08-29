"""Transport a state around a closed cycle of interpretants and see if it comes home.

Every measurement in this line of work so far has applied a cut. Spectral
truncation cuts head from tail. The complement cuts the other way. Isotropic
noise cuts signal from noise. cross/within cuts the interpretant off. The
composition test cut a chain into hops and priced each one. All of them returned
the same answer, which is that the thing does not separate, and reporting that
seven times does not make it seven findings.

If semiosis is a chain in which each interpretant becomes the representamen of
the next sign, there is no slice to take. The only question that does not
presuppose a cut is whether the chain closes on itself.

So: compose the fitted maps around a closed loop of vendors, A to B to C to D
back to A, and ask whether a state arrives back where it started. Under a single
global code every path between two vendors is the same path, so transport around
any loop is the identity and the loop is worth nothing. Any reproducible
departure from identity belongs to the cycle and to nothing smaller. It has no
two-place definition, which is what makes it the quantity worth having.

Ridge maps shrink, so a loop of length four decays even with no curvature at
all. The control is a self-loop of the same length, A to A four times, which
carries the same shrinkage and crosses no vendor boundary. Holonomy is the gap
between those two, not the raw return.

Then iterate. Unlimited semiosis is the loop applied again and again, so the map
is raised to successive powers and the return is tracked. A chain that closes
holds. A chain that drifts walks away at a rate the powers expose.

  python scripts/semiosis_holonomy.py --vendor tag:states:manifest ...
"""
import argparse
import itertools
import json

import numpy as np
import torch

from xvendor_fit_n import load_vendor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vendor", action="append", required=True)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--powers", type=int, default=8,
                   help="how many times to go round, to expose the attractor")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="/root/semiosis_holonomy.json")
    return p.parse_args()


def fit(X, Y, alpha):
    G = X.T @ X
    lam = alpha * torch.diagonal(G).mean()
    return torch.linalg.solve(
        G + lam * torch.eye(G.shape[0], device=G.device, dtype=G.dtype), X.T @ Y)


def returns(P, T):
    """Did each transported row come back to itself, and how far did it turn?"""
    Pn = torch.nn.functional.normalize(P, dim=1)
    Tn = torch.nn.functional.normalize(T, dim=1)
    S = Pn @ Tn.T
    d = torch.arange(len(P), device=S.device)
    return (float((S.argmax(1) == d).float().mean()),
            float(S[d, d].mean()),
            float((P.norm(dim=1) / T.norm(dim=1).clamp_min(1e-9)).median()))


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
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(n)
    n_te = int(a.holdout_frac * n)
    te = torch.tensor(perm[:n_te]).cuda()
    tr = torch.tensor(perm[n_te:]).cuda()
    print(f"\naligned on {n} keys, train {n - n_te}, holdout {n_te}")

    X = {}
    for v in V:
        pos = {k: i for i, k in enumerate(v["key"])}
        idx = np.array([pos[k] for k in order])
        M = torch.tensor(v["item"][idx]).cuda().double()
        X[v["tag"]] = M - M[tr].mean(0, keepdim=True)

    M = {}
    for A, B in itertools.product(tags, repeat=2):
        M[(A, B)] = fit(X[A][tr], X[B][tr], a.ridge)

    res = {"question": "does a state transported around a closed cycle of "
                       "interpretants come home",
           "vendors": tags, "n_aligned": n, "n_holdout": n_te,
           "control": "self-loop of equal length, same shrinkage, no boundary "
                      "crossed. holonomy is the gap, not the raw return",
           "cycles": {}, "self_loops": {}, "iterated": {}}

    # A self-loop of each length is the shrinkage-only baseline.
    for L in (3, 4):
        for A in tags:
            P = X[A][te]
            for _ in range(L):
                P = P @ M[(A, A)]
            r, c, nr = returns(P, X[A][te])
            res["self_loops"][f"{A}x{L}"] = {"r@1": r, "cos": c, "norm_ratio": nr}
    for L in (3, 4):
        v = [res["self_loops"][f"{A}x{L}"] for A in tags]
        print(f"self-loop length {L}: mean return r@1 "
              f"{np.mean([x['r@1'] for x in v]):.4f}  "
              f"cos {np.mean([x['cos'] for x in v]):.4f}")

    print(f"\n{'cycle':<42} {'return r@1':>10s} {'cos':>7s} {'vs self-loop':>13s}")
    rows = []
    for L in (3, 4):
        base = float(np.mean([res["self_loops"][f"{A}x{L}"]["r@1"] for A in tags]))
        for rest in itertools.permutations(tags[1:], L - 1):
            for start in ([tags[0]] if L == len(tags) else tags):
                cyc = (start,) + tuple(t for t in rest if t != start)
                if len(cyc) != L or len(set(cyc)) != L:
                    continue
                P = X[cyc[0]][te]
                for i in range(L):
                    P = P @ M[(cyc[i], cyc[(i + 1) % L])]
                r, c, nr = returns(P, X[cyc[0]][te])
                name = " -> ".join(cyc) + f" -> {cyc[0]}"
                if name in res["cycles"]:
                    continue
                res["cycles"][name] = {"r@1": r, "cos": c, "norm_ratio": nr,
                                       "length": L, "self_loop_r@1": base,
                                       "holonomy_gap": base - r}
                rows.append((name, r, c, base - r))
                print(f"{name:<42} {r:10.4f} {c:7.4f} {base - r:+13.4f}")

    gaps = [g for _, _, _, g in rows]
    print(f"\nholonomy gap across {len(rows)} cycles: mean {np.mean(gaps):+.4f}  "
          f"sd {np.std(gaps):.4f}  min {np.min(gaps):+.4f}  max {np.max(gaps):+.4f}")

    # Unlimited semiosis: go round again and again on the longest cycle.
    cyc = tuple(tags)
    W = torch.eye(X[cyc[0]].shape[1], device="cuda", dtype=torch.float64)
    for i in range(len(cyc)):
        W = W @ M[(cyc[i], cyc[(i + 1) % len(cyc)])]
    Ws = torch.eye(W.shape[0], device="cuda", dtype=torch.float64)
    Ws_self = torch.eye(W.shape[0], device="cuda", dtype=torch.float64)
    Wself = M[(cyc[0], cyc[0])]
    for _ in range(len(cyc)):
        Ws_self = Ws_self @ Wself
    print(f"\ngoing round {' -> '.join(cyc)} repeatedly:")
    print(f"{'laps':>5s} {'return r@1':>10s} {'cos':>7s} {'norm':>7s}   "
          f"{'self-loop r@1':>13s} {'cos':>7s}")
    S = Ws_self.clone()
    for k in range(1, a.powers + 1):
        Ws = Ws @ W
        S = S @ Ws_self
        r, c, nr = returns(X[cyc[0]][te] @ Ws, X[cyc[0]][te])
        rs, cs, _ = returns(X[cyc[0]][te] @ S, X[cyc[0]][te])
        res["iterated"][k] = {"r@1": r, "cos": c, "norm_ratio": nr,
                              "self_loop_r@1": rs, "self_loop_cos": cs}
        print(f"{k:5d} {r:10.4f} {c:7.4f} {nr:7.4f}   {rs:13.4f} {cs:7.4f}")

    tb = there_and_back(X, M, tags, te, len(cyc) * a.powers, res)
    cyc_final = res["iterated"][a.powers]
    res["summary"] = {
        "mean_holonomy_gap_one_lap": float(np.mean(gaps)),
        "sd_holonomy_gap_one_lap": float(np.std(gaps)),
        "max_holonomy_gap_one_lap": float(np.max(gaps)),
        "n_cycles": len(rows),
        "hops_compared": len(cyc) * a.powers,
        "cycle_r@1_at_depth": cyc_final["r@1"],
        "cycle_cos_at_depth": cyc_final["cos"],
        "self_loop_r@1_at_depth": cyc_final["self_loop_r@1"],
        "there_and_back_mean_r@1_at_depth": float(np.mean([v["r@1"] for v in tb.values()])),
        "there_and_back_mean_cos_at_depth": float(np.mean([v["cos"] for v in tb.values()])),
        "reading": "one lap cannot resolve this. the gap at a single pass is "
                   "inside its own spread, which is what every cut-based "
                   "measurement was reporting. the same maps iterated show "
                   "whether the chain closes or spirals, and the controls say "
                   "whether a spiral is fitting error or path."}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {a.out}")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {a.out}")


def there_and_back(X, M, tags, te, hops, res):
    """Cross a boundary and come straight back, which is a loop with no area.

    A four-cycle that decays under iteration could be nothing more than
    cross-vendor fits being harder than self-fits, with the error compounding.
    The there-and-back loop crosses the same boundaries and pays the same
    per-hop fitting cost, but retraces its own edge instead of enclosing a
    region. If both decay alike at equal hop count, the effect is fitting error.
    If only the cycle decays, the departure belongs to the path.
    """
    print(f"\nthere-and-back controls, compared at equal hop count:")
    print(f"{'loop':<28} {'hops':>5s} {'return r@1':>10s} {'cos':>7s}")
    out = {}
    for A, B in itertools.permutations(tags, 2):
        W = M[(A, B)] @ M[(B, A)]
        P, acc = X[A][te], torch.eye(W.shape[0], device="cuda", dtype=torch.float64)
        for _ in range(hops // 2):
            acc = acc @ W
        r, c, _ = returns(P @ acc, X[A][te])
        out[f"{A} -> {B} -> {A}"] = {"hops": hops, "r@1": r, "cos": c}
        print(f"{A + ' -> ' + B + ' -> ' + A:<28} {hops:5d} {r:10.4f} {c:7.4f}")
    res["there_and_back"] = out
    return out


if __name__ == "__main__":
    main()
