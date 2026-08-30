#!/usr/bin/env python3
"""One shared frame fitted from all four backbones at once, instead of twelve pairwise maps.

Everything cross-vendor so far has been bilateral: fit a ridge map from one
vendor to one other, twelve of them for four vendors. The satellite scene probe
then showed a probe transports between any pair at a cost of 0.0024, which is
the bilateral way of hinting that the four backbones might share a single frame
rather than twelve compatible private ones.

This asks that directly. Fit ONE frame from all four simultaneously (MAXVAR
generalised CCA), giving four encoders in and four decoders out, and compare
against the bilateral maps on the same holdout:

  via-joint near direct        a single shared frame is sufficient, and the
                               twelve pairwise maps were describing one object
  via-joint well below direct  each pair shares something the other two do not,
                               and there is no common frame

It also revisits the loop degradation, but with a control that the first version
of this script lacked. Routing every hop through the joint frame flattens the
edge-count ladder completely, to 1.0000 at thirty-six hops. That looked like the
shared frame buying back the degradation, and it is not. The composed via-joint
map collapses to rank 128 of 256, exactly the joint width, and then stays there:
once the route has projected onto the shared subspace, later hops cannot discard
dimensions that are already gone. The pairwise maps are full rank and compound
error across directions the joint route threw away in its first hop.

So the ladder is run three ways, and only the last two are comparable:

  pairwise, full rank    the original measurement
  pairwise, rank matched every map truncated to the joint width, so both routes
                         pass through the same size bottleneck
  via joint              the shared frame

If rank-matched pairwise also flattens, the flattening was always a rank effect
and the joint frame deserves no credit for it. If rank-matched pairwise still
degrades while the joint route holds, the shared frame is doing real work.

Consensus is the third question. If the frame is real, three vendors read
together should predict the fourth better than any one of them does alone.

CPU float64 throughout. These are maps composed up to thirty-six times, which is
exactly where float32 drift would impersonate the effect being measured.

    python scripts/joint_frame.py --domain rsicd
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

TAGS = ["qwen3omni", "gemma4", "mistral", "aria"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="artifacts/nla/omni")
    p.add_argument("--domain", default="rsicd")
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--pca", type=int, default=256, help="per-vendor dims before the joint fit")
    p.add_argument("--joint", type=int, default=128, help="width of the shared frame")
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--hops", type=int, nargs="*", default=[12, 24, 36])
    p.add_argument("--out", default=None)
    return p.parse_args()


def fit(X, Y, alpha):
    G = X.T @ X
    lam = alpha * torch.diagonal(G).mean()
    return torch.linalg.solve(
        G + lam * torch.eye(G.shape[0], dtype=G.dtype), X.T @ Y)


def truncate(M, r):
    """Top-r slice of a map, so a pairwise route passes the same bottleneck as the joint one."""
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vh[:r]


def r_at_1(P, T):
    Pn = torch.nn.functional.normalize(P, dim=1)
    Tn = torch.nn.functional.normalize(T, dim=1)
    S = Pn @ Tn.T
    d = torch.arange(len(P))
    return float((S.argmax(1) == d).double().mean())


def walk(Z, M, te, route, hops):
    if hops % (len(route) - 1):
        raise SystemExit(f"{hops} hops does not divide route of length {len(route) - 1}")
    P = Z[route[0]][te]
    for _ in range(hops // (len(route) - 1)):
        for i in range(len(route) - 1):
            P = P @ M[(route[i], route[i + 1])]
    return r_at_1(P, Z[route[0]][te])


def main():
    a = parse_args()
    torch.set_num_threads(os.cpu_count() or 8)
    a.out = a.out or os.path.join(a.dir, f"joint_frame_{a.domain}.json")
    rows = json.load(open(os.path.join(a.dir, f"{a.domain}_manifest.json")))["rows"]

    print(f"domain {a.domain}   cpu float64")
    V = {}
    for t in TAGS:
        f = os.path.join(a.dir, "states", f"{a.domain}_{t}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        ok = z["ok"]
        if len(rows) != len(ok):
            raise SystemExit(f"{t}: manifest {len(rows)} rows, states {len(ok)}")
        V[t] = {"img": z["item"][ok],
                "key": np.array([r["key"] for r, k in zip(rows, ok) if k])}
    tags = [t for t in TAGS if t in V]
    if len(tags) < 4:
        raise SystemExit(f"need four vendors, have {tags}")

    shared = set(V[tags[0]]["key"])
    for t in tags[1:]:
        shared &= set(V[t]["key"])
    order = [k for k in V[tags[0]]["key"] if k in shared]
    n = len(order)
    perm = np.random.default_rng(0).permutation(n)
    n_te = int(a.holdout_frac * n)
    te, tr = torch.tensor(perm[:n_te]), torch.tensor(perm[n_te:])
    print(f"  aligned {n}, train {n - n_te}, holdout {n_te}")

    # Whiten each vendor to a common width so no backbone wins the joint fit on
    # raw dimensionality alone.
    Z, dims = {}, {}
    for t in tags:
        pos = {k: i for i, k in enumerate(V[t]["key"])}
        M0 = torch.tensor(V[t]["img"][np.array([pos[k] for k in order])]).double()
        dims[t] = int(M0.shape[1])
        M0 = M0 - M0[tr].mean(0, keepdim=True)
        _, S, Vh = torch.linalg.svd(M0[tr], full_matrices=False)
        k = min(a.pca, int((S > S[0] * 1e-8).sum()))
        Z[t] = (M0 @ Vh[:k].T) / (S[:k] / np.sqrt(len(tr)))
    print(f"  native dims {dims}, whitened to {Z[tags[0]].shape[1]}")

    # MAXVAR GCCA: the shared frame is the top subspace of the stacked whitened
    # views, which is the directions all four agree on at once.
    _, Sj, Vhj = torch.linalg.svd(torch.cat([Z[t][tr] for t in tags], 1),
                                  full_matrices=False)
    r = min(a.joint, Vhj.shape[0])
    Wj = {t: fit(Z[t][tr], (torch.cat([Z[q][tr] for q in tags], 1) @ Vhj[:r].T),
                 a.ridge) for t in tags}
    Bj = {t: fit(Z[t][tr] @ Wj[t], Z[t][tr], a.ridge) for t in tags}
    ev = float((Sj[:r] ** 2).sum() / (Sj ** 2).sum())
    print(f"  joint frame width {r}, holds {ev:.4f} of stacked variance")

    res = {"question": "is there one frame shared by all four backbones, or "
                       "twelve compatible private ones",
           "domain": a.domain, "vendors": tags, "native_dims": dims,
           "whitened_dims": int(Z[tags[0]].shape[1]), "joint_width": int(r),
           "joint_variance_retained": round(ev, 4),
           "n_train": int(len(tr)), "n_test": int(n_te),
           "maps_bilateral": len(tags) * (len(tags) - 1),
           "maps_joint": 2 * len(tags),
           "direct": {}, "via_joint": {}, "consensus": {}, "ladder": {}}

    Md = {(x, y): fit(Z[x][tr], Z[y][tr], a.ridge)
          for x, y in itertools.product(tags, repeat=2)}
    Mj = {(x, y): Wj[x] @ Bj[y] for x, y in itertools.product(tags, repeat=2)}

    print(f"\n{'from':<11}{'to':<11}{'direct':>9}{'via joint':>11}{'delta':>9}")
    dd, jj = [], []
    for x, y in itertools.product(tags, repeat=2):
        if x == y:
            continue
        d1 = r_at_1(Z[x][te] @ Md[(x, y)], Z[y][te])
        j1 = r_at_1(Z[x][te] @ Mj[(x, y)], Z[y][te])
        res["direct"][f"{x}->{y}"] = round(d1, 4)
        res["via_joint"][f"{x}->{y}"] = round(j1, 4)
        dd.append(d1)
        jj.append(j1)
        print(f"{x:<11}{y:<11}{d1:9.4f}{j1:11.4f}{j1 - d1:+9.4f}")

    # Can three vendors read together beat the best single one reading alone?
    print(f"\n{'target':<11}{'best single':>12}{'three together':>15}{'gain':>8}")
    for y in tags:
        src = [x for x in tags if x != y]
        singles = {x: r_at_1(Z[x][te] @ Md[(x, y)], Z[y][te]) for x in src}
        C = fit(torch.cat([Z[x][tr] for x in src], 1), Z[y][tr], a.ridge)
        trio = r_at_1(torch.cat([Z[x][te] for x in src], 1) @ C, Z[y][te])
        best = max(singles.values())
        res["consensus"][y] = {"best_single": round(best, 4),
                               "best_single_source": max(singles, key=singles.get),
                               "three_together": round(trio, 4),
                               "gain": round(trio - best, 4)}
        print(f"{y:<11}{best:12.4f}{trio:15.4f}{trio - best:+8.4f}")

    A, B, C, D = tags
    routes = {"self_loop": [A, A], "1_edge": [A, B, A],
              "2_edges": [A, B, C, B, A], "3_edges": [A, B, C, D, C, B, A],
              "four_cycle": [A, B, C, D, A]}
    Mr = {k: truncate(v, r) for k, v in Md.items()}
    print(f"\n{'route':<13}{'hops':>5}{'pairwise':>10}{'rank-matched':>14}"
          f"{'via joint':>11}")
    for name, route in routes.items():
        res["ladder"][name] = {}
        for h in a.hops:
            if h % (len(route) - 1):
                continue
            p1 = walk(Z, Md, te, route, h)
            m1 = walk(Z, Mr, te, route, h)
            q1 = walk(Z, Mj, te, route, h)
            res["ladder"][name][str(h)] = {"pairwise": round(p1, 4),
                                           "pairwise_rank_matched": round(m1, 4),
                                           "via_joint": round(q1, 4)}
            print(f"{name:<13}{h:5d}{p1:10.4f}{m1:14.4f}{q1:11.4f}")

    g = torch.Generator().manual_seed(0)
    sh = torch.randperm(n_te, generator=g)
    res["shuffled_floor"] = round(
        r_at_1(Z[A][te] @ Md[(A, B)], Z[B][te][sh]), 4)

    res["summary"] = {
        "mean_direct": round(float(np.mean(dd)), 4),
        "mean_via_joint": round(float(np.mean(jj)), 4),
        "joint_cost": round(float(np.mean(dd) - np.mean(jj)), 4),
        "mean_consensus_gain": round(
            float(np.mean([v["gain"] for v in res["consensus"].values()])), 4),
        "shuffled_floor": res["shuffled_floor"],
        "ladder_caveat": "via_joint flattens the ladder because the composed map "
                         "collapses to the joint width and cannot lose further "
                         "dimensions. compare against pairwise_rank_matched, not "
                         "against full-rank pairwise.",
        "reading": "via_joint near direct means four encoders and four decoders "
                   "replace twelve pairwise maps at little cost, so the backbones "
                   "share one frame. on the ladder, only rank-matched pairwise is "
                   "a fair comparison for the joint route."}
    json.dump(res, open(a.out, "w"), indent=2)
    s = res["summary"]
    print(f"\ndirect      {s['mean_direct']}   ({res['maps_bilateral']} maps)")
    print(f"via joint   {s['mean_via_joint']}   ({res['maps_joint']} maps), "
          f"cost {s['joint_cost']}")
    print(f"consensus   mean gain over best single {s['mean_consensus_gain']:+}")
    print(f"shuffled    {s['shuffled_floor']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
