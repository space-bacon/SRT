"""Does L20's advantage over L47 survive more training data?

A head fitted on 4,000 images reads 0.293 at L20 against 0.230 at L47, which
puts the shipped tap layer in question. Before re-encoding train2017 at a new
depth, which is an overnight job, this asks the cheaper question: is the gap
stable as the fit gets more data, or is it a small-sample effect that closes?

Same held-out 1,000 images throughout, so only the training size moves. Several
seeds per point, because a gap of 0.06 read off one fit is not a gap.

    python scripts/tap_layer_scaling.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

LAYERS = [4, 12, 20, 28, 36, 44, 47, 54, 60]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="/root/depth_states_all9.npz")
    p.add_argument("--cache-layers", default="4,12,20,28,36,44,47,54,60",
                   help="layer order the cache was written with")
    p.add_argument("--layers", default="20,47")
    p.add_argument("--sizes", default="250,500,1000,2000,4000")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--holdout", type=int, default=1000)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--out", default="/root/tap_layer_scaling.json")
    return p.parse_args()


def fit_and_score(I_tr, T_tr, I_te, T_te, dim, epochs, lr, tau, seed):
    torch.manual_seed(seed)
    mu_i, mu_t = I_tr.mean(0, keepdim=True), T_tr.mean(0, keepdim=True)
    Wi = torch.nn.Linear(I_tr.shape[1], dim).cuda()
    Wt = torch.nn.Linear(T_tr.shape[1], dim).cuda()
    opt = torch.optim.Adam(list(Wi.parameters()) + list(Wt.parameters()), lr=lr)
    n = len(I_tr)
    for _ in range(epochs):
        perm = torch.randperm(n, device="cuda")
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            if len(idx) < 8:
                continue
            a = F.normalize(Wi(I_tr[idx] - mu_i), dim=1)
            b = F.normalize(Wt(T_tr[idx] - mu_t), dim=1)
            logits = a @ b.T / tau
            lab = torch.arange(len(idx), device="cuda")
            loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        a = F.normalize(Wi(I_te - mu_i), dim=1)
        b = F.normalize(Wt(T_te - mu_t), dim=1)
        S = a @ b.T
        d = torch.arange(len(a), device="cuda")
        rank = (S > S[d, d][:, None]).sum(1) + 1
    return float((rank == 1).float().mean())


def main() -> None:
    a = parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    cache_layers = [int(x) for x in a.cache_layers.split(",")]
    sizes = [int(x) for x in a.sizes.split(",")]
    z = np.load(a.cache, allow_pickle=True)
    img, txt = z["img"], z["txt"]
    n_all = img.shape[0]

    rng = np.random.default_rng(0)
    order = rng.permutation(n_all)
    te, pool = order[:a.holdout], order[a.holdout:]
    print(f"{n_all} images, {len(pool)} available to train on, {len(te)} held out")

    out = {}
    for L in layers:
        li = cache_layers.index(L)
        I = torch.tensor(img[:, li], dtype=torch.float32, device="cuda")
        T = torch.tensor(txt[:, li], dtype=torch.float32, device="cuda")
        out[f"L{L}"] = {}
        for n in sizes:
            if n > len(pool):
                continue
            vals = []
            for s in range(a.seeds):
                sub = np.random.default_rng(100 + s).choice(pool, size=n, replace=False)
                vals.append(fit_and_score(I[sub], T[sub], I[te], T[te],
                                          a.dim, a.epochs, a.lr, a.tau, s))
            out[f"L{L}"][str(n)] = {"mean_r@1": float(np.mean(vals)),
                                    "std": float(np.std(vals)),
                                    "seeds": vals}
            print(f"  L{L:<3d} n={n:<5d} r@1 {np.mean(vals):.3f} +/- {np.std(vals):.3f}",
                  flush=True)

    print()
    print(f"{'n':>7s} {'L20':>16s} {'L47':>16s} {'gap':>8s}")
    for n in sizes:
        k = str(n)
        if k in out.get("L20", {}) and k in out.get("L47", {}):
            g = out["L20"][k]["mean_r@1"] - out["L47"][k]["mean_r@1"]
            print(f"{n:>7d} {out['L20'][k]['mean_r@1']:>8.3f}+/-{out['L20'][k]['std']:<6.3f} "
                  f"{out['L47'][k]['mean_r@1']:>8.3f}+/-{out['L47'][k]['std']:<6.3f} {g:>+8.3f}")

    with open(a.out, "w") as f:
        json.dump({"question": "does the L20 advantage over L47 survive more training data",
                   "holdout": a.holdout, "seeds_per_point": a.seeds,
                   "results": out}, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
