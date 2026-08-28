"""Fit one shared space across several backbones, and test cross-model retrieval.

Each model gets its own image tower and text tower projecting into a single
d-dimensional space. Training pushes every model's image vector toward every
model's text vector for the same photograph, so the space is shared rather than
per-model. If that works, a gallery encoded by one vendor is searchable by
another vendor's encoder.

Three controls decide whether any number here means anything:

  solo        each model fitted alone, as the upper reference. If joint costs a
              lot of within-model R@1, the sharing is expensive and we say so.
  derangement an empirical floor per direction, from shuffled pairings. Floors
              are arm-dependent, so analytic chance is not a substitute.
  centering   per-model mean removed before any cosine, because these states are
              strongly anisotropic and raw cosine is not interpretable.

    python scripts/joint_space_fit.py --states /root/states_*.npz
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--states", nargs="+", required=True)
    p.add_argument("--layer-idx", type=int, default=1, help="which tapped layer")
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--holdout", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--derangements", type=int, default=20)
    p.add_argument("--out", default="/root/joint_space.json")
    return p.parse_args()


class Towers(torch.nn.Module):
    def __init__(self, dims: dict[str, int], dim: int):
        super().__init__()
        self.img = torch.nn.ModuleDict(
            {k: torch.nn.Linear(d, dim) for k, d in dims.items()})
        self.txt = torch.nn.ModuleDict(
            {k: torch.nn.Linear(d, dim) for k, d in dims.items()})


def ranks_of(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Rank of the true image for each caption."""
    S = B @ A.T
    dg = torch.arange(len(A), device=A.device)
    return (S > S[dg, dg][:, None]).sum(1) + 1


def score(rank: torch.Tensor) -> dict:
    return {"r@1": float((rank == 1).float().mean()),
            "r@10": float((rank <= 10).float().mean()),
            "median_rank": float(rank.float().median())}


def fit(models, I, T, tr, te, dim, epochs, lr, tau, joint: bool):
    dims = {m: I[m].shape[1] for m in models}
    net = Towers(dims, dim).cuda()
    mu_i = {m: I[m][tr].mean(0, keepdim=True) for m in models}
    mu_t = {m: T[m][tr].mean(0, keepdim=True) for m in models}
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(tr)
    for ep in range(epochs):
        perm = torch.randperm(n, device="cuda")
        for s in range(0, n, 1024):
            idx = tr[perm[s:s + 1024]]
            if len(idx) < 8:
                continue
            a = {m: F.normalize(net.img[m](I[m][idx] - mu_i[m]), dim=1) for m in models}
            b = {m: F.normalize(net.txt[m](T[m][idx] - mu_t[m]), dim=1) for m in models}
            lab = torch.arange(len(idx), device="cuda")
            # Joint: every image model against every text model. Solo: only the
            # diagonal, which is the same recipe each model would get alone.
            pairs = [(x, y) for x in models for y in models
                     if joint or x == y]
            loss = 0.0
            for x, y in pairs:
                lg = a[x] @ b[y].T / tau
                loss = loss + 0.5 * (F.cross_entropy(lg, lab)
                                     + F.cross_entropy(lg.T, lab))
            loss = loss / len(pairs)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 20 == 0:
            print(f"    epoch {ep} loss {float(loss):.4f}", flush=True)
    with torch.no_grad():
        A = {m: F.normalize(net.img[m](I[m][te] - mu_i[m]), dim=1) for m in models}
        B = {m: F.normalize(net.txt[m](T[m][te] - mu_t[m]), dim=1) for m in models}
    return A, B


def main() -> None:
    a = parse_args()
    paths = sorted({p for g in a.states for p in glob.glob(g)})
    I, T, models = {}, {}, []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        m = str(z["model"]) if "model" in z.files else p
        tag = m.split("/")[-1]
        li = min(a.layer_idx, z["img"].shape[1] - 1)
        I[tag] = torch.tensor(z["img"][:, li], dtype=torch.float32, device="cuda")
        T[tag] = torch.tensor(z["txt"][:, li], dtype=torch.float32, device="cuda")
        models.append(tag)
        print(f"  {tag}: {I[tag].shape}", flush=True)
    n = min(len(I[m]) for m in models)
    rng = np.random.default_rng(0)
    order = torch.tensor(rng.permutation(n), device="cuda")
    te, tr = order[:a.holdout], order[a.holdout:]
    print(f"\n{len(models)} models, fit {len(tr)}, hold out {len(te)}\n", flush=True)

    res = {"models": models, "n_train": len(tr), "n_holdout": len(te),
           "question": "can one linear space serve several unrelated backbones, "
                       "so a gallery encoded by one is searchable by another",
           "controls": {}}

    print("solo reference:", flush=True)
    As, Bs = fit(models, I, T, tr, te, a.dim, a.epochs, a.lr, a.tau, joint=False)
    res["solo"] = {m: score(ranks_of(As[m], Bs[m])) for m in models}

    print("\njoint space:", flush=True)
    Aj, Bj = fit(models, I, T, tr, te, a.dim, a.epochs, a.lr, a.tau, joint=True)

    res["joint"] = {}
    for x in models:
        for y in models:
            res["joint"][f"{y}_text -> {x}_image"] = score(ranks_of(Aj[x], Bj[y]))

    # Empirical floor per direction: same vectors, pairing broken.
    print("\nderangement floors:", flush=True)
    floors = {}
    for x in models:
        for y in models:
            meds = []
            for s in range(a.derangements):
                g = np.random.default_rng(1000 + s)
                perm = g.permutation(len(te))
                # reject fixed points so no caption keeps its own image
                while (perm == np.arange(len(te))).any():
                    perm = g.permutation(len(te))
                p = torch.tensor(perm, device="cuda")
                meds.append(float(ranks_of(Aj[x][p], Bj[y]).float().median()))
            floors[f"{y}_text -> {x}_image"] = {
                "median_mean": float(np.mean(meds)),
                "median_sd": float(np.std(meds)),
                "n": a.derangements}
    res["controls"]["derangement_floor"] = floors
    res["controls"]["note"] = (
        "floors are measured per direction because a floor is a property of the "
        "arm being scored, not of the gallery")

    diag = [res["joint"][f"{m}_text -> {m}_image"]["r@1"] for m in models]
    off = [v["r@1"] for k, v in res["joint"].items()
           if k.split("_text")[0] != k.split("-> ")[1].replace("_image", "")]
    res["summary"] = {
        "within_model_mean_r@1": float(np.mean(diag)),
        "cross_model_mean_r@1": float(np.mean(off)) if off else None,
        "solo_mean_r@1": float(np.mean([res["solo"][m]["r@1"] for m in models])),
    }
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwithin {res['summary']['within_model_mean_r@1']:.4f}  "
          f"cross {res['summary']['cross_model_mean_r@1']:.4f}  "
          f"solo {res['summary']['solo_mean_r@1']:.4f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
