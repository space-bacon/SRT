"""Fit ONE shared space over images, audio and video from a single omni backbone.

The claim being tested is that one linear map can place every modality beside
text in a single space, so a mixed gallery is searchable by one text query. A
per-modality tower would score better and prove less, so that variant is fitted
too and reported as the upper reference: the gap is the price of sharing.

Controls, all of which can kill the result:
  solo         per-modality towers, the upper reference
  derangement  empirical floor per direction, since floors are arm-dependent
  centering    per-modality mean removed; raw cosine on these states measured
               +0.869 between unrelated items and sat exactly at chance

    python scripts/omni_joint_fit.py --states '/root/omni_states_s*.npz'
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

MODS = ("image", "audio", "video")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--states", nargs="+", required=True)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--derangements", type=int, default=20)
    p.add_argument("--out", default="/root/omni_joint.json")
    return p.parse_args()


def load(patterns):
    items, texts, mods = [], [], []
    seen = []
    for g in patterns:
        for p in sorted(glob.glob(g)):
            # A glob like omni_states_s*.npz also matches the mid-run
            # *.partial.npz checkpoints and any smoke file, which silently
            # doubles the data and voids the result. Take final shards only.
            base = os.path.basename(p)
            if ".partial." in base or not re.fullmatch(r"omni_states_s\d+\.npz", base):
                print(f"  skipping {base}", flush=True)
                continue
            z = np.load(p, allow_pickle=True)
            ok = z["ok"]
            items.append(z["item"][ok])
            texts.append(z["text"][ok])
            mods += [str(m) for m, k in zip(z["modality"], ok) if k]
            seen.append(f"{base}({int(ok.sum())})")
    print(f"  loaded: {', '.join(seen)}", flush=True)
    return (np.concatenate(items).astype(np.float32),
            np.concatenate(texts).astype(np.float32), np.array(mods))


def ranks(A, B):
    S = B @ A.T
    d = torch.arange(len(A), device=A.device)
    return (S > S[d, d][:, None]).sum(1) + 1


def score(r):
    return {"r@1": float((r == 1).float().mean()),
            "r@10": float((r <= 10).float().mean()),
            "median_rank": float(r.float().median()),
            "n": int(len(r))}


def fit(I, T, mods, tr, te, dim, epochs, lr, tau, shared: bool):
    """shared=True: one item tower for every modality. False: one per modality.

    `tr` and `te` are numpy index arrays; modality lookup uses them directly.
    """
    d_in = I.shape[1]
    keys = ["all"] if shared else list(MODS)
    Wi = torch.nn.ModuleDict({k: torch.nn.Linear(d_in, dim) for k in keys}).cuda()
    Wt = torch.nn.Linear(d_in, dim).cuda()
    # Centre per modality: the anisotropy is modality-specific, so one global
    # mean would leave each modality's offset intact.
    tr_t = torch.tensor(tr, device="cuda")
    mu_i = {m: I[torch.tensor(tr[mods[tr] == m], device="cuda")].mean(0, keepdim=True)
            for m in MODS if (mods[tr] == m).any()}
    mu_t = T[tr_t].mean(0, keepdim=True)
    opt = torch.optim.Adam(list(Wi.parameters()) + list(Wt.parameters()), lr=lr)

    def project(idx_np):
        idx = torch.tensor(idx_np, device="cuda")
        a = torch.zeros(len(idx_np), dim, device="cuda")
        sub = mods[idx_np]
        for m in MODS:
            sel = sub == m
            if not sel.any():
                continue
            k = "all" if shared else m
            rows = torch.tensor(np.where(sel)[0], device="cuda")
            a[rows] = Wi[k](I[idx][rows] - mu_i[m])
        return F.normalize(a, dim=1), F.normalize(Wt(T[idx] - mu_t), dim=1)

    n = len(tr)
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for s in range(0, n, 512):
            idx_np = tr[perm[s:s + 512]]
            if len(idx_np) < 8:
                continue
            a, b = project(idx_np)
            lg = a @ b.T / tau
            lab = torch.arange(len(idx_np), device="cuda")
            loss = 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 40 == 0:
            print(f"    epoch {ep} loss {float(loss):.4f}", flush=True)
    with torch.no_grad():
        return project(te)


def main() -> None:
    a = parse_args()
    I_np, T_np, mods_np = load(a.states)
    print(f"loaded {len(I_np)} items: "
          f"{ {m: int((mods_np == m).sum()) for m in MODS} }", flush=True)

    I = torch.tensor(I_np, device="cuda")
    T = torch.tensor(T_np, device="cuda")
    mods = np.array(mods_np)
    n = len(I)
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    n_te = int(a.holdout_frac * n)
    te_np, tr_np = order[:n_te], order[n_te:]

    res = {"question": "can one linear map place image, audio and video beside "
                       "text in a single searchable space",
           "n_total": n, "n_train": len(tr_np), "n_holdout": len(te_np),
           "counts": {m: int((mods == m).sum()) for m in MODS},
           "anisotropy_note": "raw cosine between unrelated items measured +0.869 "
                              "on these states and retrieval sat exactly at "
                              "chance; every number here is per-modality centred"}

    for tag, shared in (("shared", True), ("per_modality", False)):
        print(f"\n{tag} tower:", flush=True)
        A, B = fit(I, T, mods, tr_np, te_np, a.dim, a.epochs, a.lr, a.tau, shared)
        te_mods = mods[te_np]
        # One mixed index holding every modality, queried by text.
        block = {"mixed_gallery": score(ranks(A, B))}
        for m in MODS:
            sel = np.where(te_mods == m)[0]
            if len(sel) > 1:
                s = torch.tensor(sel, device="cuda")
                block[m] = score(ranks(A[s], B[s]))
        res[tag] = block
        print(f"  mixed-gallery r@1 {block['mixed_gallery']['r@1']:.4f} "
              f"median {block['mixed_gallery']['median_rank']:.0f}", flush=True)

        if tag == "shared":
            floors = []
            for s in range(a.derangements):
                g = np.random.default_rng(500 + s)
                perm = g.permutation(len(te_np))
                while (perm == np.arange(len(te_np))).any():
                    perm = g.permutation(len(te_np))
                p = torch.tensor(perm, device="cuda")
                floors.append(float(ranks(A[p], B).float().median()))
            res["derangement_floor"] = {
                "median_mean": float(np.mean(floors)),
                "median_sd": float(np.std(floors)),
                "n": a.derangements,
                "analytic_chance": (len(te_np) + 1) / 2}
            print(f"  floor {np.mean(floors):.0f} +- {np.std(floors):.0f} "
                  f"(analytic {(len(te_np)+1)/2:.0f})", flush=True)

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
