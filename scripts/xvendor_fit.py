"""Can a gallery encoded by one vendor be searched by another vendor's encoder?

Two backbones that never exchanged a byte encode the same items. Each gets its
own linear map into one shared space, trained only on (item, caption) pairs.
The question is whether the OFF-DIAGONAL works: Qwen-encoded gallery, gemma
query, and the reverse.

Controls that decide whether any of this means anything:
  within-vendor   each vendor against itself, the upper reference
  derangement     empirical floor per direction, since floors are arm-dependent
  centering       per (vendor, modality) mean removed; raw cosine on these
                  states sits at chance

    python scripts/xvendor_fit.py --a '/root/omni_states_s*.npz' \
                                  --b '/root/gemma4_states_s*.npz'
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--a", nargs="+", required=True)
    p.add_argument("--b", nargs="+", required=True)
    p.add_argument("--name-a", default="qwen3omni")
    p.add_argument("--name-b", default="gemma4")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--derangements", type=int, default=20)
    p.add_argument("--out", default="/root/xvendor.json")
    return p.parse_args()


def load(patterns):
    """Final shards only: a *_s*.npz glob also matches mid-run .partial files,
    which silently doubles the data and voids the result."""
    it, tx, md = [], [], []
    for g in patterns:
        for p in sorted(glob.glob(g)):
            b = os.path.basename(p)
            if ".partial." in b or not re.search(r"_s\d+\.npz$", b):
                continue
            z = np.load(p, allow_pickle=True)
            ok = z["ok"]
            it.append(z["item"][ok])
            tx.append(z["text"][ok])
            md += [str(m) for m, k in zip(z["modality"], ok) if k]
    return (np.concatenate(it).astype(np.float32),
            np.concatenate(tx).astype(np.float32), np.array(md))


def ranks(A, B):
    S = B @ A.T
    d = torch.arange(len(A), device=A.device)
    return (S > S[d, d][:, None]).sum(1) + 1


def score(r):
    return {"r@1": float((r == 1).float().mean()),
            "r@10": float((r <= 10).float().mean()),
            "median_rank": float(r.float().median()), "n": int(len(r))}


def main() -> None:
    a = parse_args()
    Ia, Ta, Ma = load(a.a)
    Ib, Tb, Mb = load(a.b)
    print(f"{a.name_a}: {len(Ia)} items {dict(zip(*np.unique(Ma, return_counts=True)))}")
    print(f"{a.name_b}: {len(Ib)} items {dict(zip(*np.unique(Mb, return_counts=True)))}")

    # The two encoders ran over different row sets (gemma-4 has no audio tower),
    # so align on the modalities both actually produced, in manifest order.
    keep = [m for m in ("image", "video", "audio")
            if (Ma == m).any() and (Mb == m).any()]
    ia = np.concatenate([np.where(Ma == m)[0] for m in keep])
    ib = np.concatenate([np.where(Mb == m)[0] for m in keep])
    n = min(len(ia), len(ib))
    ia, ib = ia[:n], ib[:n]
    mods = Ma[ia]
    assert (Ma[ia] == Mb[ib]).all(), "modality alignment mismatch"
    print(f"aligned on {keep}: {n} paired items\n")

    X = {a.name_a: (torch.tensor(Ia[ia]).cuda(), torch.tensor(Ta[ia]).cuda()),
         a.name_b: (torch.tensor(Ib[ib]).cuda(), torch.tensor(Tb[ib]).cuda())}
    names = [a.name_a, a.name_b]

    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    n_te = int(a.holdout_frac * n)
    te, tr = order[:n_te], order[n_te:]

    Wi = torch.nn.ModuleDict(
        {k: torch.nn.Linear(X[k][0].shape[1], a.dim) for k in names}).cuda()
    Wt = torch.nn.ModuleDict(
        {k: torch.nn.Linear(X[k][1].shape[1], a.dim) for k in names}).cuda()
    mu = {k: {m: X[k][0][torch.tensor(tr[mods[tr] == m]).cuda()].mean(0, keepdim=True)
              for m in keep} for k in names}
    mut = {k: X[k][1][torch.tensor(tr).cuda()].mean(0, keepdim=True) for k in names}
    opt = torch.optim.Adam(list(Wi.parameters()) + list(Wt.parameters()), lr=a.lr)

    def project(k, idx_np):
        idx = torch.tensor(idx_np).cuda()
        out = torch.zeros(len(idx_np), a.dim, device="cuda")
        sub = mods[idx_np]
        for m in keep:
            sel = sub == m
            if sel.any():
                rows = torch.tensor(np.where(sel)[0]).cuda()
                out[rows] = Wi[k](X[k][0][idx][rows] - mu[k][m])
        return F.normalize(out, dim=1), F.normalize(Wt[k](X[k][1][idx] - mut[k]), dim=1)

    for ep in range(a.epochs):
        perm = np.random.permutation(len(tr))
        for s in range(0, len(tr), 512):
            idx = tr[perm[s:s + 512]]
            if len(idx) < 8:
                continue
            P = {k: project(k, idx) for k in names}
            lab = torch.arange(len(idx), device="cuda")
            loss = 0.0
            # Every vendor's item against every vendor's text, so the space is
            # shared rather than two spaces that happen to sit side by side.
            for x in names:
                for y in names:
                    lg = P[x][0] @ P[y][1].T / a.tau
                    loss = loss + 0.5 * (F.cross_entropy(lg, lab)
                                         + F.cross_entropy(lg.T, lab))
            loss = loss / (len(names) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 40 == 0:
            print(f"  epoch {ep} loss {float(loss):.4f}", flush=True)

    with torch.no_grad():
        P = {k: project(k, te) for k in names}

    res = {"question": "can a gallery encoded by one vendor be searched by "
                       "another vendor's encoder",
           "vendors": names, "modalities": keep, "n_paired": n,
           "n_train": len(tr), "n_holdout": len(te),
           "directions": {}, "floors": {}}

    for x in names:
        for y in names:
            tag = f"{y}_text -> {x}_gallery"
            res["directions"][tag] = score(ranks(P[x][0], P[y][1]))
            kind = "within-vendor" if x == y else "CROSS-VENDOR"
            s = res["directions"][tag]
            print(f"  {kind:<13} {tag:<38} r@1 {s['r@1']:.4f} median {s['median_rank']:.0f}")

    for x in names:
        for y in names:
            meds = []
            for s_ in range(a.derangements):
                g = np.random.default_rng(700 + s_)
                pm = g.permutation(len(te))
                while (pm == np.arange(len(te))).any():
                    pm = g.permutation(len(te))
                meds.append(float(ranks(P[x][0][torch.tensor(pm).cuda()],
                                        P[y][1]).float().median()))
            res["floors"][f"{y}_text -> {x}_gallery"] = {
                "median_mean": float(np.mean(meds)),
                "median_sd": float(np.std(meds))}

    cross = [v["r@1"] for k, v in res["directions"].items()
             if k.split("_text")[0] != k.split("-> ")[1].replace("_gallery", "")]
    within = [v["r@1"] for k, v in res["directions"].items()
              if k.split("_text")[0] == k.split("-> ")[1].replace("_gallery", "")]
    res["summary"] = {"cross_vendor_mean_r@1": float(np.mean(cross)),
                      "within_vendor_mean_r@1": float(np.mean(within)),
                      "retention": float(np.mean(cross) / np.mean(within))
                      if np.mean(within) else None}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\ncross {np.mean(cross):.4f} within {np.mean(within):.4f} "
          f"retention {res['summary']['retention']:.3f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
