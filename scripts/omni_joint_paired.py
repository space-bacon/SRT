#!/usr/bin/env python3
"""Does one shared tower beat a tower per modality when the two are paired query by query?

`artifacts/nla/omni/omni_joint.json` found the shared tower above the per-modality towers on every gallery, from one
unseeded run with no paired test. This refits both on the same states, split and hyperparameters as
`scripts/omni_joint_fit.py` (the default_rng(0) holdout, dim 512, 120 epochs, lr 1e-3, tau 0.05, batch 512, items
centred per modality and texts pooled, both from the train split), over seeded runs on the CPU. The towers are paired
on each held-out text query: an exact sign test per seed and gallery on the queries only one tower answers at rank 1,
and across seeds each query's hit rate, shared minus per-modality, with its standard error and a sign test. Raw
anisotropy and a derangement floor are reported beside the result. One JSON line per seed and tower is appended to
the seeds file as the run proceeds, and a rerun resumes from it.

    python3 scripts/omni_joint_paired.py
"""
import argparse
import glob
import json
import math
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

OMNI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "nla", "omni")
MODS = ("image", "audio", "video")
GALLERIES = ("mixed_gallery",) + MODS


def load():
    items, texts, mods = [], [], []
    for p in sorted(glob.glob(os.path.join(OMNI, "omni_states_s*.npz"))):
        if not re.fullmatch(r"omni_states_s\d+\.npz", os.path.basename(p)):
            continue
        z = np.load(p, allow_pickle=True)
        ok = z["ok"]
        items.append(z["item"][ok])
        texts.append(z["text"][ok])
        mods += [str(m) for m, k in zip(z["modality"], ok) if k]
    return np.concatenate(items).astype(np.float32), np.concatenate(texts).astype(np.float32), np.array(mods)


def ranks(A, B):
    S = B @ A.T
    d = torch.arange(len(A))
    return (S > S[d, d][:, None]).sum(1) + 1


def fit(I, T, mods, tr, te, shared, dim=512, epochs=120, lr=1e-3, tau=0.05):
    keys = ["all"] if shared else list(MODS)
    Wi = torch.nn.ModuleDict({k: torch.nn.Linear(I.shape[1], dim) for k in keys})
    Wt = torch.nn.Linear(T.shape[1], dim)
    mu_i = {m: I[torch.tensor(tr[mods[tr] == m])].mean(0, keepdim=True) for m in MODS if (mods[tr] == m).any()}
    mu_t = T[torch.tensor(tr)].mean(0, keepdim=True)
    opt = torch.optim.Adam(list(Wi.parameters()) + list(Wt.parameters()), lr=lr)

    def project(idx_np):
        idx = torch.tensor(idx_np)
        a = torch.zeros(len(idx_np), dim)
        sub = mods[idx_np]
        for m in MODS:
            sel = sub == m
            if sel.any():
                rows = torch.tensor(np.where(sel)[0])
                a[rows] = Wi["all" if shared else m](I[idx][rows] - mu_i[m])
        return F.normalize(a, dim=1), F.normalize(Wt(T[idx] - mu_t), dim=1)

    n = len(tr)
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for s in range(0, n, 512):
            idx_np = tr[perm[s:s + 512]]
            if len(idx_np) < 8:
                continue
            a, b = project(idx_np)
            lg = a @ b.T / tau
            lab = torch.arange(len(idx_np))
            loss = 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab))
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return project(te)


def hits(A, B, te_mods):
    """Rank-1 hit per held-out text query, on the mixed gallery and within each modality's own gallery."""
    out = {"mixed_gallery": (ranks(A, B) == 1).numpy()}
    for m in MODS:
        s = torch.tensor(np.where(te_mods == m)[0])
        out[m] = (ranks(A[s], B[s]) == 1).numpy()
    return out


def sign_p(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def raw_anisotropy(X, rng, pairs=20000):
    i, j = rng.integers(0, len(X), pairs), rng.integers(0, len(X), pairs)
    keep = i != j
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return float(np.mean(np.sum(Xn[i[keep]] * Xn[j[keep]], axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OMNI, "omni_joint_paired.json"))
    ap.add_argument("--seeds-log", default=os.path.join(OMNI, "omni_joint_paired_seeds.jsonl"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--derangements", type=int, default=20)
    args = ap.parse_args()
    torch.set_num_threads(8)
    I_np, T_np, mods = load()
    I, T = torch.tensor(I_np), torch.tensor(T_np)
    order = np.random.default_rng(0).permutation(len(I))
    n_te = int(0.2 * len(I))
    te, tr = order[:n_te], order[n_te:]
    te_mods = mods[te]
    a_rng = np.random.default_rng(1)
    res = {"question": "does one shared tower beat a tower per modality when the two are paired query by query",
           "replicates": "artifacts/nla/omni/omni_joint.json, one unseeded run",
           "protocol": {"split": "default_rng(0), holdout 0.2", "dim": 512, "epochs": 120, "lr": 1e-3, "tau": 0.05,
                        "batch": 512, "centring": "per modality for items, pooled for text, train split",
                        "device": "cpu", "seeds": list(range(args.seeds))},
           "n_total": len(I), "n_train": len(tr), "n_holdout": len(te),
           "holdout_counts": {m: int((te_mods == m).sum()) for m in MODS},
           "raw_anisotropy": {"items": raw_anisotropy(I_np, a_rng), "texts": raw_anisotropy(T_np, a_rng)}}
    done = {}
    if os.path.exists(args.seeds_log):
        for line in open(args.seeds_log):
            r = json.loads(line)
            done[(r["seed"], r["tower"])] = r
    per_seed = {}
    for seed in range(args.seeds):
        h = {}
        for tower, shared in (("shared", True), ("per_modality", False)):
            if (seed, tower) in done:
                h[tower] = {g: np.array(v, dtype=bool) for g, v in done[(seed, tower)]["hits"].items()}
                continue
            torch.manual_seed(seed)
            np.random.seed(seed)
            A, B = fit(I, T, mods, tr, te, shared)
            h[tower] = hits(A, B, te_mods)
            row = {"seed": seed, "tower": tower, "r1": {g: float(v.mean()) for g, v in h[tower].items()},
                   "hits": {g: v.astype(int).tolist() for g, v in h[tower].items()}}
            if shared:
                floors = []
                for s in range(args.derangements):
                    g = np.random.default_rng(500 + s)
                    perm = g.permutation(len(te))
                    while (perm == np.arange(len(te))).any():
                        perm = g.permutation(len(te))
                    floors.append(float(ranks(A[torch.tensor(perm)], B).float().median()))
                row["derangement_floor_median"] = [float(np.mean(floors)), float(np.std(floors))]
            with open(args.seeds_log, "a") as f:
                f.write(json.dumps(row) + "\n")
            done[(seed, tower)] = row
            print(f"seed {seed} {tower}: " + ", ".join(f"{g} {v:.4f}" for g, v in row["r1"].items()), flush=True)
        per_seed[seed] = {}
        for g in GALLERIES:
            s_, p_ = h["shared"][g], h["per_modality"][g]
            b, c = int(np.sum(s_ & ~p_)), int(np.sum(~s_ & p_))
            per_seed[seed][g] = {"shared": float(s_.mean()), "per_modality": float(p_.mean()), "wins": b,
                                 "losses": c, "sign_p": round(sign_p(b, c), 5), "n": int(len(s_))}
    res["per_seed"] = per_seed
    agg = {}
    for g in GALLERIES:
        gaps = np.array([per_seed[s][g]["shared"] - per_seed[s][g]["per_modality"] for s in per_seed])
        sh = np.array([per_seed[s][g]["shared"] for s in per_seed])
        pm = np.array([per_seed[s][g]["per_modality"] for s in per_seed])
        agg[g] = {"shared_mean": round(float(sh.mean()), 4), "shared_sd": round(float(sh.std(ddof=1)), 4),
                  "per_modality_mean": round(float(pm.mean()), 4), "per_modality_sd": round(float(pm.std(ddof=1)), 4),
                  "gap_mean": round(float(gaps.mean()), 4), "gap_sd": round(float(gaps.std(ddof=1)), 4),
                  "gap_over_se": round(float(gaps.mean() / (gaps.std(ddof=1) / math.sqrt(len(gaps)))), 2)
                  if gaps.std(ddof=1) > 0 else None,
                  "seeds_shared_ahead": int(np.sum(gaps > 0)), "seeds_significant_at_0.05":
                  sum(1 for s in per_seed if per_seed[s][g]["sign_p"] < 0.05), "n": per_seed[0][g]["n"]}
    res["across_seeds"] = agg
    # Paired by query with training noise averaged out: each query's hit rate over seeds, shared minus per-modality.
    items = {}
    for g in GALLERIES:
        sh = np.mean([np.array(done[(s, "shared")]["hits"][g]) for s in range(args.seeds)], axis=0)
        pm = np.mean([np.array(done[(s, "per_modality")]["hits"][g]) for s in range(args.seeds)], axis=0)
        d = sh - pm
        b, c = int(np.sum(d > 0)), int(np.sum(d < 0))
        se = float(d.std(ddof=1) / math.sqrt(len(d)))
        items[g] = {"gap": round(float(d.mean()), 4), "se": round(se, 4),
                    "gap_over_se": round(float(d.mean()) / se, 2) if se else None,
                    "smallest_detectable_2se": round(2 * se, 4), "queries_shared_ahead": b,
                    "queries_per_modality_ahead": c, "sign_p": round(sign_p(b, c), 5), "n": int(len(d))}
    res["across_items"] = items
    fl = [done[(s, "shared")]["derangement_floor_median"][0] for s in range(args.seeds)]
    res["derangement_floor_median_mean"] = round(float(np.mean(fl)), 2)
    res["analytic_chance_median"] = (len(te) + 1) / 2
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, args.out)
    print(json.dumps({"raw_anisotropy": res["raw_anisotropy"], "across_items": items,
                      "floor": [res["derangement_floor_median_mean"], res["analytic_chance_median"]]}, indent=1))


if __name__ == "__main__":
    main()
