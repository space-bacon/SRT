#!/usr/bin/env python3
"""Fit a shared retrieval space across N vendors and measure cross-vendor search.

Generalises xvendor_fit.py past two vendors.

Alignment is by manifest `key`, not array position. The vendors were sharded
differently (4 shards for the omni pair, 2 for the image-only pair) and any row
a vendor failed to encode shifts every later row, so positional alignment is
silently wrong the moment one encoder drops an item the others kept.

  --vendor tag:/states_s*.npz:/manifest_s*.json   (repeat per vendor)
"""
import argparse
import glob
import json
import re

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vendor", action="append", required=True,
                   help="tag:states_glob:manifest_glob")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--derangements", type=int, default=20)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--save-map", default=None)
    p.add_argument("--mu-from", default=None,
                   help="ABLATION ONLY: take the centering means from a saved "
                        "map instead of recomputing them on this gallery")
    p.add_argument("--out", default="/root/xvendor4.json")
    return p.parse_args()


def shard_sorted(pattern):
    """Final shards only, ordered by shard index."""
    fs = [f for f in glob.glob(pattern) if re.search(r"_s\d+\.(npz|json)$", f)]
    return sorted(fs, key=lambda f: int(re.search(r"_s(\d+)\.", f).group(1)))


def load_vendor(spec):
    tag, states_glob, man_glob = spec.split(":", 2)
    sf, mf = shard_sorted(states_glob), shard_sorted(man_glob)
    if len(sf) != len(mf):
        raise SystemExit(f"{tag}: {len(sf)} state shards but {len(mf)} manifests")
    item, text, keys, mods = [], [], [], []
    for s, m in zip(sf, mf):
        z = np.load(s)
        rows = json.load(open(m))["rows"]
        ok = z["ok"]
        if len(rows) != len(ok):
            raise SystemExit(f"{tag}: {m} has {len(rows)} rows, {s} has {len(ok)}")
        item.append(z["item"][ok])
        text.append(z["text"][ok])
        keys += [r["key"] for r, k in zip(rows, ok) if k]
        mods += [r["modality"] for r, k in zip(rows, ok) if k]
    print(f"  {tag:<10} {len(keys):5d} rows from {len(sf)} shards "
          f"({sum(1 for _ in keys)} ok)")
    return {"tag": tag,
            "item": np.concatenate(item).astype(np.float32),
            "text": np.concatenate(text).astype(np.float32),
            "key": np.array(keys), "mod": np.array(mods)}


def ranks(A, B):
    S = B @ A.T
    d = torch.arange(len(A), device=A.device)
    return (S > S[d, d][:, None]).sum(1) + 1


def score(r):
    return {"r@1": float((r == 1).float().mean()),
            "r@10": float((r <= 10).float().mean()),
            "median_rank": float(r.float().median()), "n": int(len(r))}


def main():
    a = parse_args()
    print("loading vendors:")
    V = [load_vendor(s) for s in a.vendor]
    tags = [v["tag"] for v in V]

    shared = set(V[0]["key"])
    for v in V[1:]:
        shared &= set(v["key"])
    order = [k for k in V[0]["key"] if k in shared]
    print(f"\naligned on {len(order)} keys shared by all {len(V)} vendors")

    X = {}
    for v in V:
        pos = {k: i for i, k in enumerate(v["key"])}
        idx = np.array([pos[k] for k in order])
        X[v["tag"]] = (torch.tensor(v["item"][idx]).cuda(),
                       torch.tensor(v["text"][idx]).cuda())
    mods = V[0]["mod"][np.array([{k: i for i, k in enumerate(V[0]["key"])}[k]
                                 for k in order])]
    keep = sorted(set(mods.tolist()))
    print(f"modalities: {keep}")

    n = len(order)
    rng = np.random.default_rng(0)
    perm0 = rng.permutation(n)
    n_te = int(a.holdout_frac * n)
    te, tr = perm0[:n_te], perm0[n_te:]

    Wi = torch.nn.ModuleDict(
        {t: torch.nn.Linear(X[t][0].shape[1], a.dim) for t in tags}).cuda()
    Wt = torch.nn.ModuleDict(
        {t: torch.nn.Linear(X[t][1].shape[1], a.dim) for t in tags}).cuda()
    mu = {t: {m: X[t][0][torch.tensor(tr[mods[tr] == m]).cuda()].mean(0, keepdim=True)
              for m in keep if (mods[tr] == m).any()} for t in tags}
    mut = {t: X[t][1][torch.tensor(tr).cuda()].mean(0, keepdim=True) for t in tags}
    if a.mu_from:
        # The anisotropy direction is domain-specific, so a mean fitted on one
        # gallery under-corrects on another. This path exists to measure that
        # penalty, never to produce a headline number.
        saved = torch.load(a.mu_from, map_location="cpu")
        missing = [t for t in tags if t not in saved["mu"]]
        if missing:
            raise SystemExit(f"--mu-from lacks vendors {missing}")
        mu = {t: {m: saved["mu"][t][m].cuda() for m in mu[t] if m in saved["mu"][t]}
              for t in tags}
        mut = {t: saved["mu_txt"][t].cuda() for t in tags}
        for t in tags:
            if not mu[t]:
                raise SystemExit(f"--mu-from has no matching modality for {t}")
        print(f"CARRIED means from {a.mu_from} (ablation, not a headline run)")
    opt = torch.optim.Adam(list(Wi.parameters()) + list(Wt.parameters()), lr=a.lr)

    def project(t, idx_np):
        idx = torch.tensor(idx_np).cuda()
        out = torch.zeros(len(idx_np), a.dim, device="cuda")
        sub = mods[idx_np]
        for m in keep:
            sel = sub == m
            if sel.any():
                rows = torch.tensor(np.where(sel)[0]).cuda()
                out[rows] = Wi[t](X[t][0][idx][rows] - mu[t][m])
        return F.normalize(out, dim=1), F.normalize(Wt[t](X[t][1][idx] - mut[t]), dim=1)

    for ep in range(a.epochs):
        p = np.random.permutation(len(tr))
        for s in range(0, len(tr), 512):
            idx = tr[p[s:s + 512]]
            if len(idx) < 8:
                continue
            P = {t: project(t, idx) for t in tags}
            lab = torch.arange(len(idx), device="cuda")
            loss = 0.0
            for x in tags:
                for y in tags:
                    lg = P[x][0] @ P[y][1].T / a.tau
                    loss = loss + 0.5 * (F.cross_entropy(lg, lab)
                                         + F.cross_entropy(lg.T, lab))
            loss = loss / (len(tags) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 40 == 0:
            print(f"  epoch {ep} loss {float(loss):.4f}", flush=True)

    with torch.no_grad():
        P = {t: project(t, te) for t in tags}

    res = {"question": "can a gallery encoded by one vendor be searched by "
                       "any other vendor's encoder",
           "vendors": tags, "modalities": keep, "n_aligned": n,
           "n_train": len(tr), "n_holdout": len(te),
           "directions": {}, "floors": {}}
    hits = {}
    print()
    for x in tags:
        for y in tags:
            tag = f"{y}_text -> {x}_gallery"
            r = ranks(P[x][0], P[y][1])
            res["directions"][tag] = score(r)
            hits[tag] = (r.float().cpu().numpy() <= 1).astype(np.float64)
            kind = "within" if x == y else "CROSS "
            print(f"  {kind} {tag:<34} r@1 {res['directions'][tag]['r@1']:.4f}")

    for x in tags:
        for y in tags:
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

    cross_t = [f"{y}_text -> {x}_gallery" for x in tags for y in tags if x != y]
    within_t = [f"{t}_text -> {t}_gallery" for t in tags]
    c = float(np.mean([res["directions"][t]["r@1"] for t in cross_t]))
    w = float(np.mean([res["directions"][t]["r@1"] for t in within_t]))
    res["summary"] = {"n_cross_pairs": len(cross_t),
                      "cross_vendor_mean_r@1": c,
                      "within_vendor_mean_r@1": w,
                      "retention": c / w if w else None}

    g = np.random.default_rng(12345)
    boots = []
    for _ in range(a.bootstrap):
        bi = g.integers(0, len(te), len(te))
        cb = np.mean([hits[t][bi].mean() for t in cross_t])
        wb = np.mean([hits[t][bi].mean() for t in within_t])
        if wb > 0:
            boots.append(cb / wb)
    boots = np.array(boots)
    res["summary"]["retention_ci"] = {
        "n_bootstrap": int(len(boots)), "mean": float(boots.mean()),
        "sd": float(boots.std()),
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5))}

    if a.save_map:
        torch.save({"Wi": Wi.state_dict(), "Wt": Wt.state_dict(),
                    "mu": {t: {m: v.cpu() for m, v in mu[t].items()} for t in tags},
                    "mu_txt": {t: mut[t].cpu() for t in tags},
                    "vendors": tags, "dim": a.dim}, a.save_map)
        print(f"\nwrote {a.save_map}")

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    ci = res["summary"]["retention_ci"]
    print(f"\ncross {c:.4f} within {w:.4f} retention {c/w:.3f} "
          f"95% CI [{ci['ci95_low']:.3f}, {ci['ci95_high']:.3f}]")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
