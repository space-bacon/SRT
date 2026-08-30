#!/usr/bin/env python3
"""Land-use scene content in frozen states, across four backbones, on satellite imagery.

The chest-radiograph result says a frozen general-purpose backbone carries
medical pathology linearly. One domain is not a finding, and today already cost
us a claim published off a single gallery. This asks the same two questions of
an unrelated domain, with states that are already on disk and no GPU:

  per-vendor AUROC   is land-use scene content linearly present in each backbone
  transported AUROC  is it the SAME direction, or four private ones

Labels are weak: scene classes are keyword-matched out of the RSICD captions,
which recovers 17 classes with at least 100 positives across 77% of the rows.
That is the same shape of supervision ChestX-ray14 itself uses, where the
fourteen findings are NLP-mined from radiology reports, so the weakness is
inherited by every model on that benchmark too. It is still weak, and the
headline number should never be quoted without this sentence.

The label comes from the caption and the probe reads the IMAGE tower, so the
text side of each vendor is not involved and cannot leak the answer.

    python scripts/rsicd_scene_probe.py --domain rsicd
"""
import argparse
import itertools
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

from cxr_probe import auroc

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAGS = ["gemma4", "aria", "mistral", "qwen3omni"]

# Keyword to scene class. Deliberately plain: a cleverer extractor would be
# harder to audit and the point is that the labels are weak and legible.
CLASSES = {
    "airport": r"\bairport|airplane|aircraft|runway", "beach": r"\bbeach\b",
    "bridge": r"\bbridge", "farmland": r"\bfarmland|\bfarm\b|cropland|\bfields?\b",
    "forest": r"\bforest|\bwoods\b", "harbor": r"\bharbou?r|\bport\b|\bdock",
    "parking": r"\bparking", "playground": r"\bplayground", "pond": r"\bpond\b",
    "river": r"\briver", "stadium": r"\bstadium", "viaduct": r"\bviaduct",
    "residential": r"\bresidential|\bhouses\b", "industrial": r"\bindustrial",
    "mountain": r"\bmountain", "desert": r"\bdesert", "church": r"\bchurch",
    "railway": r"\brailway|\brailroad|\btrain station",
    "meadow": r"\bmeadow|\bgrassland", "bareland": r"\bbare land|\bbareland",
}
MIN_POS = 100


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="artifacts/nla/omni")
    p.add_argument("--domain", default="rsicd")
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--out", default=None)
    return p.parse_args()


def fit_probe(Xtr, Ytr, epochs, lr, wd):
    W = torch.zeros(Xtr.shape[1], Ytr.shape[1], device=DEV, requires_grad=True)
    b = torch.zeros(Ytr.shape[1], device=DEV, requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        loss = F.binary_cross_entropy_with_logits(Xtr @ W + b, Ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W.detach(), b.detach()


def fit_map(X, Y, alpha):
    G = X.T @ X
    lam = alpha * torch.diagonal(G).mean()
    return torch.linalg.solve(
        G + lam * torch.eye(G.shape[0], device=G.device, dtype=G.dtype), X.T @ Y)


def mean_auroc(S, Yte, names):
    v = [auroc(S[:, j], Yte[:, j]) for j in range(len(names))]
    return float(np.nanmean(v)), v


def main():
    a = parse_args()
    a.out = a.out or os.path.join(a.dir, f"{a.domain}_scene_probe.json")
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
                "key": np.array([r["key"] for r, k in zip(rows, ok) if k]),
                "cap": [r["caption"] for r, k in zip(rows, ok) if k]}
    tags = [t for t in TAGS if t in V]
    shared = set(V[tags[0]]["key"])
    for t in tags[1:]:
        shared &= set(V[t]["key"])
    order = [k for k in V[tags[0]]["key"] if k in shared]
    n = len(order)

    cap = dict(zip(V[tags[0]]["key"], V[tags[0]]["cap"]))
    lab = {c: np.array([1.0 if re.search(p, cap[k].lower()) else 0.0 for k in order])
           for c, p in CLASSES.items()}
    names = sorted(c for c in CLASSES if lab[c].sum() >= MIN_POS)
    Y = np.stack([lab[c] for c in names], 1).astype(np.float32)
    print(f"  {n} images, {len(names)} classes with >= {MIN_POS} positives")

    perm = np.random.default_rng(0).permutation(n)
    n_te = int(a.holdout_frac * n)
    te = torch.tensor(perm[:n_te]).to(DEV)
    tr = torch.tensor(perm[n_te:]).to(DEV)
    Yte, Ytr = Y[perm[:n_te]], torch.tensor(Y[perm[n_te:]]).to(DEV)

    X = {}
    for t in tags:
        pos = {k: i for i, k in enumerate(V[t]["key"])}
        idx = np.array([pos[k] for k in order])
        M = torch.tensor(V[t]["img"][idx].astype(np.float32)).to(DEV).double()
        mu, sd = M[tr].mean(0, keepdim=True), M[tr].std(0, keepdim=True) + 1e-6
        X[t] = (M - mu) / sd

    res = {"question": "is land-use scene content linearly present in every "
                       "backbone, and is it the same direction",
           "domain": a.domain, "vendors": tags, "classes": names,
           "n_train": int(len(tr)), "n_test": int(n_te),
           "label_source": "keyword match on RSICD captions, weak supervision, "
                           "same shape as ChestX-ray14's NLP-mined findings",
           "native": {}, "transported": {}, "self_map": {}, "shuffled": {}}

    P = {}
    print(f"\n{'vendor':<12}{'mean AUROC':>11}")
    for t in tags:
        W, b = fit_probe(X[t][tr].float(), Ytr, a.epochs, a.lr, a.wd)
        P[t] = (W.double(), b.double())
        m, per = mean_auroc((X[t][te] @ P[t][0] + P[t][1]).cpu().numpy(), Yte, names)
        res["native"][t] = {"mean_auroc": round(m, 4),
                            "per_class": {c: round(v, 4) for c, v in zip(names, per)}}
        print(f"{t:<12}{m:11.4f}")

    M = {}
    for c, aa in itertools.product(tags, repeat=2):
        M[(c, aa)] = fit_map(X[c][tr], X[aa][tr], a.ridge)

    print(f"\n{'probe from':<12}{'read on':<12}{'mean AUROC':>11}{'vs native':>11}")
    for aa in tags:
        W, b = P[aa]
        ms, _ = mean_auroc((X[aa][te] @ M[(aa, aa)] @ W + b).cpu().numpy(), Yte, names)
        res["self_map"][aa] = round(ms, 4)
        for c in tags:
            if c == aa:
                continue
            m, _ = mean_auroc((X[c][te] @ M[(c, aa)] @ W + b).cpu().numpy(), Yte, names)
            res["transported"][f"{aa} probe on {c}"] = {
                "mean_auroc": round(m, 4),
                "vs_native_target": round(m - res["native"][c]["mean_auroc"], 4)}
            print(f"{aa:<12}{c:<12}{m:11.4f}"
                  f"{m - res['native'][c]['mean_auroc']:+11.4f}")

    rng = np.random.default_rng(0)
    Ysh = Y[perm[n_te:]].copy()
    for j in range(Ysh.shape[1]):
        rng.shuffle(Ysh[:, j])
    aa = tags[0]
    Wsh, bsh = fit_probe(X[aa][tr].float(), torch.tensor(Ysh).to(DEV),
                         a.epochs, a.lr, a.wd)
    for c in tags:
        m, _ = mean_auroc(
            (X[c][te] @ M[(c, aa)] @ Wsh.double() + bsh.double()).cpu().numpy(),
            Yte, names)
        res["shuffled"][f"{aa} shuffled probe on {c}"] = round(m, 4)

    nv = [v["mean_auroc"] for v in res["native"].values()]
    tv = [v["mean_auroc"] for v in res["transported"].values()]
    res["summary"] = {
        "mean_native": round(float(np.mean(nv)), 4),
        "mean_self_map": round(float(np.mean(list(res["self_map"].values()))), 4),
        "mean_transported": round(float(np.mean(tv)), 4),
        "transport_cost": round(float(np.mean(nv) - np.mean(tv)), 4),
        "mean_shuffled": round(float(np.mean(list(res["shuffled"].values()))), 4),
        "vendor_spread": round(float(max(nv) - min(nv)), 4),
        "reading": "transported near native means one probe reads every backbone "
                   "and the scene direction is shared. transported near the "
                   "shuffled floor means each backbone carries scene content in "
                   "its own private direction."}
    json.dump(res, open(a.out, "w"), indent=2)
    s = res["summary"]
    print(f"\nnative       {s['mean_native']}   spread across vendors {s['vendor_spread']}")
    print(f"self-map     {s['mean_self_map']}")
    print(f"transported  {s['mean_transported']}   cost {s['transport_cost']}")
    print(f"shuffled     {s['mean_shuffled']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
