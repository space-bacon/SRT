#!/usr/bin/env python3
"""Probe every (layer, pooling) variant and report which one the findings prefer.

The hypothesis is that mean-pooling 256 image tokens is what caps the focal
findings. If it is right, max and top16 should lift Nodule and Mass while
leaving Cardiomegaly and Emphysema roughly where they are, since a diffuse
finding was never being diluted in the first place.

Reuses the probe and the controls from cxr_probe so the numbers stay comparable:
same official patient-disjoint split, same shuffled floor, same folded view
baseline.

    python scripts/cxr_pool_sweep.py --states /root/cxr14_pool_gemma4.npz \
        --manifest /root/cxr14_manifest.json --out /root/cxr14_pool_sweep.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cxr_probe import FINDINGS, auroc, fit  # noqa: E402

FOCAL = ["Nodule", "Mass", "Pneumonia"]
DIFFUSE = ["Cardiomegaly", "Emphysema", "Effusion", "Edema"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--states", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--out", default="/root/cxr14_pool_sweep.json")
    return p.parse_args()


def main():
    a = parse_args()
    rows = json.load(open(a.manifest))["rows"]
    z = np.load(a.states, allow_pickle=True)
    ok = z["ok"]
    rows = [r for r, k in zip(rows, ok) if k]
    X = z["item"][ok]                      # (n, n_layer, n_pool, d)
    layers = [int(x) for x in z["layers"]]
    pools = [str(x) for x in z["pools"]]
    print(f"{len(rows)} images, layers {layers}, pools {pools}, dim {X.shape[-1]}")

    split = np.array([r["split"] for r in rows])
    tr, te = split == "train", split == "test"
    if {r["patient_id"] for r, m in zip(rows, tr) if m} & \
       {r["patient_id"] for r, m in zip(rows, te) if m}:
        raise SystemExit("patients on both sides of the split")

    Y = np.zeros((len(rows), len(FINDINGS)), np.float32)
    for i, r in enumerate(rows):
        for j, f in enumerate(FINDINGS):
            if f in r["labels"]:
                Y[i, j] = 1.0
    is_ap = np.array([r["view"] == "AP" for r in rows], np.float32)
    Yte = Y[te]
    view = {}
    for j, f in enumerate(FINDINGS):
        v = auroc(is_ap[te], Yte[:, j])
        view[f] = max(v, 1 - v)

    results = {}
    print(f"\n{'layer':>6s} {'pool':>6s} {'mean':>7s} {'focal':>7s} {'diffuse':>8s}"
          f"  {'Nodule':>7s} {'Mass':>7s} {'Cardio':>7s}")
    for li, L in enumerate(layers):
        for pi, pool in enumerate(pools):
            Xt = torch.tensor(X[:, li, pi].astype(np.float32), device="cuda")
            mu = Xt[torch.tensor(tr)].mean(0, keepdim=True)
            sd = Xt[torch.tensor(tr)].std(0, keepdim=True) + 1e-6
            Xt = (Xt - mu) / sd
            S = fit(Xt[torch.tensor(tr)], torch.tensor(Y[tr], device="cuda"),
                    Xt[torch.tensor(te)], a.epochs, a.lr, a.wd)
            per = {f: round(auroc(S[:, j], Yte[:, j]), 4)
                   for j, f in enumerate(FINDINGS)}
            beat = sum(1 for f in FINDINGS if per[f] > view[f])
            key = f"L{L}_{pool}"
            results[key] = {"layer": L, "pool": pool, "per_finding": per,
                            "mean_auroc": round(float(np.mean(list(per.values()))), 4),
                            "focal_mean": round(float(np.mean([per[f] for f in FOCAL])), 4),
                            "diffuse_mean": round(float(np.mean([per[f] for f in DIFFUSE])), 4),
                            "beats_view": beat}
            r = results[key]
            print(f"{L:6d} {pool:>6s} {r['mean_auroc']:7.4f} {r['focal_mean']:7.4f} "
                  f"{r['diffuse_mean']:8.4f}  {per['Nodule']:7.4f} {per['Mass']:7.4f} "
                  f"{per['Cardiomegaly']:7.4f}")

    best = max(results, key=lambda k: results[k]["mean_auroc"])
    best_focal = max(results, key=lambda k: results[k]["focal_mean"])
    json.dump({"view_baseline": {k: round(v, 4) for k, v in view.items()},
               "variants": results, "best_mean": best,
               "best_focal": best_focal}, open(a.out, "w"), indent=2)
    print(f"\nbest mean AUROC   {best}  {results[best]['mean_auroc']:.4f}")
    print(f"best on focal     {best_focal}  {results[best_focal]['focal_mean']:.4f}")
    print(f"  (mean-pool baseline was 0.7179 on the 35k pilot)")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
