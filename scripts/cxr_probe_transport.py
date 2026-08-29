#!/usr/bin/env python3
"""Train the pathology probe on one backbone, read it on another, and round the loop.

Running a probe separately per vendor answers a weak question: does each model
carry chest pathology somewhere. Four independently fitted probes can each score
0.75 while pointing in four unrelated directions, and that would not be one
finding, it would be four.

The stronger question is whether it is the SAME direction. So the probe is
fitted once on vendor A, and then read on vendor C's states after a ridge map
carries them into A's space. If a probe that has never seen C's encoder still
scores on C's radiographs, the pathology direction survives transport and
"train once, read everywhere" is a measurement rather than a slogan.

Then the loop, because a single hop is the measurement that could not resolve
anything in the retrieval work. Carry a state all the way around the vendor
cycle back to A and read it with A's own probe. What degrades, and how fast,
is the part a one-hop test cannot see.

Every map is fitted on the official TRAIN rows only, and every AUROC is on the
official test split, so no patient crosses the line at any stage.

Controls, in the same shape as the single-vendor probe:
  shuffled   labels permuted before fitting the probe, refit, same transport
  self-map   A -> A through a fitted ridge, which carries the shrinkage cost
             without crossing a vendor boundary

    python scripts/cxr_probe_transport.py --states-dir /root
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from cxr_probe import FINDINGS, auroc

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAGS = ["gemma4", "aria", "mistral", "qwen3omni"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--states-dir", default="/root")
    p.add_argument("--manifest", default="/root/cxr14_manifest.json")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--out", default="/root/cxr14_transport.json")
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


def mean_auroc(S, Yte):
    v = [auroc(S[:, j], Yte[:, j]) for j in range(len(FINDINGS))]
    return float(np.nanmean(v)), v


def main():
    a = parse_args()
    rows = json.load(open(a.manifest))["rows"]

    have = [t for t in TAGS
            if os.path.exists(os.path.join(a.states_dir, f"cxr14_{t}.npz"))]
    if len(have) < 2:
        raise SystemExit(f"need at least two vendors, found {have}")
    print(f"device {DEV}   vendors {have}")

    X, ok_ref = {}, None
    for t in have:
        z = np.load(os.path.join(a.states_dir, f"cxr14_{t}.npz"))
        ok = z["ok"]
        if ok_ref is None:
            ok_ref = ok
        elif not np.array_equal(ok, ok_ref):
            # Different drops mean row i is a different radiograph per vendor.
            raise SystemExit(f"{t}: ok mask differs from {have[0]}; rows would "
                             f"not correspond and every number would be wrong")
        X[t] = torch.tensor(z["item"][ok].astype(np.float32)).to(DEV).double()
        print(f"  {t:<10} {X[t].shape}")

    kept = [r for r, k in zip(rows, ok_ref) if k]
    split = np.array([r["split"] for r in kept])
    tr = torch.tensor(np.where(split == "train")[0]).to(DEV)
    te = torch.tensor(np.where(split == "test")[0]).to(DEV)
    pat_tr = {kept[i]["patient_id"] for i in tr.tolist()}
    pat_te = {kept[i]["patient_id"] for i in te.tolist()}
    if pat_tr & pat_te:
        raise SystemExit("patients on both sides of the split")
    print(f"  train {len(tr)}  test {len(te)}  patient overlap 0")

    Y = np.zeros((len(kept), len(FINDINGS)), np.float32)
    for i, r in enumerate(kept):
        for j, f in enumerate(FINDINGS):
            if f in r["labels"]:
                Y[i, j] = 1.0
    Yte = Y[te.cpu().numpy()]
    Ytr = torch.tensor(Y[tr.cpu().numpy()]).to(DEV)

    # Standardise inside each vendor on train rows, then keep that frame: the
    # probe is fitted in it and every transported state must arrive in it too.
    st = {}
    for t in have:
        mu = X[t][tr].mean(0, keepdim=True)
        sd = X[t][tr].std(0, keepdim=True) + 1e-6
        X[t] = (X[t] - mu) / sd
        st[t] = (mu, sd)

    print("\nfitting one probe per vendor")
    P = {}
    for t in have:
        W, b = fit_probe(X[t][tr].float(), Ytr, a.epochs, a.lr, a.wd)
        P[t] = (W.double(), b.double())
        m, _ = mean_auroc((X[t][te] @ P[t][0] + P[t][1]).cpu().numpy(), Yte)
        print(f"  {t:<10} native mean AUROC {m:.4f}")

    print("\nfitting maps between vendors, train rows only")
    M = {}
    for c, aa in itertools.product(have, repeat=2):
        M[(c, aa)] = fit_map(X[c][tr], X[aa][tr], a.ridge)

    res = {"question": "is it the same direction in every backbone, and does it "
                       "survive a trip around the vendor cycle",
           "vendors": have, "n_train": len(tr), "n_test": len(te),
           "native": {}, "transported": {}, "self_map": {}, "cycle": {},
           "shuffled": {}}
    for t in have:
        m, per = mean_auroc((X[t][te] @ P[t][0] + P[t][1]).cpu().numpy(), Yte)
        res["native"][t] = {"mean_auroc": round(m, 4),
                            "per_finding": {f: round(v, 4)
                                            for f, v in zip(FINDINGS, per)}}

    print(f"\n{'probe trained on':<18}{'read on':<12}{'mean AUROC':>11}"
          f"{'vs native':>11}")
    for aa in have:
        W, b = P[aa]
        m_self, _ = mean_auroc((X[aa][te] @ M[(aa, aa)] @ W + b).cpu().numpy(), Yte)
        res["self_map"][aa] = round(m_self, 4)
        print(f"{aa:<18}{'itself (map)':<12}{m_self:11.4f}"
              f"{m_self - res['native'][aa]['mean_auroc']:+11.4f}")
        for c in have:
            if c == aa:
                continue
            m, per = mean_auroc((X[c][te] @ M[(c, aa)] @ W + b).cpu().numpy(), Yte)
            res["transported"][f"{aa} probe on {c}"] = {
                "mean_auroc": round(m, 4),
                "vs_native_target": round(m - res["native"][c]["mean_auroc"], 4),
                "per_finding": {f: round(v, 4) for f, v in zip(FINDINGS, per)}}
            print(f"{aa:<18}{c:<12}{m:11.4f}"
                  f"{m - res['native'][c]['mean_auroc']:+11.4f}")

    if len(have) >= 3:
        print(f"\n{'cycle':<44}{'mean AUROC':>11}{'vs native':>11}")
        for start in have:
            cyc = [start] + [t for t in have if t != start]
            T = X[start][te]
            for i in range(len(cyc)):
                T = T @ M[(cyc[i], cyc[(i + 1) % len(cyc)])]
            W, b = P[start]
            m, _ = mean_auroc((T @ W + b).cpu().numpy(), Yte)
            name = " -> ".join(cyc) + f" -> {start}"
            res["cycle"][name] = round(m, 4)
            print(f"{name:<44}{m:11.4f}"
                  f"{m - res['native'][start]['mean_auroc']:+11.4f}")

    print("\nshuffled-label floor, same transport")
    rng = np.random.default_rng(0)
    Ysh = Y[tr.cpu().numpy()].copy()
    for j in range(Ysh.shape[1]):
        rng.shuffle(Ysh[:, j])
    aa = have[0]
    Wsh, bsh = fit_probe(X[aa][tr].float(), torch.tensor(Ysh).to(DEV),
                         a.epochs, a.lr, a.wd)
    for c in have:
        m, _ = mean_auroc(
            (X[c][te] @ M[(c, aa)] @ Wsh.double() + bsh.double()).cpu().numpy(), Yte)
        res["shuffled"][f"{aa} shuffled probe on {c}"] = round(m, 4)
        print(f"  {aa} shuffled probe on {c:<12} {m:.4f}")

    tv = [v["mean_auroc"] for v in res["transported"].values()]
    nv = [v["mean_auroc"] for v in res["native"].values()]
    res["summary"] = {
        "mean_native": round(float(np.mean(nv)), 4),
        "mean_transported": round(float(np.mean(tv)), 4) if tv else None,
        "transport_cost": round(float(np.mean(nv) - np.mean(tv)), 4) if tv else None,
        "mean_self_map": round(float(np.mean(list(res["self_map"].values()))), 4),
        "mean_cycle": (round(float(np.mean(list(res["cycle"].values()))), 4)
                       if res["cycle"] else None),
        "mean_shuffled": round(float(np.mean(list(res["shuffled"].values()))), 4),
        "reading": "transported near native means the pathology direction is "
                   "shared and a probe fitted once reads every backbone. "
                   "transported near the shuffled floor means each backbone "
                   "carries pathology in its own private direction and four "
                   "probes are four findings, not one."}
    json.dump(res, open(a.out, "w"), indent=2)
    s = res["summary"]
    print(f"\nnative      {s['mean_native']}")
    print(f"self-map    {s['mean_self_map']}")
    print(f"transported {s['mean_transported']}   cost {s['transport_cost']}")
    print(f"cycle       {s['mean_cycle']}")
    print(f"shuffled    {s['mean_shuffled']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
