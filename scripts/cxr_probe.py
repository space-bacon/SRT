#!/usr/bin/env python3
"""Linear probe for the fourteen ChestX-ray14 findings on frozen hidden states.

The question is whether a general-purpose backbone, never trained on medical
imaging, already carries chest pathology linearly. The probe is deliberately
linear: anything stronger measures the probe rather than the representation.

Three controls, because a bare AUROC here is not interpretable.

  shuffled     labels permuted within the training split and refitted. Anything
               above 0.5 on a held-out set is leakage or a bug.
  view-only    the AP/PA marker used alone as the score. Portable AP films are
               taken of sicker, bedbound patients, so this is a real route to a
               high AUROC that involves no pathology at all. A finding where the
               probe fails to clear this baseline has not been detected.
  CheXNet      0.841 mean AUROC, supervised end to end on the same fourteen
               labels and the same official split. The reference, not a rival.

Splits come from the official patient-disjoint list, never invented here, since
one patient on both sides inflates everything.

    python scripts/cxr_probe.py --states /root/cxr14_gemma4.npz \
        --manifest /root/cxr14_manifest.json --out /root/cxr14_probe.json
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
            "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
            "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
CHEXNET = 0.841


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--states", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--out", default="/root/cxr14_probe.json")
    return p.parse_args()


def auroc(scores, labels):
    """Rank-based, ties averaged. Returns nan when a class is absent."""
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def boot_ci(scores, labels, n, rng):
    if n == 0:
        return None
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        a = auroc(scores[idx], labels[idx])
        if a == a:
            vals.append(a)
    if not vals:
        return None
    return {"ci95_low": round(float(np.percentile(vals, 2.5)), 4),
            "ci95_high": round(float(np.percentile(vals, 97.5)), 4)}


def fit(Xtr, Ytr, Xte, epochs, lr, wd):
    """One Linear(d, 14) under BCE is fourteen independent logistic regressions."""
    W = torch.zeros(Xtr.shape[1], Ytr.shape[1], device="cuda", requires_grad=True)
    b = torch.zeros(Ytr.shape[1], device="cuda", requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        loss = F.binary_cross_entropy_with_logits(Xtr @ W + b, Ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (Xte @ W + b).cpu().numpy()


def main():
    a = parse_args()
    rng = np.random.default_rng(0)

    rows = json.load(open(a.manifest))["rows"]
    z = np.load(a.states)
    ok = z["ok"]
    if len(rows) != len(ok):
        raise SystemExit(f"manifest has {len(rows)} rows, states have {len(ok)}")
    rows = [r for r, k in zip(rows, ok) if k]
    X = z["item"][ok].astype(np.float32)
    print(f"{len(rows)} encoded images, dim {X.shape[1]}")

    split = np.array([r["split"] for r in rows])
    tr, te = split == "train", split == "test"
    pat_tr = {r["patient_id"] for r, m in zip(rows, tr) if m}
    pat_te = {r["patient_id"] for r, m in zip(rows, te) if m}
    overlap = pat_tr & pat_te
    print(f"  train {tr.sum()}  test {te.sum()}  "
          f"patient overlap {len(overlap)} (must be 0)")
    if overlap:
        raise SystemExit("patients appear in both splits; result would be inflated")

    Y = np.zeros((len(rows), len(FINDINGS)), np.float32)
    for i, r in enumerate(rows):
        for j, f in enumerate(FINDINGS):
            if f in r["labels"]:
                Y[i, j] = 1.0
    is_ap = np.array([r["view"] == "AP" for r in rows], np.float32)

    Xt = torch.tensor(X, device="cuda")
    # Centred on the training split only, so the holdout is untouched.
    mu = Xt[torch.tensor(tr)].mean(0, keepdim=True)
    sd = Xt[torch.tensor(tr)].std(0, keepdim=True) + 1e-6
    Xt = (Xt - mu) / sd
    Xtr, Xte = Xt[torch.tensor(tr)], Xt[torch.tensor(te)]
    Ytr = torch.tensor(Y[tr], device="cuda")

    print("\nfitting probe", flush=True)
    S = fit(Xtr, Ytr, Xte, a.epochs, a.lr, a.wd)

    print("fitting shuffled-label floor", flush=True)
    Ysh = Y[tr].copy()
    for j in range(Ysh.shape[1]):
        rng.shuffle(Ysh[:, j])
    S_sh = fit(Xtr, torch.tensor(Ysh, device="cuda"), Xte, a.epochs, a.lr, a.wd)

    Yte = Y[te]
    out, aurocs = {}, []
    print(f"\n{'finding':22s} {'n_pos':>6s} {'AUROC':>7s} {'95% CI':>16s} "
          f"{'shuffled':>9s} {'view-only':>10s}")
    for j, f in enumerate(FINDINGS):
        y = Yte[:, j]
        au = auroc(S[:, j], y)
        ci = boot_ci(S[:, j], y, a.bootstrap, rng)
        sh = auroc(S_sh[:, j], y)
        vw = auroc(is_ap[te], y)
        out[f] = {"n_pos_test": int(y.sum()),
                  "prevalence": round(float(y.mean()), 4),
                  "auroc": round(au, 4), "auroc_ci": ci,
                  "shuffled_floor": round(sh, 4),
                  "view_only_baseline": round(vw, 4),
                  "beats_view": bool(au > vw)}
        aurocs.append(au)
        lo = f"[{ci['ci95_low']:.3f}, {ci['ci95_high']:.3f}]" if ci else "-"
        print(f"{f:22s} {int(y.sum()):6d} {au:7.4f} {lo:>16s} "
              f"{sh:9.4f} {vw:10.4f}")

    mean_auroc = float(np.nanmean(aurocs))
    n_beat = sum(1 for f in FINDINGS if out[f]["beats_view"])
    summary = {
        "mean_auroc": round(mean_auroc, 4),
        "mean_shuffled_floor": round(float(np.nanmean(
            [out[f]["shuffled_floor"] for f in FINDINGS])), 4),
        "mean_view_only": round(float(np.nanmean(
            [out[f]["view_only_baseline"] for f in FINDINGS])), 4),
        "findings_beating_view_only": n_beat,
        "chexnet_reference": CHEXNET,
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "note": "detection, not early detection: these labels describe what is "
                "visible in the image, so nothing here speaks to catching "
                "disease before it is apparent",
    }
    json.dump({"summary": summary, "per_finding": out,
               "states": a.states}, open(a.out, "w"), indent=2)
    print(f"\nmean AUROC {mean_auroc:.4f}   shuffled floor "
          f"{summary['mean_shuffled_floor']:.4f}   view-only "
          f"{summary['mean_view_only']:.4f}")
    print(f"{n_beat}/14 findings beat the view-only baseline")
    print(f"CheXNet, supervised end to end on this split: {CHEXNET}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
