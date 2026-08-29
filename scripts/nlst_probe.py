#!/usr/bin/env python3
"""Does a frozen backbone separate lesion tissue from healthy tissue, in one patient?

This subset carries no outcomes, so early detection is not on the table. The
question that is answerable without labels is narrower and still worth asking:
within the same participant and the same scanner, do the hidden states of slices
containing a segmented lesion differ from slices that do not.

The confound here is height in the chest. Lesion slices sit mid-lung and the
controls are drawn far away, so a probe can score well by reading anatomical
position and never touching the lesion. Slice position is therefore reported as
a baseline in its own right, exactly as the view marker was for radiographs. A
probe that fails to clear it has demonstrated nothing.

Two further guards. The split is participant-disjoint. And a paired within-scan
test asks whether lesion slices outrank control slices from the SAME study,
which removes participant and scanner effects entirely.

    python scripts/nlst_probe.py --states /root/nlst_gemma4.npz \
        --manifest /root/nlst_manifest.json --out /root/nlst_probe.json
"""
import argparse
import collections
import json

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--states", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--out", default="/root/nlst_probe.json")
    return p.parse_args()


def auroc(scores, labels):
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main():
    a = parse_args()
    rng = np.random.default_rng(0)

    rows = json.load(open(a.manifest))["rows"]
    z = np.load(a.states)
    ok = z["ok"]
    rows = [r for r, k in zip(rows, ok) if k]
    X = z["item"][ok].astype(np.float32)
    y = np.array([1.0 if r["has_lesion"] else 0.0 for r in rows], np.float32)
    pat = np.array([r["patient"] for r in rows])
    zpos = np.array([r["z"] for r in rows], np.float32)
    print(f"{len(rows)} slices, {int(y.sum())} lesion, {len(y)-int(y.sum())} control, "
          f"{len(set(pat))} participants, dim {X.shape[1]}")

    pats = sorted(set(pat))
    rng.shuffle(pats)
    te_pats = set(pats[:max(1, len(pats) // 3)])
    te = np.array([p in te_pats for p in pat])
    tr = ~te
    print(f"  participant-disjoint split: {tr.sum()} train / {te.sum()} test "
          f"({len(pats)-len(te_pats)}/{len(te_pats)} participants)")

    Xt = torch.tensor(X, device="cuda")
    mu = Xt[torch.tensor(tr)].mean(0, keepdim=True)
    sd = Xt[torch.tensor(tr)].std(0, keepdim=True) + 1e-6
    Xt = (Xt - mu) / sd

    def fit(labels):
        W = torch.zeros(X.shape[1], 1, device="cuda", requires_grad=True)
        b = torch.zeros(1, device="cuda", requires_grad=True)
        opt = torch.optim.AdamW([W, b], lr=a.lr, weight_decay=a.wd)
        Ytr = torch.tensor(labels[tr], device="cuda").unsqueeze(1)
        Xtr = Xt[torch.tensor(tr)]
        for _ in range(a.epochs):
            loss = F.binary_cross_entropy_with_logits(Xtr @ W + b, Ytr)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            return (Xt @ W + b).squeeze(1).cpu().numpy()

    s = fit(y)
    ysh = y[tr].copy(); rng.shuffle(ysh)
    y_shuffled = y.copy(); y_shuffled[tr] = ysh
    s_sh = fit(y_shuffled)

    au = auroc(s[te], y[te])
    boot = []
    for _ in range(a.bootstrap):
        idx = rng.integers(0, te.sum(), te.sum())
        v = auroc(s[te][idx], y[te][idx])
        if v == v:
            boot.append(v)
    ci = {"ci95_low": round(float(np.percentile(boot, 2.5)), 4),
          "ci95_high": round(float(np.percentile(boot, 97.5)), 4)} if boot else None

    # Position alone, and position alone the other way up, since either sign counts.
    au_z = auroc(zpos[te], y[te])
    au_z = max(au_z, 1 - au_z)
    au_sh = auroc(s_sh[te], y[te])

    # Paired within a single study: does the lesion slice outrank the control
    # slice from the same scan of the same participant on the same day.
    by_study = collections.defaultdict(lambda: {"les": [], "ctl": []})
    for i, r in enumerate(rows):
        if not te[i]:
            continue
        by_study[(r["patient"], r["date"])]["les" if r["has_lesion"] else "ctl"].append(s[i])
    diffs = [np.mean(d["les"]) - np.mean(d["ctl"])
             for d in by_study.values() if d["les"] and d["ctl"]]
    n_pos = sum(1 for d in diffs if d > 0)

    traj = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r["has_lesion"]:
            traj[r["timepoint"]].append(float(s[i]))

    out = {
        "auroc": round(au, 4), "auroc_ci": ci,
        "position_only_baseline": round(au_z, 4),
        "shuffled_floor": round(au_sh, 4),
        "beats_position": bool(au > au_z),
        "paired_within_study": {
            "n_studies": len(diffs),
            "n_lesion_higher": int(n_pos),
            "mean_margin": round(float(np.mean(diffs)), 4) if diffs else None,
        },
        "lesion_score_by_timepoint": {
            str(t): round(float(np.mean(v)), 4) for t, v in sorted(traj.items())},
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "n_participants": len(pats),
        "note": "no outcomes in this subset: lesion-vs-healthy discrimination, "
                "not early detection",
    }
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\n  probe AUROC        {au:.4f}"
          + (f"  CI [{ci['ci95_low']:.3f}, {ci['ci95_high']:.3f}]" if ci else ""))
    print(f"  position-only      {au_z:.4f}   <- must be beaten")
    print(f"  shuffled floor     {au_sh:.4f}")
    print(f"  paired within scan {n_pos}/{len(diffs)} studies rank lesion higher")
    print(f"  by timepoint       {out['lesion_score_by_timepoint']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
