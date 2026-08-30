#!/usr/bin/env python3
"""Price the two variances the ChestX-ray14 card does not separate, and the ones it undersold.

Dipankar Sarkar, reviewing the card:

  1. The headline has no interval. Fourteen per-finding CIs are published and
     the macro figure they average into is bare.
  2. The published interval holds ONE variance. The bootstrap resamples patients
     with the head held fixed. The shuffled floor permutes training labels and
     REFITS, which is a different and larger quantity, and it appears in no
     interval on either card.
  3. The floor is one seed. Per-finding floors scatter about 0.024 while the
     published SEs average 0.012, so a macro floor landing 0.0002 off 0.5 is
     roughly a 1-in-40 draw, not a bound.
  4. Two things cut our way and we claimed neither: the gap survives dropping
     any single finding, and weighting by positives rather than equally roughly
     doubles it.

All four are answered here from the published states, no re-encoding.

  --seeds 20 turns the floor into a distribution, and keeps the per-seed macro
  so the 12-of-14 count gets a null too, which is the number he asked for.
  The macro CI resamples patients and recomputes the macro over all fourteen
  findings inside each draw, rather than averaging fourteen separate intervals.

    python scripts/cxr_probe_floor_seeds.py --states states/cxr14_gemma4_full112k.npz
"""
import argparse
import json

import numpy as np
import torch

from cxr_probe import FINDINGS, MATCHED, auroc, fit, pick_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--states", default="artifacts/nla/omni/states/cxr14_gemma4_full112k.npz")
    p.add_argument("--manifest", default="artifacts/nla/cxr14_manifest.json")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--out", default="artifacts/nla/cxr14_floor_seeds.json")
    return p.parse_args()


def macro(S, Y):
    return float(np.nanmean([auroc(S[:, j], Y[:, j]) for j in range(Y.shape[1])]))


def main():
    a = parse_args()
    rows = json.load(open(a.manifest))["rows"]
    z = np.load(a.states)
    ok = z["ok"]
    rows = [r for r, k in zip(rows, ok) if k]
    X = z["item"][ok].astype(np.float32)

    split = np.array([r["split"] for r in rows])
    tr, te = split == "train", split == "test"
    Y = np.zeros((len(rows), len(FINDINGS)), np.float32)
    for i, r in enumerate(rows):
        for j, f in enumerate(FINDINGS):
            if f in r["labels"]:
                Y[i, j] = 1.0
    Yte = Y[te]
    pat = np.array([r["patient_id"] for r, m in zip(rows, te) if m])
    dev = pick_device()
    print(f"{len(rows)} images, dim {X.shape[1]}, device {dev}")
    print(f"  train {tr.sum()}  test {te.sum()}  patients {len(np.unique(pat))}")

    Xt = torch.tensor(X, device=dev)
    trt, tet = torch.tensor(tr, device=dev), torch.tensor(te, device=dev)
    mu, sd = Xt[trt].mean(0, keepdim=True), Xt[trt].std(0, keepdim=True) + 1e-6
    Xt = (Xt - mu) / sd
    Xtr, Xte, Ytr = Xt[trt], Xt[tet], torch.tensor(Y[tr], device=dev)

    print("\nfitting the probe", flush=True)
    S = fit(Xtr, Ytr, Xte, a.epochs, a.lr, a.wd)
    per = np.array([auroc(S[:, j], Yte[:, j]) for j in range(len(FINDINGS))])
    wang = np.array([MATCHED["per_finding"][f] for f in FINDINGS])
    npos = Yte.sum(0)
    m = float(per.mean())
    print(f"  macro {m:.6f}   Wang {wang.mean():.6f}   gap {m - wang.mean():+.6f}"
          f"   ahead on {(per > wang).sum()} of 14")

    res = {"question": "how wide is the headline, and does the floor hold as a bound",
           "credit": "every item here was raised by Dipankar Sarkar",
           "states": a.states, "macro_auroc": round(m, 6),
           "wang_macro": round(float(wang.mean()), 6),
           "gap": round(m - float(wang.mean()), 6),
           "ahead_on": int((per > wang).sum()),
           "per_finding": {f: round(float(v), 4) for f, v in zip(FINDINGS, per)}}

    # 1. a macro interval, recomputed inside each patient draw ---------------
    print("\nmacro CI, patient-clustered, macro recomputed per draw", flush=True)
    rng = np.random.default_rng(0)
    uniq = np.unique(pat)
    by = {g: np.where(pat == g)[0] for g in uniq}
    draws = []
    for _ in range(a.bootstrap):
        idx = np.concatenate([by[g] for g in rng.choice(uniq, len(uniq))])
        yb = Yte[idx]
        keep = [j for j in range(len(FINDINGS)) if 0 < yb[:, j].sum() < len(idx)]
        draws.append(np.mean([auroc(S[idx, j], yb[:, j]) for j in keep]))
    draws = np.array(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    se = float(draws.std())
    res["macro_ci"] = {"ci95": [round(float(lo), 4), round(float(hi), 4)],
                       "se": round(se, 4),
                       "gap_over_se": round((m - float(wang.mean())) / se, 2),
                       "note": "one number, not fourteen averaged"}
    print(f"  95% CI [{lo:.4f}, {hi:.4f}]   SE {se:.4f}   "
          f"gap / SE {(m - wang.mean()) / se:.2f}")

    # 2. the floor as a distribution ----------------------------------------
    print(f"\nrefitting the shuffled floor, {a.seeds} seeds", flush=True)
    fl_macro, fl_per, fl_count = [], [], []
    for s in range(a.seeds):
        r = np.random.default_rng(1000 + s)
        Ysh = Y[tr].copy()
        for j in range(Ysh.shape[1]):
            r.shuffle(Ysh[:, j])
        Ssh = fit(Xtr, torch.tensor(Ysh, device=dev), Xte, a.epochs, a.lr, a.wd)
        p = np.array([auroc(Ssh[:, j], Yte[:, j]) for j in range(len(FINDINGS))])
        fl_per.append(p)
        fl_macro.append(float(p.mean()))
        fl_count.append(int((p > wang).sum()))
        print(f"  seed {s:2d}  macro {p.mean():.4f}   beats Wang on {fl_count[-1]:2d}",
              flush=True)
    fl_macro = np.array(fl_macro)
    fl_per = np.array(fl_per)
    res["floor_seeds"] = {
        "n": a.seeds, "macro_mean": round(float(fl_macro.mean()), 4),
        "macro_sd": round(float(fl_macro.std()), 4),
        "macro_min": round(float(fl_macro.min()), 4),
        "macro_max": round(float(fl_macro.max()), 4),
        "published_single_seed": 0.5002,
        "per_finding_sd_mean": round(float(fl_per.std(0).mean()), 4),
        "gap_over_floor_sd": round((m - float(wang.mean())) / float(fl_macro.std()), 2)}
    f = res["floor_seeds"]
    print(f"\n  floor macro {f['macro_mean']:.4f} +/- {f['macro_sd']:.4f}"
          f"   range [{f['macro_min']:.4f}, {f['macro_max']:.4f}]")
    print(f"  gap / floor SD  {f['gap_over_floor_sd']}")

    # 3. the null on the count he asked for ---------------------------------
    res["count_null"] = {
        "observed": int((per > wang).sum()),
        "shuffled_counts": fl_count,
        "shuffled_mean": round(float(np.mean(fl_count)), 2),
        "shuffled_max": int(max(fl_count)),
        "coin_flip_p_ge_12_of_14": round(106 / 16384, 4),
        "note": "the shuffled refit gives the count a measured null, which the "
                "coin-flip figure only approximates and which ignores that "
                "findings co-occur within a patient"}
    print(f"\n  12-of-14 count: observed {res['count_null']['observed']}, "
          f"shuffled mean {res['count_null']['shuffled_mean']}, "
          f"shuffled max {res['count_null']['shuffled_max']}")

    # 4. the two things we undersold ----------------------------------------
    loo = {}
    for j, fnd in enumerate(FINDINGS):
        k = [i for i in range(len(FINDINGS)) if i != j]
        loo[fnd] = round(float(per[k].mean() - wang[k].mean()), 4)
    wmacro = float((per * npos).sum() / npos.sum())
    wwang = float((wang * npos).sum() / npos.sum())
    res["leave_one_out_gap"] = loo
    res["leave_one_out_range"] = [min(loo.values()), max(loo.values())]
    res["weighted_by_positives"] = {
        "ours": round(wmacro, 4), "wang": round(wwang, 4),
        "gap": round(wmacro - wwang, 4),
        "equal_weight_gap": round(m - float(wang.mean()), 4),
        "note": "macro gives Hernia's 86 positives the same vote as "
                "Infiltration's 6112; equal weighting is the conservative choice"}
    print(f"\n  leave-one-out gap range [{min(loo.values()):+.4f}, "
          f"{max(loo.values()):+.4f}]  (never flips)")
    print(f"  weighted by positives {wmacro:.4f} vs {wwang:.4f} = "
          f"{wmacro - wwang:+.4f}, against {m - wang.mean():+.4f} equal-weighted")

    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
