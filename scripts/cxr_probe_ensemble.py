#!/usr/bin/env python3
"""Recover the split-matched lead by reading several backbones together.

Gemma-4 scores 0.7590 on the official ChestX-ray14 split, ahead of Wang's
0.7451. Aria, identical probe and split, scores 0.7080, which is behind. So
"a frozen backbone beats the dataset authors" is currently a statement about
one backbone, and that is a weaker claim than it looked this morning.

The joint-frame run showed three vendors read together beat the best single
vendor on all four targets in both domains, eight for eight. If that holds for
pathology rather than retrieval, pooling backbones should recover the lead and
extend it, using states that are already on disk.

Three ways of combining, in increasing order of how much they could flatter us:

  logit mean    average the per-vendor probe scores. Adds ZERO parameters. If
                this wins, the gain cannot be capacity.
  concat        one probe over concatenated features. Adds parameters.
  concat, same  THE CONTROL. Concatenate a vendor with ITSELF. Identical
                parameter count to concat, zero new information. Any gain here
                is capacity, and must be subtracted from the concat reading.

Without the duplicate control, a concat win says nothing: a wider linear model
on the same split can improve for reasons that have nothing to do with the
second backbone. The paired patient-clustered bootstrap answers whether the
remaining gain is real.

    python scripts/cxr_probe_ensemble.py \
        --states gemma4=/root/cxr14_gemma4.npz aria=/root/cxr14_aria.npz \
        --manifest /root/cxr14_manifest.json --out /root/cxr14_ensemble.json
"""
import argparse
import itertools
import json

import numpy as np
import torch

from cxr_probe import FINDINGS, MATCHED, auroc, fit, pick_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--states", nargs="+", required=True, metavar="TAG=PATH")
    p.add_argument("--manifest", required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--device", default=None)
    p.add_argument("--save-probe", help="write per-vendor weights, biases and the "
                                       "train-split normalisation, without which "
                                       "the pooled score cannot be reproduced")
    p.add_argument("--out", default="/root/cxr14_ensemble.json")
    return p.parse_args()


def mean_auroc(S, Yte):
    return float(np.nanmean([auroc(S[:, j], Yte[:, j])
                             for j in range(S.shape[1])]))


def paired_ci(Sa, Sb, Yte, groups, n, rng):
    """Patient-clustered bootstrap on the DIFFERENCE of two mean AUROCs."""
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    d = []
    for _ in range(n):
        pick = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq))])
        ya = Yte[pick]
        keep = [j for j in range(ya.shape[1]) if 0 < ya[:, j].sum() < len(pick)]
        if not keep:
            continue
        d.append(np.mean([auroc(Sa[pick, j], ya[:, j]) for j in keep])
                 - np.mean([auroc(Sb[pick, j], ya[:, j]) for j in keep]))
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
            float(np.mean(np.array(d) > 0)))


def main():
    a = parse_args()
    rng = np.random.default_rng(0)
    tags = [s.split("=", 1)[0] for s in a.states]
    paths = [s.split("=", 1)[1] for s in a.states]

    rows_all = json.load(open(a.manifest))["rows"]
    Z, ok0 = {}, None
    for t, p in zip(tags, paths):
        z = np.load(p)
        if len(rows_all) != len(z["ok"]):
            raise SystemExit(f"{t}: manifest {len(rows_all)}, states {len(z['ok'])}")
        if ok0 is None:
            ok0 = z["ok"]
        elif not np.array_equal(ok0, z["ok"]):
            raise SystemExit(f"{t}: encodes a different subset; rows would not align")
        Z[t] = z["item"][ok0].astype(np.float32)
    rows = [r for r, k in zip(rows_all, ok0) if k]
    print(f"{len(rows)} images encoded by all of {tags}")
    for t in tags:
        print(f"  {t:12} dim {Z[t].shape[1]}")

    split = np.array([r["split"] for r in rows])
    tr, te = split == "train", split == "test"
    if {r["patient_id"] for r, m in zip(rows, tr) if m} & \
       {r["patient_id"] for r, m in zip(rows, te) if m}:
        raise SystemExit("patients appear in both splits; result would be inflated")
    print(f"  train {tr.sum()}  test {te.sum()}  official test_list.txt")

    Y = np.zeros((len(rows), len(FINDINGS)), np.float32)
    for i, r in enumerate(rows):
        for j, f in enumerate(FINDINGS):
            if f in r["labels"]:
                Y[i, j] = 1.0
    Yte = Y[te]
    groups = np.array([r["patient_id"] for r, m in zip(rows, te) if m])

    dev = a.device or pick_device()
    print(f"  device: {dev}")
    trt, tet = torch.tensor(tr, device=dev), torch.tensor(te, device=dev)
    Ytr = torch.tensor(Y[tr], device=dev)

    X, NORM = {}, {}
    for t in tags:
        M = torch.tensor(Z[t], device=dev)
        mu, sd = M[trt].mean(0, keepdim=True), M[trt].std(0, keepdim=True) + 1e-6
        X[t] = (M - mu) / sd
        NORM[t] = (mu.cpu(), sd.cpu())
        del M

    res = {"question": "does reading several frozen backbones together recover "
                       "and extend the split-matched lead",
           "vendors": tags, "split": "official test_list.txt",
           "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "split_matched_baseline": MATCHED["mean_auroc"],
           "single": {}, "ensemble": {}, "control": {}}

    S, WB = {}, {}
    print("\nfitting per-vendor probes", flush=True)
    for t in tags:
        S[t], W, b = fit(X[t][trt], Ytr, X[t][tet], a.epochs, a.lr, a.wd,
                         return_weights=True)
        WB[t] = (W, b)
        m = mean_auroc(S[t], Yte)
        res["single"][t] = round(m, 4)
        print(f"  {t:12} {m:.4f}   vs Wang {m - MATCHED['mean_auroc']:+.4f}")

    best = max(res["single"], key=res["single"].get)

    # Zero added parameters, so a win here cannot be explained by capacity.
    print("\nfitting ensembles", flush=True)
    Sm = np.mean([S[t] for t in tags], 0)
    res["ensemble"]["logit_mean"] = round(mean_auroc(Sm, Yte), 4)
    print(f"  logit mean   {res['ensemble']['logit_mean']:.4f}", flush=True)

    Sc = fit(torch.cat([X[t][trt] for t in tags], 1), Ytr,
             torch.cat([X[t][tet] for t in tags], 1), a.epochs, a.lr, a.wd)
    res["ensemble"]["concat"] = round(mean_auroc(Sc, Yte), 4)
    print(f"  concat       {res['ensemble']['concat']:.4f}", flush=True)

    Sd = fit(torch.cat([X[best][trt]] * len(tags), 1), Ytr,
             torch.cat([X[best][tet]] * len(tags), 1), a.epochs, a.lr, a.wd)
    res["control"][f"concat_{best}_with_itself"] = round(mean_auroc(Sd, Yte), 4)
    print(f"  control      {res['control'][f'concat_{best}_with_itself']:.4f}",
          flush=True)
    res["control"]["capacity_effect"] = round(
        mean_auroc(Sd, Yte) - res["single"][best], 4)
    res["control"]["concat_gain_net_of_capacity"] = round(
        mean_auroc(Sc, Yte) - mean_auroc(Sd, Yte), 4)

    print(f"\n{'variant':<28}{'mean AUROC':>11}{'vs best single':>15}{'vs Wang':>9}")
    tbl = [(f"{t} alone", res['single'][t]) for t in tags]
    tbl += [("logit mean (0 new params)", res["ensemble"]["logit_mean"]),
            ("concat", res["ensemble"]["concat"]),
            (f"CONTROL {best}+{best}", res["control"][f"concat_{best}_with_itself"])]
    for name, v in tbl:
        print(f"{name:<28}{v:11.4f}{v - res['single'][best]:+15.4f}"
              f"{v - MATCHED['mean_auroc']:+9.4f}")

    print("\npaired patient-clustered bootstrap vs best single vendor", flush=True)
    for name, Sx in [("logit_mean", Sm), ("concat", Sc)]:
        lo, hi, pw = paired_ci(Sx, S[best], Yte, groups, a.bootstrap, rng)
        res["ensemble"][f"{name}_vs_{best}"] = {
            "delta": round(mean_auroc(Sx, Yte) - res["single"][best], 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "frac_bootstrap_positive": round(pw, 4),
            "significant": bool(lo > 0)}
        d = res["ensemble"][f"{name}_vs_{best}"]
        print(f"  {name:<12} {d['delta']:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"{'significant' if lo > 0 else 'not significant'}")

    head = max(res["ensemble"]["logit_mean"], res["ensemble"]["concat"])
    res["summary"] = {
        "best_single": res["single"][best], "best_single_vendor": best,
        "best_ensemble": head,
        "vs_split_matched": round(head - MATCHED["mean_auroc"], 4),
        "recovers_lead": bool(head > MATCHED["mean_auroc"]),
        "reading": "logit mean adds no parameters, so a gain there is "
                   "complementary information between backbones. any concat "
                   "gain must be read net of the duplicate-vendor control."}
    json.dump(res, open(a.out, "w"), indent=2)
    if a.save_probe:
        # Each vendor is standardised with ITS OWN train-split statistics, so the
        # normalisation ships with the weights or the pooled score is not
        # reproducible from raw states.
        torch.save({"findings": FINDINGS, "vendors": tags,
                    "rule": "standardise per vendor, apply that vendor's linear "
                            "probe, average the logits across vendors",
                    "probes": {t: {"weight": WB[t][0], "bias": WB[t][1],
                                   "mu": NORM[t][0], "sd": NORM[t][1]}
                               for t in tags},
                    "metrics": res["summary"], "split": res["split"]},
                   a.save_probe)
        print(f"wrote {a.save_probe}")
    s = res["summary"]
    print(f"\nbest single   {s['best_single']} ({best})")
    print(f"best ensemble {s['best_ensemble']}   vs Wang {s['vs_split_matched']:+}")
    print(f"capacity effect from the duplicate control "
          f"{res['control']['capacity_effect']:+}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
