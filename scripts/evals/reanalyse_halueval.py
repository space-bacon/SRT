"""Re-analyse the saved HaluEval features CSV with corrected sign
conventions and length-matched evaluation.

This is a no-inference, CSV-only analysis: load → resign → AUC by feature,
NLA-only logistic, length-matched subset, and length-bucketed analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def auc_directional(y: np.ndarray, x: np.ndarray) -> tuple[float, str]:
    """Return AUC of (x, y) using whichever sign makes AUC ≥ 0.5,
    and a marker indicating the direction."""
    a = roc_auc_score(y, x)
    if a >= 0.5:
        return a, "↑→hallu"
    return 1.0 - a, "↓→hallu"


def cv_auc(X: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=500)
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="path to halueval features csv")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    y = df["label"].to_numpy().astype(int)
    print(f"[reanalysis] loaded {len(df)} rows, {y.sum()} hallu / {(1-y).sum()} right "
          f"from {args.csv}")

    nla_cols = [
        "pre_fve_centred",
        "post_fve_centred",
        "jaccard_pre",
        "jaccard_post",
        "drift",
    ]
    print("\n[reanalysis] === single-feature AUC (oracle sign) ===")
    summary: dict = {"single_feature": {}}
    for col in nla_cols + ["answer_len"]:
        a, dir_ = auc_directional(y, df[col].to_numpy())
        print(f"  {col:<22s}  AUC={a:.3f}  ({dir_})")
        summary["single_feature"][col] = {"auc": a, "direction": dir_}

    print("\n[reanalysis] === multi-feature 5-fold logistic CV ===")
    for name, cols in (
        ("NLA-only (5 feats, no length)", nla_cols),
        ("length-only", ["answer_len"]),
        ("NLA + length (6 feats)", nla_cols + ["answer_len"]),
        ("drift-only", ["drift"]),
        ("drift + post_fve", ["drift", "post_fve_centred"]),
    ):
        X = df[cols].to_numpy()
        m, s = cv_auc(X, y)
        print(f"  {name:<35s}  AUC={m:.3f} ± {s:.3f}")
        summary[name] = {"mean": m, "std": s, "cols": cols}

    # Length-matched: keep only (right, hallu) pairs whose answer_len differ
    # by ≤ 1 word, OR alternatively pool by length bucket and report
    # within-bucket AUC of NLA-only.
    print("\n[reanalysis] === length-matched pairs (|Δlen| ≤ 1 word) ===")
    by_row = df.groupby("row_id")
    keep_rows: list[int] = []
    for rid, grp in by_row:
        if len(grp) != 2:
            continue
        if abs(grp.iloc[0]["answer_len"] - grp.iloc[1]["answer_len"]) <= 1:
            keep_rows.extend(grp.index.tolist())
    sub = df.loc[keep_rows]
    print(f"  matched pairs: {len(sub)//2}/{len(df)//2}")
    if len(sub) >= 20:
        y_sub = sub["label"].to_numpy().astype(int)
        for col in nla_cols:
            a, dir_ = auc_directional(y_sub, sub[col].to_numpy())
            print(f"  {col:<22s}  AUC={a:.3f}  ({dir_})")
        X = sub[nla_cols].to_numpy()
        m, s = cv_auc(X, y_sub)
        print(f"  NLA-only logistic CV               AUC={m:.3f} ± {s:.3f}")
        summary["length_matched"] = {
            "n_pairs": len(sub) // 2,
            "nla_logistic_cv_mean": m,
            "nla_logistic_cv_std": s,
        }
    else:
        print("  too few matched pairs to evaluate")
        summary["length_matched"] = {"n_pairs": len(sub) // 2}

    # Bucketed: bin by length quartile, average within-bucket AUC of best NLA feature.
    print("\n[reanalysis] === within-length-quartile AUC ===")
    df["len_bucket"] = pd.qcut(df["answer_len"], q=4, duplicates="drop",
                                labels=False)
    bucket_aucs = []
    for b, grp in df.groupby("len_bucket"):
        if grp["label"].nunique() < 2 or len(grp) < 10:
            continue
        y_b = grp["label"].to_numpy().astype(int)
        a, _ = auc_directional(y_b, grp["drift"].to_numpy())
        print(f"  bucket {int(b)} (n={len(grp)}, mean_len={grp['answer_len'].mean():.1f})  "
              f"drift AUC={a:.3f}")
        bucket_aucs.append(a)
    if bucket_aucs:
        summary["drift_within_bucket_mean_auc"] = float(np.mean(bucket_aucs))
        print(f"  → mean within-bucket drift AUC = {np.mean(bucket_aucs):.3f}")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(f"\n[reanalysis] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
