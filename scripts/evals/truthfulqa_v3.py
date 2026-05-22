"""TruthfulQA-MC2 v3: v2 features + attention features + interactions,
evaluated with L2 logistic, sklearn GBM, LightGBM, and XGBoost.

Adds vs v2 (truthfulqa_v2.py):
  * Per-layer attention features (over a hot-band of transformer blocks):
      attn_a2q_mean_L{b}, attn_a2q_max_L{b}    answer→question attention
      attn_a2self_mean_L{b}                    answer→answer
      attn_ent_mean_L{b}                       mean answer-row entropy
      attn_head_std_L{b}                       across-head std of a→q mean
  * Layer-difference interactions for adjacent hot-band hidden-state layers:
      dlayer_drift_L{i}, dlayer_norm_L{i}
  * Cross-feature products in hot band:
      drift_x_norm_L{i}, nll_x_drift_L{i}
  * Classifier ladder: L2-logistic, sklearn-GBM, LightGBM, XGBoost.

Hot band is chosen at the call site (e.g. blocks 8..20 for Qwen-2.5-7B).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "demo"))

from nla_pipeline import BACKBONES  # noqa: E402


def parse_range(spec: str) -> list[int]:
    """Parse '8-20' or '8,9,10' into a list of ints."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


@torch.no_grad()
def features_for_choice(
    model,
    tok,
    pre_prompt: str,
    full_prompt: str,
    answer: str,
    hs_layer_idxs: list[int],
    attn_block_idxs: list[int],
    device: str,
) -> dict:
    pre_ids = tok(pre_prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tok(full_prompt, return_tensors="pt").input_ids.to(device)
    n_pre = pre_ids.shape[1]
    n_full = full_ids.shape[1]
    n_ans = n_full - n_pre
    if n_ans <= 0:
        return {}

    out = model(
        full_ids,
        output_hidden_states=True,
        output_attentions=True,
        use_cache=False,
    )
    hs = out.hidden_states            # tuple len L+1, each [1, n_full, d]
    attns = out.attentions            # tuple len L, each [1, H, n_full, n_full]
    logits = out.logits               # [1, n_full, V]

    # ---- token-level (over answer span) ----
    target_ids = full_ids[0, n_pre:n_full]
    pred_logits = logits[0, n_pre - 1 : n_full - 1, :].float()
    log_probs = F.log_softmax(pred_logits, dim=-1)
    nll_per_tok = -log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    probs = log_probs.exp()
    ent_per_tok = -(probs * log_probs).sum(dim=-1)
    top2 = log_probs.topk(2, dim=-1).values
    margin_per_tok = top2[:, 0] - top2[:, 1]

    nll_np = nll_per_tok.cpu().numpy()
    ent_np = ent_per_tok.cpu().numpy()
    mar_np = margin_per_tok.cpu().numpy()
    top3 = np.sort(nll_np)[-3:] if len(nll_np) >= 3 else nll_np

    feats: dict = {
        "answer_len_tokens": int(n_ans),
        "answer_len_words": int(len(answer.split())),
        "nll_mean": float(nll_np.mean()),
        "nll_max": float(nll_np.max()),
        "nll_std": float(nll_np.std()) if len(nll_np) > 1 else 0.0,
        "nll_top3_mean": float(top3.mean()),
        "ent_mean": float(ent_np.mean()),
        "ent_max": float(ent_np.max()),
        "margin_mean": float(mar_np.mean()),
        "margin_min": float(mar_np.min()),
    }

    # ---- per-layer hidden-state features (drift + norms) ----
    ans_slice = slice(n_pre, n_full)
    for li in hs_layer_idxs:
        layer = hs[li][0].float()
        h_pre = layer[n_pre - 1]
        h_end = layer[n_full - 1]
        cos_end = F.cosine_similarity(h_pre.unsqueeze(0),
                                       h_end.unsqueeze(0)).item()
        feats[f"drift_end_L{li}"] = float(max(0.0, 1.0 - cos_end))
        ans_h = layer[ans_slice]
        cos_ans = F.cosine_similarity(
            h_pre.unsqueeze(0).expand_as(ans_h), ans_h, dim=-1
        )
        d_ans = (1.0 - cos_ans).clamp(min=0.0)
        feats[f"drift_pool_mean_L{li}"] = float(d_ans.mean().item())
        feats[f"drift_pool_max_L{li}"] = float(d_ans.max().item())
        feats[f"drift_pool_std_L{li}"] = (
            float(d_ans.std().item()) if n_ans > 1 else 0.0
        )
        feats[f"norm_pre_L{li}"] = float(h_pre.norm().item())
        feats[f"norm_end_L{li}"] = float(h_end.norm().item())
        feats[f"norm_diff_L{li}"] = (
            feats[f"norm_end_L{li}"] - feats[f"norm_pre_L{li}"]
        )

    # ---- per-block attention features ----
    # answer rows = positions [n_pre .. n_full-1]
    # question keys = positions [0 .. n_pre-1]
    for bi in attn_block_idxs:
        a = attns[bi][0].float()                # [H, n_full, n_full]
        ans_rows = a[:, n_pre:n_full, :]        # [H, n_ans, n_full]
        a2q = ans_rows[:, :, :n_pre]            # [H, n_ans, n_pre]
        a2s = ans_rows[:, :, n_pre:n_full]      # [H, n_ans, n_ans]
        a2q_mass = a2q.sum(dim=-1)              # [H, n_ans]
        a2s_mass = a2s.sum(dim=-1)
        # entropy of each answer-row attention distribution, mean over heads+pos
        eps = 1e-12
        row_ent = -(ans_rows.clamp(min=eps) * ans_rows.clamp(min=eps).log()).sum(dim=-1)
        # head-wise mean a→q mass per answer position, then per-head average
        a2q_mean_per_head = a2q_mass.mean(dim=-1)   # [H]
        feats[f"attn_a2q_mean_B{bi}"] = float(a2q_mass.mean().item())
        feats[f"attn_a2q_max_B{bi}"] = float(a2q_mass.max().item())
        feats[f"attn_a2self_mean_B{bi}"] = float(a2s_mass.mean().item())
        feats[f"attn_ent_mean_B{bi}"] = float(row_ent.mean().item())
        feats[f"attn_head_std_B{bi}"] = float(a2q_mean_per_head.std().item())

    return feats


def add_interactions(df: pd.DataFrame, hot_layers: list[int]) -> list[str]:
    """Append interaction columns in-place; return their names."""
    new_cols: list[str] = []
    for li in hot_layers:
        if f"drift_end_L{li}" not in df.columns:
            continue
        # cross-products inside the hot band
        if f"norm_diff_L{li}" in df.columns:
            c = f"drift_x_norm_L{li}"
            df[c] = df[f"drift_pool_mean_L{li}"] * df[f"norm_diff_L{li}"]
            new_cols.append(c)
        if "nll_mean" in df.columns:
            c = f"nll_x_drift_L{li}"
            df[c] = df["nll_mean"] * df[f"drift_pool_mean_L{li}"]
            new_cols.append(c)
    # adjacent-layer differences
    for li in hot_layers[1:]:
        prev = li - 1
        if f"drift_end_L{li}" in df.columns and f"drift_end_L{prev}" in df.columns:
            c = f"dlayer_drift_L{li}"
            df[c] = df[f"drift_end_L{li}"] - df[f"drift_end_L{prev}"]
            new_cols.append(c)
        if f"norm_diff_L{li}" in df.columns and f"norm_diff_L{prev}" in df.columns:
            c = f"dlayer_norm_L{li}"
            df[c] = df[f"norm_diff_L{li}"] - df[f"norm_diff_L{prev}"]
            new_cols.append(c)
    return new_cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="qwen2.5-7b",
                    choices=list(BACKBONES.keys()))
    ap.add_argument("--n", type=int, default=817)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hs-layers", default=None,
                    help="hidden_state indices, e.g. '0-28'. "
                         "Defaults to all (0..L).")
    ap.add_argument("--attn-blocks", default=None,
                    help="transformer block indices for attention features, "
                         "e.g. '8-20'. Defaults to middle third.")
    ap.add_argument("--hot-layers", default=None,
                    help="hidden_state indices used for interaction features, "
                         "e.g. '10-20'. Defaults to attn-blocks shifted +1.")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice",
                      split="validation")
    n = min(args.n, len(ds))
    print(f"[tqa-v3] loaded {len(ds)} questions; using first {n}")

    spec = BACKBONES[args.backbone]
    print(f"[tqa-v3] loading {spec.backbone_id}")
    tok = AutoTokenizer.from_pretrained(spec.backbone_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.backbone_id,
        dtype=torch.bfloat16,
        attn_implementation="eager",   # required for output_attentions
    ).to(args.device)
    model.eval()
    L = model.config.num_hidden_layers
    hs_layer_idxs = parse_range(args.hs_layers) if args.hs_layers \
        else list(range(L + 1))
    if args.attn_blocks:
        attn_block_idxs = parse_range(args.attn_blocks)
    else:
        lo, hi = L // 3, 2 * L // 3
        attn_block_idxs = list(range(lo, hi + 1))
    if args.hot_layers:
        hot_layers = parse_range(args.hot_layers)
    else:
        hot_layers = [b + 1 for b in attn_block_idxs]

    print(f"[tqa-v3] L={L}, hs_layers={len(hs_layer_idxs)} "
          f"({hs_layer_idxs[0]}..{hs_layer_idxs[-1]}), "
          f"attn_blocks={len(attn_block_idxs)} "
          f"({attn_block_idxs[0]}..{attn_block_idxs[-1]}), "
          f"hot_layers={len(hot_layers)} "
          f"({hot_layers[0]}..{hot_layers[-1]})")

    rows: list[dict] = []
    for i in range(n):
        ex = ds[i]
        q = ex["question"].strip()
        pre = f"Q: {q}\nA:"
        for choice, lab in zip(ex["mc2_targets"]["choices"],
                               ex["mc2_targets"]["labels"]):
            choice = choice.strip()
            full = f"{pre} {choice}"
            feats = features_for_choice(
                model, tok, pre, full, choice,
                hs_layer_idxs, attn_block_idxs, args.device,
            )
            if not feats:
                continue
            feats.update({"row_id": i, "label": int(1 - lab)})
            rows.append(feats)
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"[tqa-v3] {i+1}/{n} questions, {len(rows)} choices")

    df = pd.DataFrame(rows)
    inter_cols = add_interactions(df, hot_layers)
    df.to_csv(args.out, index=False)
    print(f"[tqa-v3] wrote {len(df)} rows × {df.shape[1]} cols → {args.out}")
    print(f"[tqa-v3] added {len(inter_cols)} interaction columns")

    # ---------------- evaluation ----------------
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    y = df["label"].to_numpy().astype(int)
    groups = df["row_id"].to_numpy()
    nonfeat = {"row_id", "label"}
    feat_cols = [c for c in df.columns if c not in nonfeat]

    nll_cols = [c for c in feat_cols
                if c.startswith(("nll_", "ent_", "margin_"))]
    len_cols = [c for c in feat_cols if c.startswith("answer_len")]
    drift_cols = [c for c in feat_cols if c.startswith("drift_")
                  and "_x_" not in c and not c.startswith("dlayer_")]
    norm_cols = [c for c in feat_cols if c.startswith("norm_")]
    attn_cols = [c for c in feat_cols if c.startswith("attn_")]
    inter_set = set(inter_cols)
    inter_cols_present = [c for c in feat_cols if c in inter_set]

    print(f"\n[tqa-v3] feature families: "
          f"nll/ent/margin={len(nll_cols)}, len={len(len_cols)}, "
          f"drift={len(drift_cols)}, norm={len(norm_cols)}, "
          f"attn={len(attn_cols)}, inter={len(inter_cols_present)}, "
          f"total={len(feat_cols)}")

    gkf = GroupKFold(n_splits=5)
    Cs = [0.01, 0.1, 1.0, 10.0]

    def cv_logit(cols, name):
        X = df[cols].to_numpy()
        aucs = []
        for tr, te in gkf.split(X, y, groups=groups):
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            clf = LogisticRegressionCV(
                Cs=Cs, cv=3, max_iter=2000,
                scoring="roc_auc", n_jobs=-1,
            )
            clf.fit(Xtr, y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
        m, s = float(np.mean(aucs)), float(np.std(aucs))
        print(f"  {name:<40s}  logit AUC = {m:.4f} ± {s:.4f}")
        return m, s

    def cv_sk_gbm(cols, name):
        X = df[cols].to_numpy()
        aucs = []
        for tr, te in gkf.split(X, y, groups=groups):
            clf = GradientBoostingClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=0,
            )
            clf.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        m, s = float(np.mean(aucs)), float(np.std(aucs))
        print(f"  {name:<40s}  skGBM AUC = {m:.4f} ± {s:.4f}")
        return m, s

    def cv_lgbm(cols, name):
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            print("  (lightgbm not installed, skipping)")
            return None
        X = df[cols].to_numpy()
        aucs = []
        for tr, te in gkf.split(X, y, groups=groups):
            clf = LGBMClassifier(
                n_estimators=800, learning_rate=0.03, num_leaves=63,
                min_child_samples=20, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0,
                random_state=0, verbosity=-1,
            )
            clf.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        m, s = float(np.mean(aucs)), float(np.std(aucs))
        print(f"  {name:<40s}  LGBM  AUC = {m:.4f} ± {s:.4f}")
        return m, s

    def cv_xgb(cols, name):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            print("  (xgboost not installed, skipping)")
            return None
        X = df[cols].to_numpy()
        aucs = []
        for tr, te in gkf.split(X, y, groups=groups):
            clf = XGBClassifier(
                n_estimators=800, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, eval_metric="auc",
                tree_method="hist", random_state=0, verbosity=0,
            )
            clf.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        m, s = float(np.mean(aucs)), float(np.std(aucs))
        print(f"  {name:<40s}  XGB   AUC = {m:.4f} ± {s:.4f}")
        return m, s

    results: dict = {}
    print("\n[tqa-v3] single-family L2 logistic:")
    results["nll_family_only"] = cv_logit(nll_cols, "nll/ent/margin")
    results["drift_only"] = cv_logit(drift_cols, "drift")
    results["norm_only"] = cv_logit(norm_cols, "norm")
    if attn_cols:
        results["attn_only"] = cv_logit(attn_cols, "attention")
    if inter_cols_present:
        results["inter_only"] = cv_logit(inter_cols_present, "interactions")

    print("\n[tqa-v3] union features (no attn, no inter) — v2-equivalent:")
    v2_cols = nll_cols + len_cols + drift_cols + norm_cols
    results["v2_equiv_logit"] = cv_logit(v2_cols, "v2-equivalent logit")
    results["v2_equiv_skgbm"] = cv_sk_gbm(v2_cols, "v2-equivalent skGBM")

    print("\n[tqa-v3] ALL features (v2 + attn + inter):")
    results["all_logit"] = cv_logit(feat_cols, f"ALL ({len(feat_cols)})")
    results["all_skgbm"] = cv_sk_gbm(feat_cols, f"ALL skGBM ({len(feat_cols)})")
    r = cv_lgbm(feat_cols, f"ALL LGBM ({len(feat_cols)})")
    if r is not None:
        results["all_lgbm"] = r
    r = cv_xgb(feat_cols, f"ALL XGB ({len(feat_cols)})")
    if r is not None:
        results["all_xgb"] = r

    out_json = Path(args.out).with_suffix(".metrics.json")
    out_json.write_text(json.dumps({
        "backbone": args.backbone,
        "n_questions": int(n),
        "n_choices": int(len(df)),
        "n_features": int(len(feat_cols)),
        "n_hs_layers": len(hs_layer_idxs),
        "n_attn_blocks": len(attn_block_idxs),
        "n_hot_layers": len(hot_layers),
        "hs_layers": hs_layer_idxs,
        "attn_blocks": attn_block_idxs,
        "hot_layers": hot_layers,
        "feature_counts": {
            "nll": len(nll_cols), "len": len(len_cols),
            "drift": len(drift_cols), "norm": len(norm_cols),
            "attn": len(attn_cols), "inter": len(inter_cols_present),
        },
        "results": {k: {"mean": v[0], "std": v[1]}
                    for k, v in results.items()},
    }, indent=2))
    print(f"\n[tqa-v3] wrote metrics → {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
