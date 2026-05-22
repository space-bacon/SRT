"""TruthfulQA-MC2 v2: rich token-level + per-layer features.

Same single forward pass per (question, choice) as truthfulqa_strong.py,
but extracts substantially more signal:

  Token-level (over the answer span):
    nll_mean, nll_max, nll_std, nll_top3_mean
    ent_mean, ent_max
    margin_mean (top1 - top2 log-prob), margin_min

  Per layer (over a sparse set of layers):
    drift_end_L{i}        cos(h_pre, h_end_of_answer)
    drift_pool_mean_L{i}  mean over answer positions of 1-cos(h_pre, h_t)
    drift_pool_max_L{i}
    drift_pool_std_L{i}
    norm_end_L{i}         ||h_end_of_answer||
    norm_pre_L{i}         ||h_end_of_prompt||
    norm_diff_L{i}        ||h_end|| - ||h_pre||

Classifier: L2 logistic with C selected on inner CV, plus optional GBM.
GroupKFold on question id (no within-question leakage).
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


@torch.no_grad()
def features_for_choice(
    model,
    tok,
    pre_prompt: str,
    full_prompt: str,
    answer: str,
    layer_idxs: list[int],
    device: str,
) -> dict:
    pre_ids = tok(pre_prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tok(full_prompt, return_tensors="pt").input_ids.to(device)
    n_pre = pre_ids.shape[1]
    n_full = full_ids.shape[1]
    n_ans = n_full - n_pre
    if n_ans <= 0:
        return {}

    out = model(full_ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple of (L+1) tensors [1, n_full, d]
    logits = out.logits      # [1, n_full, V]

    # ---- token-level distributions over the answer span ----
    target_ids = full_ids[0, n_pre:n_full]                # [n_ans]
    pred_logits = logits[0, n_pre - 1 : n_full - 1, :].float()  # [n_ans, V]
    log_probs = F.log_softmax(pred_logits, dim=-1)
    nll_per_tok = -log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    # entropy: -sum p log p
    probs = log_probs.exp()
    ent_per_tok = -(probs * log_probs).sum(dim=-1)
    # margin: log p(top1) - log p(top2)
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

    # ---- per-layer features ----
    ans_slice = slice(n_pre, n_full)
    for li in layer_idxs:
        layer = hs[li][0].float()  # [n_full, d]
        h_pre = layer[n_pre - 1]
        h_end = layer[n_full - 1]
        # endpoint drift
        cos_end = F.cosine_similarity(h_pre.unsqueeze(0),
                                       h_end.unsqueeze(0)).item()
        feats[f"drift_end_L{li}"] = float(max(0.0, 1.0 - cos_end))
        # pooled drift over answer span vs end-of-prompt
        ans_h = layer[ans_slice]  # [n_ans, d]
        cos_ans = F.cosine_similarity(
            h_pre.unsqueeze(0).expand_as(ans_h), ans_h, dim=-1
        )
        d_ans = (1.0 - cos_ans).clamp(min=0.0)
        feats[f"drift_pool_mean_L{li}"] = float(d_ans.mean().item())
        feats[f"drift_pool_max_L{li}"] = float(d_ans.max().item())
        feats[f"drift_pool_std_L{li}"] = (
            float(d_ans.std().item()) if n_ans > 1 else 0.0
        )
        # norms
        feats[f"norm_pre_L{li}"] = float(h_pre.norm().item())
        feats[f"norm_end_L{li}"] = float(h_end.norm().item())
        feats[f"norm_diff_L{li}"] = (
            feats[f"norm_end_L{li}"] - feats[f"norm_pre_L{li}"]
        )
    return feats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="qwen2.5-7b",
                    choices=list(BACKBONES.keys()))
    ap.add_argument("--n", type=int, default=817)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layers", default=None,
                    help="comma-separated layer indices into hidden_states; "
                         "defaults to all layers")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice",
                      split="validation")
    n = min(args.n, len(ds))
    print(f"[tqa-v2] loaded {len(ds)} questions; using first {n}")

    spec = BACKBONES[args.backbone]
    print(f"[tqa-v2] loading {spec.backbone_id}")
    tok = AutoTokenizer.from_pretrained(spec.backbone_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.backbone_id, dtype=torch.bfloat16
    ).to(args.device)
    model.eval()
    n_layers_total = model.config.num_hidden_layers
    if args.layers:
        layer_idxs = [int(x) for x in args.layers.split(",") if x.strip()]
    else:
        layer_idxs = list(range(n_layers_total + 1))
    print(f"[tqa-v2] model has {n_layers_total} layers; using "
          f"{len(layer_idxs)} layer slices")

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
                model, tok, pre, full, choice, layer_idxs, args.device
            )
            if not feats:
                continue
            feats.update({"row_id": i, "label": int(1 - lab)})
            rows.append(feats)

        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"[tqa-v2] {i+1}/{n} questions, {len(rows)} choices")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[tqa-v2] wrote {len(df)} rows × {df.shape[1]} cols → {args.out}")

    try:
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler

        y = df["label"].to_numpy().astype(int)
        groups = df["row_id"].to_numpy()
        nonfeat = {"row_id", "label"}
        feat_cols = [c for c in df.columns if c not in nonfeat]

        # group families
        nll_cols = [c for c in feat_cols if c.startswith(
            ("nll_", "ent_", "margin_"))]
        len_cols = [c for c in feat_cols if c.startswith("answer_len")]
        drift_cols = [c for c in feat_cols if c.startswith("drift_")]
        norm_cols = [c for c in feat_cols if c.startswith("norm_")]

        print(f"\n[tqa-v2] feature families: "
              f"nll/ent/margin={len(nll_cols)}, len={len(len_cols)}, "
              f"drift={len(drift_cols)}, norm={len(norm_cols)}, "
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
                    scoring="roc_auc", n_jobs=-1
                )
                clf.fit(Xtr, y[tr])
                aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
            m, s = float(np.mean(aucs)), float(np.std(aucs))
            print(f"  {name:<38s}  logit-CV AUC = {m:.3f} ± {s:.3f}")
            return m, s

        def cv_gbm(cols, name):
            X = df[cols].to_numpy()
            aucs = []
            for tr, te in gkf.split(X, y, groups=groups):
                clf = GradientBoostingClassifier(
                    n_estimators=300, max_depth=3, learning_rate=0.05,
                    subsample=0.8, random_state=0
                )
                clf.fit(X[tr], y[tr])
                aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
            m, s = float(np.mean(aucs)), float(np.std(aucs))
            print(f"  {name:<38s}  gbm-CV   AUC = {m:.3f} ± {s:.3f}")
            return m, s

        print("\n[tqa-v2] group-CV AUC (L2-logistic with inner C search):")
        results = {}
        results["nll_family_only"] = cv_logit(nll_cols, "token NLL/ent/margin")
        results["drift_only"] = cv_logit(drift_cols, "drift (all variants)")
        results["norm_only"] = cv_logit(norm_cols, "norms")
        results["nll_plus_drift"] = cv_logit(
            nll_cols + drift_cols, "nll + drift")
        results["all_features"] = cv_logit(
            feat_cols, f"ALL ({len(feat_cols)} feats)")

        print("\n[tqa-v2] group-CV AUC (gradient boosting):")
        results["gbm_all"] = cv_gbm(feat_cols, f"GBM on ALL ({len(feat_cols)})")

        out_json = Path(args.out).with_suffix(".metrics.json")
        out_json.write_text(json.dumps({
            "backbone": args.backbone,
            "n_questions": int(n),
            "n_choices": int(len(df)),
            "n_features": int(len(feat_cols)),
            "n_layer_slices": len(layer_idxs),
            "results": {k: {"mean": v[0], "std": v[1]}
                        for k, v in results.items()},
        }, indent=2))
        print(f"\n[tqa-v2] wrote metrics → {out_json}")
    except ImportError as e:
        print(f"[tqa-v2] sklearn missing: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
