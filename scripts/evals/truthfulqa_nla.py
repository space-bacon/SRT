"""TruthfulQA-MC2 hallucination-detection harness.

Same NLA feature pipeline as `halueval_nla.py`, but on TruthfulQA's
multiple-choice format where each question carries multiple paired
(choice, is_truthful) candidates. This dataset has *no length confound*
the way HaluEval-QA does: false choices are written in the same style
and length as truthful ones, by design.

Label convention: label = 1 means "this choice is a hallucination"
(i.e. mc2_targets.labels == 0), label = 0 means truthful.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "demo"))

from nla_pipeline import BACKBONES, NLAPipeline  # noqa: E402

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "by", "for", "with", "as", "that",
    "this", "it", "its", "and", "or", "but", "if", "so", "than", "then",
    "i", "you", "he", "she", "we", "they", "them", "his", "her", "their",
    "have", "has", "had", "do", "does", "did", "will", "would", "should",
    "can", "could", "may", "might", "not", "no", "yes",
}


def _content_words(text: str) -> set[str]:
    return {
        t.lower().strip(".,!?;:\"'()[]")
        for t in text.split()
        if len(t) >= 2 and t.lower() not in _STOPWORDS
    }


@torch.no_grad()
def features_for(
    pipe: NLAPipeline,
    pre_prompt: str,
    full_prompt: str,
    answer: str,
    av_K: int = 4,
    av_max_new: int = 32,
    av_temp: float = 1.0,
) -> dict:
    v_pre = pipe.extract_hidden(pre_prompt, token_index=-1)
    v_post = pipe.extract_hidden(full_prompt, token_index=-1)
    ranked_pre = pipe.verbalize(
        v_pre, K=av_K, temperature=av_temp, max_new_tokens=av_max_new
    )
    thought_pre, raw_pre, cen_pre = (
        ranked_pre[0] if ranked_pre else ("", 0.0, 0.0)
    )
    ranked_post = pipe.verbalize(
        v_post, K=av_K, temperature=av_temp, max_new_tokens=av_max_new
    )
    thought_post, raw_post, cen_post = (
        ranked_post[0] if ranked_post else ("", 0.0, 0.0)
    )
    a_w = _content_words(answer)
    tp_w = _content_words(thought_pre)
    tq_w = _content_words(thought_post)
    jacc_pre = len(tp_w & a_w) / max(len(tp_w | a_w), 1) if (tp_w or a_w) else 0.0
    jacc_post = len(tq_w & a_w) / max(len(tq_w | a_w), 1) if (tq_w or a_w) else 0.0
    drift = 1.0 - F.cosine_similarity(
        v_pre.unsqueeze(0), v_post.unsqueeze(0)
    ).item()
    drift = max(0.0, drift)
    return {
        "pre_fve_centred": float(cen_pre),
        "post_fve_centred": float(cen_post),
        "jaccard_pre": float(jacc_pre),
        "jaccard_post": float(jacc_post),
        "drift": float(drift),
        "answer_len": float(len(answer.split())),
        "thought_pre": thought_pre.strip().replace("\n", " ")[:240],
        "thought_post": thought_post.strip().replace("\n", " ")[:240],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="qwen2.5-7b",
                    choices=list(BACKBONES.keys()))
    ap.add_argument("--n", type=int, default=200,
                    help="number of QUESTIONS (each contributes ~4 choices)")
    ap.add_argument("--av-K", type=int, default=4)
    ap.add_argument("--av-max-new", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice",
                      split="validation")
    n = min(args.n, len(ds))
    print(f"[truthfulqa-nla] loaded {len(ds)} questions; using first {n}")

    pipe = NLAPipeline(BACKBONES[args.backbone], device=args.device)
    rows: list[dict] = []
    for i in range(n):
        ex = ds[i]
        q = ex["question"].strip()
        pre = f"Q: {q}\nA:"
        choices = ex["mc2_targets"]["choices"]
        labels = ex["mc2_targets"]["labels"]
        for choice, lab in zip(choices, labels):
            choice = choice.strip()
            full = f"{pre} {choice}"
            feats = features_for(
                pipe, pre, full, choice,
                av_K=args.av_K, av_max_new=args.av_max_new,
            )
            # label = 1 means hallucination (mc2 lab == 0 means false)
            feats.update({"row_id": i, "label": int(1 - lab)})
            rows.append(feats)

        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"[truthfulqa-nla] processed {i+1}/{n} questions "
                  f"({len(rows)} choices)")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[truthfulqa-nla] wrote {len(df)} rows → {args.out}")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold

        nla_cols = [
            "pre_fve_centred", "post_fve_centred",
            "jaccard_pre", "jaccard_post", "drift",
        ]
        y = df["label"].to_numpy().astype(int)
        groups = df["row_id"].to_numpy()

        for col in nla_cols + ["answer_len"]:
            x = df[col].to_numpy()
            a = roc_auc_score(y, x)
            d = "↑→hallu" if a >= 0.5 else "↓→hallu"
            a = max(a, 1.0 - a)
            print(f"[truthfulqa-nla] {col:<22s}  AUC={a:.3f}  ({d})")

        # Group-aware CV: keep all choices of one question in the same fold
        # so we measure question-level generalisation, not within-question
        # memorisation.
        gkf = GroupKFold(n_splits=5)
        for name, cols in (
            ("NLA-only (5 feats)", nla_cols),
            ("NLA + length", nla_cols + ["answer_len"]),
            ("drift-only", ["drift"]),
        ):
            X = df[cols].to_numpy()
            aucs = []
            for tr, te in gkf.split(X, y, groups=groups):
                clf = LogisticRegression(max_iter=500)
                clf.fit(X[tr], y[tr])
                aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
            print(f"[truthfulqa-nla] {name:<25s}  group-CV AUC = "
                  f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

        out_json = Path(args.out).with_suffix(".metrics.json")
        out_json.write_text(json.dumps({
            "backbone": args.backbone,
            "n_questions": int(n),
            "n_choices": int(len(df)),
        }, indent=2))
        print(f"[truthfulqa-nla] wrote metrics → {out_json}")
    except ImportError:
        print("[truthfulqa-nla] sklearn not installed; skipping AUC eval")
    return 0


if __name__ == "__main__":
    sys.exit(main())
