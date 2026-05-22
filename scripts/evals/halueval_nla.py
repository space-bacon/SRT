"""Hallucination-detection harness for HaluEval using NLA features.

Pipeline
========

For every (knowledge, question, answer) row in HaluEval, we construct the
prompt the model would have seen had it generated `answer` itself, then
extract three NLA-derived features:

    pre_fve   : centred fve of the top-1 AV verbalisation of v_pre
                (hidden state at the end of the question, before answering).
                Low fve ⇒ the model's internal state was already diffuse /
                low-confidence before it answered.
    jaccard   : lexical Jaccard between that pre-answer thought and the
                answer text. Low Jaccard ⇒ the model "thinks" something
                different from what it claims to say (or was asked to claim).
    drift     : 1 − cos(v_pre, v_post). High drift ⇒ the act of asserting
                the answer dragged the residual stream a long way, often
                a sign of forced / fabricated content.

We then fit a tiny logistic regression on a train split and report ROC-AUC
on a held-out split. Crucially the AV is frozen — no fine-tuning happens.

Usage
-----

    python scripts/evals/halueval_nla.py \
        --backbone qwen2.5-7b \
        --split qa \
        --n 600 \
        --out artifacts/halueval/qa_qwen.csv

`--split` is one of {qa, dialogue, summarization}. `--n` is the number of
positive+negative pairs to sample (each pair contributes 2 rows: a right
answer and a hallucinated answer).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Allow running as a script from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "demo"))

from nla_pipeline import BACKBONES, NLAPipeline  # noqa: E402

# --- prompt construction ----------------------------------------------------

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


def _build_prompts(split: str, row: dict) -> tuple[str, str, str]:
    """Return (prompt_pre, prompt_full_right, prompt_full_hallu).

    prompt_pre  : tokens up to (and including) the question, BEFORE the answer.
    prompt_full : prompt_pre + ' ' + answer.
    """
    if split == "qa":
        # HaluEval QA: {knowledge, question, right_answer, hallucinated_answer}
        ctx = row.get("knowledge", "").strip()
        q = row["question"].strip()
        pre = f"{ctx}\n\nQuestion: {q}\nAnswer:"
        return (
            pre,
            f"{pre} {row['right_answer'].strip()}",
            f"{pre} {row['hallucinated_answer'].strip()}",
        )
    if split == "dialogue":
        ctx = row.get("knowledge", "").strip()
        hist = row["dialogue_history"].strip()
        pre = f"{ctx}\n\n{hist}\nResponse:"
        return (
            pre,
            f"{pre} {row['right_response'].strip()}",
            f"{pre} {row['hallucinated_response'].strip()}",
        )
    if split == "summarization":
        doc = row["document"].strip()
        pre = f"{doc}\n\nSummary:"
        return (
            pre,
            f"{pre} {row['right_summary'].strip()}",
            f"{pre} {row['hallucinated_summary'].strip()}",
        )
    raise ValueError(f"unknown split {split}")


# --- feature extraction -----------------------------------------------------


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
    ranked = pipe.verbalize(
        v_pre, K=av_K, temperature=av_temp, max_new_tokens=av_max_new
    )
    thought, raw, cen = ranked[0] if ranked else ("", 0.0, 0.0)
    t_w = _content_words(thought)
    a_w = _content_words(answer)
    jacc = len(t_w & a_w) / max(len(t_w | a_w), 1) if (t_w or a_w) else 0.0
    drift = 1.0 - F.cosine_similarity(
        v_pre.unsqueeze(0), v_post.unsqueeze(0)
    ).item()
    drift = max(0.0, drift)
    return {
        "pre_fve_raw": float(raw),
        "pre_fve_centred": float(cen),
        "jaccard": float(jacc),
        "drift": float(drift),
        "thought": thought.strip().replace("\n", " ")[:240],
    }


# --- main -------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="qwen2.5-7b",
                    choices=list(BACKBONES.keys()))
    ap.add_argument("--split", default="qa",
                    choices=["qa", "dialogue", "summarization"])
    ap.add_argument("--n", type=int, default=300,
                    help="number of source rows (each becomes 2 examples)")
    ap.add_argument("--av-K", type=int, default=4)
    ap.add_argument("--av-max-new", type=int, default=32)
    ap.add_argument("--out", required=True,
                    help="output CSV path for per-example features+labels")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset-id", default="pminervini/HaluEval",
                    help="HF dataset id to load")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    ds = load_dataset(args.dataset_id, args.split, split="data")
    n = min(args.n, len(ds))
    print(f"[halueval-nla] loaded {len(ds)} rows from {args.dataset_id}/{args.split}; "
          f"using first {n}")

    pipe = NLAPipeline(BACKBONES[args.backbone], device=args.device)
    rows: list[dict] = []
    for i in range(n):
        ex = ds[i]
        try:
            pre, full_right, full_hallu = _build_prompts(args.split, ex)
        except KeyError as e:
            print(f"[halueval-nla] row {i} missing field {e}; skipping")
            continue

        right_answer = full_right[len(pre):].strip()
        hallu_answer = full_hallu[len(pre):].strip()

        for label, full, answer in (
            (0, full_right, right_answer),
            (1, full_hallu, hallu_answer),
        ):
            feats = features_for(
                pipe, pre, full, answer,
                av_K=args.av_K, av_max_new=args.av_max_new,
            )
            feats.update({"row_id": i, "label": label})
            rows.append(feats)

        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"[halueval-nla] processed {i+1}/{n} rows")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[halueval-nla] wrote {len(df)} rows → {args.out}")

    # Quick in-script eval: logistic regression on (jaccard, drift, pre_fve).
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold

        X = df[["pre_fve_centred", "jaccard", "drift"]].to_numpy()
        y = df["label"].to_numpy()

        # Single-feature baselines.
        for col in ("pre_fve_centred", "jaccard", "drift"):
            s = -df[col].to_numpy() if col != "drift" else df[col].to_numpy()
            try:
                auc = roc_auc_score(y, s)
                print(f"[halueval-nla] AUC single-feature {col:<18s} = {auc:.3f}")
            except ValueError:
                pass

        # 5-fold CV with logistic regression.
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        aucs = []
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(max_iter=200)
            clf.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        print(f"[halueval-nla] AUC logistic(3 feats) 5-fold = "
              f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

        out_json = Path(args.out).with_suffix(".metrics.json")
        out_json.write_text(json.dumps({
            "backbone": args.backbone,
            "split": args.split,
            "n_rows": int(len(df)),
            "auc_cv_mean": float(np.mean(aucs)),
            "auc_cv_std": float(np.std(aucs)),
        }, indent=2))
        print(f"[halueval-nla] wrote metrics → {out_json}")
    except ImportError:
        print("[halueval-nla] sklearn not installed; skipping AUC eval")

    return 0


if __name__ == "__main__":
    sys.exit(main())
