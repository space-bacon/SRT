"""TruthfulQA-MC2 with stronger features: per-layer drift + answer NLL.

Improvements over `truthfulqa_nla.py`:
  * **answer_nll**  - mean per-token NLL of the answer span given the prompt.
    This is the standard MC2 baseline; any honest detector must clear it.
  * **drift_L{i}**  - 1 - cos(h_pre, h_post) at multiple layers, not just
    the last one. Mid layers often carry the cleanest semantics.
  * Single forward pass per (question, choice) over the full prompt with
    `output_hidden_states=True`, so we get the answer logits and all layer
    activations in one shot.

Label: 1 = hallucination (mc2 label == 0), 0 = truthful.
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
    # hidden_states: tuple of (n_layers+1) tensors, each [1, n_full, d]
    hs = out.hidden_states
    logits = out.logits  # [1, n_full, V]

    # Answer NLL: predict tokens at positions [n_pre, ..., n_full-1]
    # from logits at positions [n_pre-1, ..., n_full-2].
    target_ids = full_ids[0, n_pre:n_full]
    pred_logits = logits[0, n_pre - 1 : n_full - 1, :]
    log_probs = F.log_softmax(pred_logits.float(), dim=-1)
    nll = -log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1).mean().item()

    # Per-layer drift between last prompt token (n_pre-1) and last answer
    # token (n_full-1).
    drifts: dict[str, float] = {}
    for li in layer_idxs:
        layer = hs[li][0]  # [n_full, d]
        h_pre = layer[n_pre - 1]
        h_post = layer[n_full - 1]
        cos = F.cosine_similarity(
            h_pre.float().unsqueeze(0), h_post.float().unsqueeze(0)
        ).item()
        drifts[f"drift_L{li}"] = float(max(0.0, 1.0 - cos))

    feats = {
        "answer_nll": float(nll),
        "answer_len_tokens": int(n_ans),
        "answer_len_words": int(len(answer.split())),
    }
    feats.update(drifts)
    return feats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="qwen2.5-7b",
                    choices=list(BACKBONES.keys()))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layers", default="6,14,22,28",
                    help="comma-separated layer indices into hidden_states "
                         "(0=embeddings, len-1=final)")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    layer_idxs = [int(x) for x in args.layers.split(",") if x.strip()]

    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice",
                      split="validation")
    n = min(args.n, len(ds))
    print(f"[tqa-strong] loaded {len(ds)} questions; using first {n}")

    spec = BACKBONES[args.backbone]
    print(f"[tqa-strong] loading {spec.backbone_id}")
    tok = AutoTokenizer.from_pretrained(spec.backbone_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.backbone_id, dtype=torch.bfloat16
    ).to(args.device)
    model.eval()
    n_layers_total = model.config.num_hidden_layers
    print(f"[tqa-strong] model has {n_layers_total} layers "
          f"(hidden_states len = {n_layers_total + 1}); using {layer_idxs}")

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

        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"[tqa-strong] {i+1}/{n} questions, {len(rows)} choices")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[tqa-strong] wrote {len(df)} rows → {args.out}")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler

        y = df["label"].to_numpy().astype(int)
        groups = df["row_id"].to_numpy()
        drift_cols = [c for c in df.columns if c.startswith("drift_L")]
        all_single = ["answer_nll", "answer_len_tokens"] + drift_cols

        print("\n[tqa-strong] single-feature oracle-sign AUC:")
        for col in all_single:
            x = df[col].to_numpy()
            a = roc_auc_score(y, x)
            d = "↑→hallu" if a >= 0.5 else "↓→hallu"
            print(f"  {col:<22s}  AUC={max(a,1-a):.3f}  ({d})")

        gkf = GroupKFold(n_splits=5)

        def cv(cols, name):
            X = StandardScaler().fit_transform(df[cols].to_numpy())
            aucs = []
            for tr, te in gkf.split(X, y, groups=groups):
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X[tr], y[tr])
                aucs.append(
                    roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
                )
            print(f"  {name:<32s}  group-CV AUC = "
                  f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
            return float(np.mean(aucs)), float(np.std(aucs))

        print("\n[tqa-strong] group-CV logistic AUC:")
        m = {}
        m["nll_only"] = cv(["answer_nll"], "answer_nll only (BASELINE)")
        m["drift_only"] = cv(drift_cols, f"drift ({len(drift_cols)} layers)")
        m["nll_plus_drift"] = cv(["answer_nll"] + drift_cols,
                                  "answer_nll + drift")
        m["nll_plus_drift_plus_len"] = cv(
            ["answer_nll", "answer_len_tokens"] + drift_cols,
            "nll + drift + length"
        )

        out_json = Path(args.out).with_suffix(".metrics.json")
        out_json.write_text(json.dumps({
            "backbone": args.backbone,
            "n_questions": int(n),
            "n_choices": int(len(df)),
            "layers": layer_idxs,
            "group_cv": {k: {"mean": v[0], "std": v[1]} for k, v in m.items()},
        }, indent=2))
        print(f"\n[tqa-strong] wrote metrics → {out_json}")
    except ImportError:
        print("[tqa-strong] sklearn not installed; skipping AUC eval")
    return 0


if __name__ == "__main__":
    sys.exit(main())
