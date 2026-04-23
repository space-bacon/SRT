#!/usr/bin/env python3
"""Hallucination probe (Tier-1 differentiator C).

For each labelled (question, answer) pair from TruthfulQA mc2, run a single
forward pass of the v5 adapter over "Q: ... A: ..." and extract per-answer-
span features:

  - max_r_hat, mean_r_hat       (BEN reflexivity estimate)
  - max_chain, mean_chain       (per-token chain residual = squared error
                                  of chain_predictor(div_i) vs div_{i+1})
  - max_div_norm, mean_div_norm (L2 of MAH divergence at last layer)
  - mean_ce                     (next-token CE on the answer span)

Label = 1 if answer is in the incorrect set, 0 if in the correct set.

Compute ROC-AUC for each feature. The headline number is whichever channel
gives the highest AUROC (training-free hallucination detector quality).

Usage:
    python scripts/hallucination_probe.py \\
        --adapter checkpoints/adapter_v5/best_adapter.pt \\
        --max-questions 200 \\
        --out artifacts/hallucination/v5_truthfulqa.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from srt.adapter import SRTAdapter  # noqa: E402
from srt.config import SRTConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("srt.halluc")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--adapter", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--dataset", default="truthfulqa/truthful_qa",
                   help="HuggingFace dataset id.")
    p.add_argument("--config", default="multiple_choice")
    p.add_argument("--split", default="validation")
    p.add_argument("--max-questions", type=int, default=200)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--prompt-template",
                   default="Q: {q}\nA: {a}",
                   help="Format string with {q} and {a} placeholders.")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default=None)
    return p.parse_args()


def get_device(req: str | None) -> torch.device:
    if req:
        return torch.device(req)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def auroc(scores: list[float], labels: list[int]) -> float:
    """ROC-AUC by Mann-Whitney U. Higher score = more likely positive class.

    labels: 1 = positive (hallucinated), 0 = negative (truthful).
    """
    if not scores or len(set(labels)) < 2:
        return float("nan")
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Rank scores (average ranks for ties)
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-indexed
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_pos_ranks = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    u = sum_pos_ranks - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


@torch.no_grad()
def features_for_pair(
    model: SRTAdapter,
    tokenizer,
    question: str,
    answer: str,
    template: str,
    device: torch.device,
    max_seq_len: int,
) -> dict | None:
    """Return per-answer-span features for one (q, a) pair, or None on error."""
    full_text = template.format(q=question.strip(), a=answer.strip())
    prefix = template.format(q=question.strip(), a="").rstrip()  # everything up to "A: "

    enc_full = tokenizer(full_text, return_tensors="pt", add_special_tokens=False,
                         truncation=True, max_length=max_seq_len)
    enc_pref = tokenizer(prefix, return_tensors="pt", add_special_tokens=False,
                         truncation=True, max_length=max_seq_len)
    full_ids = enc_full["input_ids"].to(device)
    pref_len = int(enc_pref["input_ids"].size(1))
    seq_len = int(full_ids.size(1))
    if seq_len <= pref_len:
        return None  # answer collapsed to nothing

    # Build labels: -100 for prefix tokens, real ids for answer tokens
    labels = full_ids.clone()
    labels[0, :pref_len] = -100
    attn = torch.ones_like(full_ids)

    out = model(
        input_ids=full_ids,
        attention_mask=attn,
        labels=labels,
    )

    r_hat = out.ben_output.r_hat[0].float()  # (T,)

    # Chain residual per token (mean over consecutive divergence pairs)
    if out.chain_residual_per_token is not None:
        chain_res = out.chain_residual_per_token[0].float()
    else:
        chain_res = torch.zeros(seq_len, device=device)

    # Divergence norm at last layer
    div_last = out.divergences[-1] if out.divergences else None
    if div_last is not None:
        div_norm = div_last.norm(dim=-1)[0].float()  # (T,)
    else:
        div_norm = torch.zeros(seq_len, device=device)

    # Slice answer span (positions pref_len..seq_len-1)
    span = slice(pref_len, seq_len)
    return {
        "n_tokens": seq_len - pref_len,
        "max_r_hat": float(r_hat[span].max()),
        "mean_r_hat": float(r_hat[span].mean()),
        "max_chain": float(chain_res[span].max()),
        "mean_chain": float(chain_res[span].mean()),
        "max_div_norm": float(div_norm[span].max()),
        "mean_div_norm": float(div_norm[span].mean()),
        "mean_ce": float(out.ce_loss) if out.ce_loss is not None else float("nan"),
    }


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    config = SRTConfig(backbone_id=args.backbone, backbone_dtype=args.dtype)
    logger.info("Building model on %s ...", device)
    model = SRTAdapter(config).to(device)
    state = torch.load(args.adapter, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("Loaded adapter: missing=%d unexpected=%d",
                len(missing), len(unexpected))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load TruthfulQA mc2
    from datasets import load_dataset
    logger.info("Loading %s/%s split=%s", args.dataset, args.config, args.split)
    ds = load_dataset(args.dataset, args.config, split=args.split)
    n_q = min(args.max_questions, len(ds))
    logger.info("Probing %d questions", n_q)

    rows: list[dict] = []
    for qi in range(n_q):
        item = ds[qi]
        question = item["question"]
        # mc2 has a richer label set
        if "mc2_targets" in item and item["mc2_targets"]["choices"]:
            tgt = item["mc2_targets"]
        else:
            tgt = item["mc1_targets"]
        choices = tgt["choices"]
        targets = tgt["labels"]  # 1 = correct, 0 = incorrect

        for ai, (ans, tgt_label) in enumerate(zip(choices, targets)):
            feat = features_for_pair(
                model, tokenizer, question, ans,
                args.prompt_template, device, args.max_seq_len,
            )
            if feat is None:
                continue
            label = 1 - int(tgt_label)  # 1 = hallucinated (positive class)
            row = {"qid": qi, "aid": ai, "label": label,
                   "question": question, "answer": ans, **feat}
            rows.append(row)

        if (qi + 1) % 25 == 0:
            logger.info("[%d/%d] %d (q,a) pairs collected", qi + 1, n_q, len(rows))

    # AUROC per feature
    feature_names = [
        "max_r_hat", "mean_r_hat", "max_chain", "mean_chain",
        "max_div_norm", "mean_div_norm", "mean_ce",
    ]
    labels = [r["label"] for r in rows]
    aurocs = {
        name: auroc([r[name] for r in rows], labels) for name in feature_names
    }
    pos = sum(labels)
    neg = len(labels) - pos

    summary = {
        "n_pairs": len(rows),
        "n_positive_hallucinated": pos,
        "n_negative_truthful": neg,
        "aurocs": aurocs,
        "feature_means_positive": {
            name: sum(r[name] for r in rows if r["label"] == 1) / max(pos, 1)
            for name in feature_names
        },
        "feature_means_negative": {
            name: sum(r[name] for r in rows if r["label"] == 0) / max(neg, 1)
            for name in feature_names
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows_path = args.out.with_suffix(".rows.jsonl")
    with rows_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print("\n" + "=" * 60)
    print(f"HALLUCINATION PROBE — {args.adapter.name}")
    print("=" * 60)
    print(f"  pairs                   {len(rows)} ({pos} pos / {neg} neg)")
    print()
    print("  AUROC by feature (higher = stronger hallucination detector):")
    for name in feature_names:
        print(f"    {name:18s} {aurocs[name]:.4f}")
    print("=" * 60)
    print(f"Wrote {args.out}")
    print(f"Wrote {rows_path}")


if __name__ == "__main__":
    main()
