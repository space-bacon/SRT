#!/usr/bin/env python3
"""Regime-head calibration curve (Tier-2 F).

Runs the adapter over val data, collects per-token P(supercritical) from
BEN's regime_logits and the ground-truth label (r_true > 0). Bins by
predicted probability and plots empirical positive rate against bin
midpoint (reliability diagram) plus ECE.

Usage:
    python scripts/regime_calibration.py \
        --adapter checkpoints/adapter_v5/best_adapter.pt \
        --val-data data/all_val.jsonl \
        --max-samples 2000 \
        --out artifacts/regime_calibration/v5_step17000.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from srt.adapter import SRTAdapter
from srt.config import SRTConfig
from srt.data.dataset import SRTAdapterDataset, make_collate_fn

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("srt.calib")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--adapter", required=True)
    p.add_argument("--val-data", required=True)
    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-bins", type=int, default=15)
    p.add_argument("--out", required=True)
    return p.parse_args()


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> tuple[float, list[dict]]:
    """Expected Calibration Error + per-bin breakdown."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict] = []
    total = len(probs)
    weighted = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        if n == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0,
                         "mean_pred": None, "empirical_pos_rate": None,
                         "gap": None})
            continue
        mean_p = float(probs[mask].mean())
        emp = float(labels[mask].mean())
        gap = abs(mean_p - emp)
        weighted += (n / total) * gap
        bins.append({"lo": float(lo), "hi": float(hi), "n": n,
                     "mean_pred": mean_p, "empirical_pos_rate": emp,
                     "gap": gap})
    return float(weighted), bins


@torch.no_grad()
def collect(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    probs_all, labels_all = [], []
    for bi, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()
                 if isinstance(v, torch.Tensor)}
        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"])
        # P(supercritical) = softmax(regime_logits)[..., 1]
        p = torch.softmax(out.ben_output.regime_logits.float(), dim=-1)[..., 1]
        r_true = batch["r_true"].float()
        mask = batch["r_mask"].bool() & batch["attention_mask"].bool()
        probs_all.append(p[mask].cpu().numpy())
        labels_all.append((r_true[mask] > 0).cpu().numpy().astype(np.int8))
        if (bi + 1) % 10 == 0:
            logger.info("batch %d   tokens collected=%d",
                        bi + 1, sum(len(x) for x in probs_all))
    return np.concatenate(probs_all), np.concatenate(labels_all)


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SRTConfig()
    logger.info("Building model on %s ...", device)
    model = SRTAdapter(cfg).to(device, dtype=torch.bfloat16)
    model.load_adapter(args.adapter)
    model.eval()
    logger.info("Loaded adapter: %s", args.adapter)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = SRTAdapterDataset(args.val_data, tokenizer,
                             max_seq_len=args.max_seq_len,
                             max_samples=args.max_samples)
    loader = DataLoader(base, batch_size=args.batch_size, shuffle=False,
                        collate_fn=make_collate_fn(tokenizer.pad_token_id),
                        num_workers=2)

    probs, labels = collect(model, loader, device)
    logger.info("Total tokens: %d  (positives %d, %.3f)", len(labels),
                int(labels.sum()), labels.mean())

    ece_val, bins = ece(probs, labels, args.n_bins)
    base_rate = float(labels.mean())
    brier = float(((probs - labels) ** 2).mean())

    # AUROC via Mann-Whitney U
    pos = probs[labels == 1]
    neg = probs[labels == 0]
    if len(pos) and len(neg):
        ranks = probs.argsort().argsort() + 1
        rank_sum_pos = ranks[labels == 1].sum()
        u = rank_sum_pos - len(pos) * (len(pos) + 1) / 2
        auroc = float(u / (len(pos) * len(neg)))
    else:
        auroc = float("nan")

    result = {
        "adapter": args.adapter,
        "n_tokens": int(len(labels)),
        "base_rate": base_rate,
        "ece": ece_val,
        "brier": brier,
        "auroc": auroc,
        "n_bins": args.n_bins,
        "bins": bins,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print(f"REGIME CALIBRATION — {args.adapter}")
    print("=" * 60)
    print(f"  tokens       {len(labels)}")
    print(f"  base rate    {base_rate:.4f}")
    print(f"  ECE          {ece_val:.4f}")
    print(f"  Brier        {brier:.4f}")
    print(f"  AUROC        {auroc:.4f}")
    print()
    print(f"  {'bin':>10}   {'n':>7}  {'mean_pred':>10}  {'empirical':>10}  {'gap':>7}")
    for b in bins:
        if b["n"] == 0:
            continue
        print(f"  [{b['lo']:.2f},{b['hi']:.2f}]   {b['n']:7d}  "
              f"{b['mean_pred']:10.4f}  {b['empirical_pos_rate']:10.4f}  "
              f"{b['gap']:7.4f}")
    print("=" * 60)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
