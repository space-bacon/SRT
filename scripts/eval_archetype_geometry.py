#!/usr/bin/env python3
"""Evaluate archetype-class geometry of a trained adapter.

For each row in the archetype generations JSONL, pull the per-sample
community-head embedding (encoded vector, pre-prototype-mixing). Then
compute, per-archetype:

  - per-class centroid cosine similarities (off-diagonal mean & max)
  - recall@1 by nearest centroid (held-out via leave-one-out centroid)
  - in-class vs out-class cosine separation

Usage:
    python scripts/eval_archetype_geometry.py \
        --adapter checkpoints/adapter_v9/best_adapter.pt \
        --archetypes artifacts/archetype_probe/generations.jsonl \
        --output artifacts/archetype_geometry_v9.json \
        --max-seq-len 256 --batch-size 8 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from srt.adapter import SRTAdapter
from srt.config import SRTConfig

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("srt.archeval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--adapter", required=True)
    p.add_argument("--archetypes", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default=None)
    return p.parse_args()


def get_device(req: str | None) -> torch.device:
    if req:
        return torch.device(req)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    log.info(f"Device {device}, dtype {dtype}")
    log.info(f"Loading adapter from {args.adapter}")

    cfg = SRTConfig()
    cfg.backbone_name = args.backbone
    cfg.dtype = args.dtype
    model = SRTAdapter(cfg).to(device)
    model.load_adapter(args.adapter)
    model.eval()

    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = []
    with open(args.archetypes) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    log.info(f"Loaded {len(rows)} archetype rows")

    # Tokenize and embed
    embeddings: list[torch.Tensor] = []
    labels: list[int] = []
    names: dict[int, str] = {}

    bs = args.batch_size
    n_batches = (len(rows) + bs - 1) // bs
    for bi in range(n_batches):
        chunk = rows[bi * bs:(bi + 1) * bs]
        texts = [r["text"] for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_seq_len)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn)
        # community_output.encoded is (B, d_community)
        emb = out.community_output.encoded.detach().float().cpu()
        embeddings.append(emb)
        for r in chunk:
            labels.append(int(r["archetype_id"]))
            names[int(r["archetype_id"])] = r["archetype_name"]
        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            log.info(f"  batch {bi+1}/{n_batches}  rows={(bi+1)*bs}")

    E = torch.cat(embeddings, dim=0)  # (N, d)
    y = torch.tensor(labels)
    log.info(f"Embeddings: shape={tuple(E.shape)}")

    # Per-class centroids (L2-normalized centroid of normalized vectors)
    En = F.normalize(E, dim=-1)
    classes = sorted(set(labels))
    K = len(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}
    centroids = torch.zeros(K, En.size(1))
    counts = torch.zeros(K)
    for i, c in enumerate(labels):
        centroids[cls_idx[c]] += En[i]
        counts[cls_idx[c]] += 1
    centroids = centroids / counts.unsqueeze(-1).clamp(min=1)
    Cn = F.normalize(centroids, dim=-1)

    # Pairwise centroid cosine sim
    cos_centroids = Cn @ Cn.T  # (K, K)
    off = cos_centroids.clone()
    off.fill_diagonal_(float("nan"))
    off_flat = off[~torch.isnan(off)]
    centroid_off_diag_mean = float(off_flat.mean())
    centroid_off_diag_max = float(off_flat.max())
    centroid_off_diag_min = float(off_flat.min())

    # Recall@1 with leave-one-out centroid
    correct = 0
    for i in range(En.size(0)):
        c = labels[i]
        ci = cls_idx[c]
        # rebuild centroid for class c without sample i
        if counts[ci].item() > 1:
            new_sum = centroids[ci] * counts[ci] - En[i]
            new_centroid = new_sum / (counts[ci] - 1)
        else:
            new_centroid = centroids[ci]
        loo = centroids.clone()
        loo[ci] = new_centroid
        loo_n = F.normalize(loo, dim=-1)
        sims = (En[i:i+1] @ loo_n.T).squeeze(0)
        pred = classes[int(sims.argmax())]
        if pred == c:
            correct += 1
    recall_at_1 = correct / En.size(0)

    # In-class vs out-class cosines (sample-to-centroid)
    sims_to_own = []
    sims_to_other = []
    for i in range(En.size(0)):
        c = labels[i]
        ci = cls_idx[c]
        s = (En[i:i+1] @ Cn.T).squeeze(0)
        sims_to_own.append(float(s[ci]))
        mask = torch.ones(K, dtype=torch.bool); mask[ci] = False
        sims_to_other.append(float(s[mask].mean()))
    in_mean = float(torch.tensor(sims_to_own).mean())
    out_mean = float(torch.tensor(sims_to_other).mean())

    summary = {
        "adapter": args.adapter,
        "n_rows": len(rows),
        "n_classes": K,
        "centroid_off_diag_cos_mean": centroid_off_diag_mean,
        "centroid_off_diag_cos_max": centroid_off_diag_max,
        "centroid_off_diag_cos_min": centroid_off_diag_min,
        "recall_at_1_loo": recall_at_1,
        "sample_to_own_centroid_cos_mean": in_mean,
        "sample_to_other_centroid_cos_mean": out_mean,
        "separation_in_minus_out": in_mean - out_mean,
        "class_counts": {names[c]: int(counts[cls_idx[c]].item())
                          for c in classes},
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Wrote {args.output}")

    print()
    print("=" * 60)
    print(f"ARCHETYPE GEOMETRY  ({args.adapter})")
    print("=" * 60)
    print(f"  classes:                          {K}")
    print(f"  centroid off-diag cos mean/max:   "
          f"{centroid_off_diag_mean:.4f} / {centroid_off_diag_max:.4f}")
    print(f"  recall@1 (leave-one-out):         {recall_at_1:.4f}")
    print(f"  sample→own / other centroid cos:  "
          f"{in_mean:.4f} / {out_mean:.4f}  (sep={in_mean-out_mean:+.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
