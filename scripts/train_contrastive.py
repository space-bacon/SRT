#!/usr/bin/env python3
"""Contrastive (InfoNCE) fine-tune of the SRT-Adapter for retrieval / MTEB.

Reads a JSONL file where each row is one of:

    {"query": "...", "positive": "...", "negatives": ["...", "..."]}
    {"query": "...", "positive": "..."}                # in-batch negs only
    {"anchor": "...", "positive": "...", "negative": "..."}   # NLI triplet

Loss: InfoNCE over (query embedding, candidate embeddings) with
in-batch negatives plus any explicit hard negatives. Embeddings are
``community_output.encoded`` from the adapter (L2-normalized).

Optional auxiliary losses (off by default here so contrastive dominates):
``--archetype-supcon-weight``, ``--community-supcon-weight``. These keep
the interpretability geometry alive when nonzero.

Usage:
    python scripts/train_contrastive.py \
        --backbone Qwen/Qwen2.5-7B \
        --warm-start checkpoints/adapter_v8a/best_adapter.pt \
        --train-data data/contrastive/msmarco_hard_negs.jsonl \
        --val-data data/contrastive/val_pairs.jsonl \
        --output-dir checkpoints/adapter_v12 \
        --batch-size 32 --negatives-per-row 7 \
        --temperature 0.02 --lr 1e-4 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from srt.adapter import SRTAdapter
from srt.config import SRTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("srt.contrastive")


# ─────────────────────────── data ──────────────────────────────


@dataclass
class Pair:
    query: str
    positive: str
    negatives: list[str]


class ContrastiveJsonl(Dataset):
    def __init__(self, path: str, max_negatives: int) -> None:
        self.rows: list[Pair] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if "query" in d:
                    q = d["query"]
                    pos = d["positive"]
                    negs = list(d.get("negatives", []))[:max_negatives]
                elif "anchor" in d:
                    q = d["anchor"]
                    pos = d["positive"]
                    negs = [d["negative"]] if "negative" in d else []
                else:
                    raise ValueError(
                        f"Row missing query/anchor: {list(d.keys())}"
                    )
                self.rows.append(Pair(query=q, positive=pos, negatives=negs))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Pair:
        return self.rows[idx]


def collate(
    batch: list[Pair],
    tok,
    max_seq_len: int,
    negatives_per_row: int,
):
    queries = [p.query for p in batch]
    positives = [p.positive for p in batch]
    negatives_flat: list[str] = []
    neg_counts: list[int] = []
    for p in batch:
        n = p.negatives[:negatives_per_row]
        negatives_flat.extend(n)
        neg_counts.append(len(n))

    def enc(texts: list[str]):
        return tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )

    out = {
        "q": enc(queries),
        "p": enc(positives),
        "neg_counts": neg_counts,
    }
    if negatives_flat:
        out["n"] = enc(negatives_flat)
    return out


# ─────────────────────────── model ─────────────────────────────


@torch.no_grad()
def _eval_pairs(
    model: SRTAdapter,
    tok,
    pairs: list[Pair],
    device: torch.device,
    max_seq_len: int,
    batch_size: int,
) -> dict[str, float]:
    """Cheap val: in-batch retrieval recall@1 over `pairs`."""
    model.eval()

    def embed(texts: list[str]) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = tok(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            )
            o = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            v = F.normalize(o.community_output.encoded.float(), dim=-1)
            out.append(v.cpu())
        return torch.cat(out, dim=0)

    q = embed([p.query for p in pairs])
    d = embed([p.positive for p in pairs])
    sims = q @ d.T  # (N, N)
    top1 = sims.argmax(dim=-1)
    recall_at_1 = (top1 == torch.arange(len(pairs))).float().mean().item()
    # Mean rank of true positive
    ranks = (sims.argsort(dim=-1, descending=True) ==
             torch.arange(len(pairs)).unsqueeze(-1)).float().argmax(dim=-1)
    mrr = (1.0 / (ranks.float() + 1)).mean().item()
    return {"val_recall@1": recall_at_1, "val_mrr": mrr, "n_val": len(pairs)}


def info_nce_step(
    model: SRTAdapter,
    batch: dict,
    device: torch.device,
    temperature: float,
):
    q_enc = batch["q"]
    p_enc = batch["p"]
    n_enc = batch.get("n")
    neg_counts = batch["neg_counts"]
    B = q_enc["input_ids"].size(0)

    q_out = model(
        input_ids=q_enc["input_ids"].to(device),
        attention_mask=q_enc["attention_mask"].to(device),
    )
    p_out = model(
        input_ids=p_enc["input_ids"].to(device),
        attention_mask=p_enc["attention_mask"].to(device),
    )
    q = F.normalize(q_out.community_output.encoded.float(), dim=-1)  # (B, d)
    p = F.normalize(p_out.community_output.encoded.float(), dim=-1)  # (B, d)

    candidates = [p]  # in-batch positives + in-batch negatives
    target_offsets = list(range(B))
    if n_enc is not None and n_enc["input_ids"].size(0) > 0:
        n_out = model(
            input_ids=n_enc["input_ids"].to(device),
            attention_mask=n_enc["attention_mask"].to(device),
        )
        n = F.normalize(n_out.community_output.encoded.float(), dim=-1)
        candidates.append(n)
    cand = torch.cat(candidates, dim=0)  # (B + sum(neg), d)

    logits = (q @ cand.T) / temperature  # (B, M)
    targets = torch.tensor(target_offsets, device=device, dtype=torch.long)
    loss = F.cross_entropy(logits, targets)

    with torch.no_grad():
        acc = (logits.argmax(dim=-1) == targets).float().mean().item()
    return loss, {"infonce": loss.item(), "infonce_acc": acc}


# ─────────────────────────── train loop ────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--warm-start", default=None)
    p.add_argument("--train-data", required=True)
    p.add_argument("--val-data", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--negatives-per-row", type=int, default=7)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=2000)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    log.info("Device %s, dtype %s", device, args.dtype)
    log.info("Loading adapter (warm-start=%s)", args.warm_start)

    cfg = SRTConfig()
    cfg.backbone_name = args.backbone
    cfg.dtype = args.dtype
    model = SRTAdapter(cfg).to(device)
    if args.warm_start:
        model.load_adapter(args.warm_start)
    model.train()

    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = ContrastiveJsonl(args.train_data, max_negatives=args.negatives_per_row)
    if args.max_train_samples:
        train_ds.rows = train_ds.rows[: args.max_train_samples]
    log.info("Train rows: %d", len(train_ds))

    val_pairs: list[Pair] = []
    if args.val_data:
        val_ds = ContrastiveJsonl(args.val_data, max_negatives=0)
        val_pairs = val_ds.rows[: args.max_val_samples]
        log.info("Val rows: %d", len(val_pairs))

    def collate_fn(batch):
        return collate(batch, tok, args.max_seq_len, args.negatives_per_row)

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)

    total_steps = len(loader) * args.epochs
    warmup = max(1, args.warmup_steps)

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    best_metric = -1.0
    step = 0
    t0 = time.time()
    log.info("Total steps: %d  warmup: %d", total_steps, warmup)
    for epoch in range(args.epochs):
        for batch in loader:
            for g in optim.param_groups:
                g["lr"] = lr_at(step)

            optim.zero_grad(set_to_none=True)
            loss, stats = info_nce_step(model, batch, device, args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optim.step()

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                rec = {
                    "step": step,
                    "epoch": epoch,
                    "lr": lr_at(step),
                    "elapsed_s": round(elapsed, 1),
                    **stats,
                }
                log.info(
                    "step %d  loss %.4f  acc %.3f  lr %.2e  elapsed %.0fs",
                    step, stats["infonce"], stats["infonce_acc"],
                    lr_at(step), elapsed,
                )
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")

            if val_pairs and step > 0 and step % args.val_every == 0:
                vstats = _eval_pairs(
                    model, tok, val_pairs, device,
                    args.max_seq_len, args.batch_size,
                )
                vstats["step"] = step
                vstats["kind"] = "val"
                log.info(
                    "VAL step %d  recall@1 %.4f  mrr %.4f  n=%d",
                    step, vstats["val_recall@1"], vstats["val_mrr"],
                    vstats["n_val"],
                )
                with log_path.open("a") as f:
                    f.write(json.dumps(vstats) + "\n")

                if vstats["val_recall@1"] > best_metric:
                    best_metric = vstats["val_recall@1"]
                    model.save_adapter(str(out_dir / "best_adapter.pt"))
                    log.info("  saved best (recall@1 %.4f)", best_metric)
                model.train()

            step += 1

        # End-of-epoch save
        model.save_adapter(str(out_dir / f"adapter_epoch{epoch+1}.pt"))

    model.save_adapter(str(out_dir / "final_adapter.pt"))
    cfg_dict = {
        "args": vars(args),
        "best_val_recall@1": best_metric,
        "total_steps": step,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2, default=str))
    log.info("Done. best_val_recall@1=%.4f  steps=%d", best_metric, step)


if __name__ == "__main__":
    main()
