#!/usr/bin/env python3
"""DDP version of train_contrastive.py.

Differences from train_contrastive.py:
- torch.distributed init via torchrun env vars (LOCAL_RANK, RANK, WORLD_SIZE).
- DistributedSampler on training set (sampler.set_epoch each epoch).
- Adapter wrapped in DistributedDataParallel.
- InfoNCE uses cross-rank gather of query + candidate embeddings
  (torch.distributed.nn.functional.all_gather is grad-aware), so
  effective in-batch negatives = batch_size * world_size * (1 + neg_per_row).
  Disable with --no-cross-rank-negs for per-rank InfoNCE.
- Only rank 0: writes logs, saves checkpoints, runs validation.
- HF push remains an external concern (use scripts/hf_push_watcher.sh).

Launch:
    torchrun --standalone --nproc_per_node=2 scripts/train_contrastive_ddp.py \\
        --backbone Qwen/Qwen2.5-7B \\
        --warm-start artifacts/checkpoints/v8a/best_adapter.pt \\
        --train-data data/contrastive/train.jsonl \\
        --val-data   data/contrastive/val.jsonl \\
        --output-dir checkpoints/adapter_v14 \\
        --batch-size 32 --negatives-per-row 1 \\
        --temperature 0.05 --lr 1e-4 --epochs 1 \\
        --max-seq-len 128 --warmup-steps 200 \\
        --val-every 1000 --log-every 100 --grad-clip 1.0 \\
        --num-workers 2 --dtype bfloat16
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
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer

from srt.adapter import SRTAdapter
from srt.config import SRTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("srt.contrastive_ddp")


# ─────────────────────── ddp helpers ───────────────────────────


def ddp_setup() -> tuple[int, int, int, bool]:
    """Returns (rank, local_rank, world_size, is_distributed)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )
        return rank, local_rank, world_size, True
    return 0, 0, 1, False


def ddp_cleanup(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def all_gather_with_grad(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """All-gather along dim 0, preserving gradient on local slice.

    Standard contrastive DDP trick: collect remote embeddings (no grad),
    keep local embeddings with grad, concatenate. Backward only flows
    through the local slice, but the loss still sees all remote
    embeddings as negatives -> correct gradient direction.
    """
    if world_size == 1:
        return tensor
    rank = dist.get_rank()
    gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather, tensor.detach())
    gather[rank] = tensor  # restore grad on our slice
    return torch.cat(gather, dim=0)


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

    out = {"q": enc(queries), "p": enc(positives), "neg_counts": neg_counts}
    if negatives_flat:
        out["n"] = enc(negatives_flat)
    return out


# ─────────────────────────── model ─────────────────────────────


@torch.no_grad()
def _eval_pairs(
    model_inner: SRTAdapter,
    tok,
    pairs: list[Pair],
    device: torch.device,
    max_seq_len: int,
    batch_size: int,
) -> dict[str, float]:
    model_inner.eval()

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
            o = model_inner(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            v = F.normalize(o.community_output.encoded.float(), dim=-1)
            out.append(v.cpu())
        return torch.cat(out, dim=0)

    q = embed([p.query for p in pairs])
    d = embed([p.positive for p in pairs])
    sims = q @ d.T
    top1 = sims.argmax(dim=-1)
    recall_at_1 = (top1 == torch.arange(len(pairs))).float().mean().item()
    ranks = (sims.argsort(dim=-1, descending=True) ==
             torch.arange(len(pairs)).unsqueeze(-1)).float().argmax(dim=-1)
    mrr = (1.0 / (ranks.float() + 1)).mean().item()
    return {"val_recall@1": recall_at_1, "val_mrr": mrr, "n_val": len(pairs)}


def _pad_to(t: torch.Tensor, length: int, value: int = 0) -> torch.Tensor:
    """Right-pad 2-D (B, L) tensor along dim 1 to ``length`` with ``value``."""
    cur = t.size(1)
    if cur >= length:
        return t[:, :length]
    pad = torch.full((t.size(0), length - cur), value, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=1)


def info_nce_step(
    model: torch.nn.Module,
    batch: dict,
    device: torch.device,
    temperature: float,
    world_size: int,
    cross_rank_negs: bool,
    pad_token_id: int = 0,
):
    """Single-forward DDP-safe InfoNCE.

    DDP requires exactly one forward per backward when find_unused_parameters
    is True (and even with it False, multiple forwards re-run reducer setup).
    We concatenate q | p | n along the batch axis (right-padded to a common
    seq len), do ONE forward, then split outputs back.
    """
    q_enc = batch["q"]
    p_enc = batch["p"]
    n_enc = batch.get("n")
    B = q_enc["input_ids"].size(0)
    Bp = p_enc["input_ids"].size(0)
    Bn = n_enc["input_ids"].size(0) if n_enc is not None else 0

    # Common max length across q/p/n for this batch
    Lq = q_enc["input_ids"].size(1)
    Lp = p_enc["input_ids"].size(1)
    Ln = n_enc["input_ids"].size(1) if n_enc is not None else 0
    Lmax = max(Lq, Lp, Ln)

    ids = [
        _pad_to(q_enc["input_ids"], Lmax, pad_token_id),
        _pad_to(p_enc["input_ids"], Lmax, pad_token_id),
    ]
    masks = [
        _pad_to(q_enc["attention_mask"], Lmax, 0),
        _pad_to(p_enc["attention_mask"], Lmax, 0),
    ]
    if n_enc is not None and Bn > 0:
        ids.append(_pad_to(n_enc["input_ids"], Lmax, pad_token_id))
        masks.append(_pad_to(n_enc["attention_mask"], Lmax, 0))

    input_ids = torch.cat(ids, dim=0).to(device)
    attention_mask = torch.cat(masks, dim=0).to(device)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    enc = F.normalize(out.community_output.encoded.float(), dim=-1)  # (B+Bp+Bn, d)

    q = enc[:B]
    p = enc[B:B + Bp]
    candidates = [p]
    if Bn > 0:
        n = enc[B + Bp:]
        candidates.append(n)
    cand_local = torch.cat(candidates, dim=0)  # (B + sum(neg_local), d)

    if cross_rank_negs and world_size > 1:
        # Gather candidates across all ranks; queries stay local but
        # logits include all-rank candidates.
        cand_full = all_gather_with_grad(cand_local, world_size)
        # Local queries stay local. Targets index into cand_full at the
        # offset corresponding to *this* rank's positives within cand_full.
        rank = dist.get_rank()
        per_rank_M = cand_local.size(0)  # same on every rank (drop_last=True + same neg_per_row)
        offset = rank * per_rank_M
        targets = torch.arange(B, device=device, dtype=torch.long) + offset
        logits = (q @ cand_full.T) / temperature
    else:
        targets = torch.arange(B, device=device, dtype=torch.long)
        logits = (q @ cand_local.T) / temperature

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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-cross-rank-negs", action="store_true",
                   help="Disable cross-rank negative gathering (per-rank InfoNCE).")
    p.add_argument("--find-unused-parameters", action="store_true", default=True,
                   help="DDP flag; on because the adapter has heads (MAH/BEN/RRM/archetype) "
                        "that do not contribute to the contrastive loss path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, is_distributed = ddp_setup()
    main_proc = is_main(rank)

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    if main_proc:
        out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    if main_proc:
        log.info(
            "DDP rank=%d/%d device=%s dtype=%s cross_rank_negs=%s",
            rank, world_size, device, args.dtype, not args.no_cross_rank_negs,
        )
        log.info("Loading adapter (warm-start=%s)", args.warm_start)

    cfg = SRTConfig()
    cfg.backbone_name = args.backbone
    cfg.dtype = args.dtype
    model_inner = SRTAdapter(cfg).to(device)
    if args.warm_start:
        model_inner.load_adapter(args.warm_start)
    model_inner.train()

    if is_distributed:
        # Broadcast initial weights so all ranks start identical.
        # static_graph=True locks in participating params after step 0,
        # which is required because SRTAdapter's forward exercises many
        # heads (MAH/BEN/RRM/archetype) whose outputs are not consumed by
        # the contrastive loss path (community_output.encoded only).
        model = DDP(
            model_inner,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            static_graph=True,
            broadcast_buffers=False,
        )
        adapter_for_save = model.module
    else:
        model = model_inner
        adapter_for_save = model_inner

    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = ContrastiveJsonl(args.train_data, max_negatives=args.negatives_per_row)
    if args.max_train_samples:
        train_ds.rows = train_ds.rows[: args.max_train_samples]
    if main_proc:
        log.info("Train rows: %d (per-rank batch=%d, world=%d, effective=%d)",
                 len(train_ds), args.batch_size, world_size,
                 args.batch_size * world_size)

    val_pairs: list[Pair] = []
    if args.val_data and main_proc:
        val_ds = ContrastiveJsonl(args.val_data, max_negatives=0)
        val_pairs = val_ds.rows[: args.max_val_samples]
        log.info("Val rows (rank0 only): %d", len(val_pairs))

    def collate_fn(batch):
        return collate(batch, tok, args.max_seq_len, args.negatives_per_row)

    if is_distributed:
        sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True, seed=args.seed,
        )
        loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True,
        )
    else:
        sampler = None
        loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True,
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
    if main_proc:
        log.info("Total steps: %d  warmup: %d", total_steps, warmup)

    cross_rank_negs = not args.no_cross_rank_negs

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            for g in optim.param_groups:
                g["lr"] = lr_at(step)

            optim.zero_grad(set_to_none=True)
            loss, stats = info_nce_step(
                model, batch, device, args.temperature,
                world_size, cross_rank_negs,
                pad_token_id=tok.pad_token_id or 0,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optim.step()

            if main_proc and step % args.log_every == 0:
                elapsed = time.time() - t0
                rec = {
                    "step": step, "epoch": epoch,
                    "lr": lr_at(step), "elapsed_s": round(elapsed, 1),
                    "world_size": world_size, **stats,
                }
                log.info(
                    "step %d  loss %.4f  acc %.3f  lr %.2e  elapsed %.0fs",
                    step, stats["infonce"], stats["infonce_acc"],
                    lr_at(step), elapsed,
                )
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")

            if val_pairs and step > 0 and step % args.val_every == 0 and main_proc:
                vstats = _eval_pairs(
                    adapter_for_save, tok, val_pairs, device,
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
                    adapter_for_save.save_adapter(str(out_dir / "best_adapter.pt"))
                    log.info("  saved best (recall@1 %.4f)", best_metric)
                model.train()

            # Sync at val boundary so non-rank0 don't run ahead during eval
            if is_distributed and step > 0 and step % args.val_every == 0:
                dist.barrier()

            step += 1

        if main_proc:
            adapter_for_save.save_adapter(str(out_dir / f"adapter_epoch{epoch+1}.pt"))

    if main_proc:
        adapter_for_save.save_adapter(str(out_dir / "final_adapter.pt"))
        cfg_dict = {
            "args": vars(args),
            "best_val_recall@1": best_metric,
            "total_steps": step,
            "world_size": world_size,
            "effective_batch": args.batch_size * world_size,
        }
        (out_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2, default=str))
        log.info("Done. best_val_recall@1=%.4f  steps=%d", best_metric, step)

    ddp_cleanup(is_distributed)


if __name__ == "__main__":
    main()
