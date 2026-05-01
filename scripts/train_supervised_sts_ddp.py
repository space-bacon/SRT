#!/usr/bin/env python3
"""DDP supervised STS regression for SRT adapter.

Loss: MSE between cosine(emb_s1, emb_s2) and human-labeled similarity score in [0, 1].
Best-checkpoint criterion: Spearman rank correlation on a held-out dev set
(matches MTEB STS metric directly).

Architecture: single SRTAdapter forward over concatenated (s1 || s2), then
split outputs and compute pairwise cosine. No negatives, no all_gather --
loss is per-row, so DDP just averages gradients normally.

Launch:
    torchrun --standalone --nproc_per_node=2 scripts/train_supervised_sts_ddp.py \\
        --backbone Qwen/Qwen2.5-7B \\
        --warm-start /root/srt-adapter/checkpoints/adapter_v15a/best_adapter.pt \\
        --train-data data/supervised_sts_v17/train.jsonl \\
        --dev-data   data/supervised_sts_v17/dev.jsonl \\
        --output-dir /root/srt-adapter/checkpoints/adapter_v17 \\
        --batch-size 32 --max-seq-len 128 \\
        --lr 1e-5 --warmup-steps 50 --epochs 5 \\
        --val-every 100 --log-every 20 \\
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
log = logging.getLogger("srt.supervised_sts_ddp")


# ─────────────────────── ddp helpers ───────────────────────────


def ddp_setup() -> tuple[int, int, int, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, local_rank, world_size, True
    return 0, 0, 1, False


def ddp_cleanup(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


# ─────────────────────────── data ──────────────────────────────


@dataclass
class StsPair:
    s1: str
    s2: str
    score: float


class StsJsonl(Dataset):
    def __init__(self, path: str) -> None:
        self.rows: list[StsPair] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self.rows.append(StsPair(
                    s1=d["sentence1"], s2=d["sentence2"], score=float(d["score"]),
                ))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> StsPair:
        return self.rows[idx]


def collate(batch: list[StsPair], tok, max_seq_len: int):
    s1s = [p.s1 for p in batch]
    s2s = [p.s2 for p in batch]
    scores = torch.tensor([p.score for p in batch], dtype=torch.float32)
    enc1 = tok(s1s, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len)
    enc2 = tok(s2s, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len)
    return {"s1": enc1, "s2": enc2, "scores": scores}


def _pad_to(t: torch.Tensor, length: int, value: int = 0) -> torch.Tensor:
    cur = t.size(1)
    if cur >= length:
        return t[:, :length]
    pad = torch.full((t.size(0), length - cur), value, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=1)


# ─────────────────── forward / loss / eval ─────────────────────


def cosine_mse_step(
    model: torch.nn.Module,
    batch: dict,
    device: torch.device,
    pad_token_id: int = 0,
):
    """Single-forward DDP-safe cosine-MSE step.

    Concatenate s1 || s2 along batch axis (right-padded to common Lmax),
    one forward through SRTAdapter, split outputs, compute pairwise cosine
    and MSE against labeled score.
    """
    s1_enc = batch["s1"]
    s2_enc = batch["s2"]
    scores = batch["scores"].to(device)
    B = s1_enc["input_ids"].size(0)

    L1 = s1_enc["input_ids"].size(1)
    L2 = s2_enc["input_ids"].size(1)
    Lmax = max(L1, L2)

    ids = torch.cat([
        _pad_to(s1_enc["input_ids"], Lmax, pad_token_id),
        _pad_to(s2_enc["input_ids"], Lmax, pad_token_id),
    ], dim=0).to(device)
    masks = torch.cat([
        _pad_to(s1_enc["attention_mask"], Lmax, 0),
        _pad_to(s2_enc["attention_mask"], Lmax, 0),
    ], dim=0).to(device)

    out = model(input_ids=ids, attention_mask=masks)
    enc = F.normalize(out.community_output.encoded.float(), dim=-1)  # (2B, d)
    e1 = enc[:B]
    e2 = enc[B:]
    cos = (e1 * e2).sum(dim=-1)  # (B,)

    loss = F.mse_loss(cos, scores)
    with torch.no_grad():
        # rough proxy: pearson correlation on this minibatch
        cm = cos - cos.mean()
        sm = scores - scores.mean()
        denom = (cm.pow(2).sum().sqrt() * sm.pow(2).sum().sqrt()).clamp_min(1e-8)
        pearson = (cm * sm).sum() / denom
    return loss, {"mse": loss.item(), "pearson_batch": pearson.item()}


def _spearmanr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation between two 1-D tensors (CPU, float64)."""
    from scipy.stats import spearmanr
    r = spearmanr(a.cpu().float().numpy(), b.cpu().float().numpy())
    # scipy>=1.10 returns SignificanceResult / older returns tuple
    return float(getattr(r, "statistic", r[0]))


@torch.no_grad()
def eval_dev(
    model_inner: SRTAdapter,
    tok,
    pairs: list[StsPair],
    device: torch.device,
    max_seq_len: int,
    batch_size: int,
) -> dict[str, float]:
    model_inner.eval()

    def embed(texts: list[str]) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_seq_len)
            o = model_inner(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            v = F.normalize(o.community_output.encoded.float(), dim=-1)
            out.append(v.cpu())
        return torch.cat(out, dim=0)

    e1 = embed([p.s1 for p in pairs])
    e2 = embed([p.s2 for p in pairs])
    cos = (e1 * e2).sum(dim=-1)
    scores = torch.tensor([p.score for p in pairs], dtype=torch.float32)
    spear = _spearmanr(cos, scores)
    pear_num = ((cos - cos.mean()) * (scores - scores.mean())).sum()
    pear_den = ((cos - cos.mean()).pow(2).sum().sqrt()
                * (scores - scores.mean()).pow(2).sum().sqrt()).clamp_min(1e-8)
    pear = float(pear_num / pear_den)
    return {"val_spearman": spear, "val_pearson": pear, "n_val": len(pairs)}


# ─────────────────────────── train loop ────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--warm-start", default=None)
    p.add_argument("--train-data", required=True)
    p.add_argument("--dev-data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
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
            "DDP rank=%d/%d device=%s dtype=%s (supervised STS, cosine-MSE)",
            rank, world_size, device, args.dtype,
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

    train_ds = StsJsonl(args.train_data)
    if args.max_train_samples:
        train_ds.rows = train_ds.rows[: args.max_train_samples]
    if main_proc:
        log.info("Train rows: %d (per-rank batch=%d, world=%d, effective=%d)",
                 len(train_ds), args.batch_size, world_size,
                 args.batch_size * world_size)

    dev_pairs: list[StsPair] = []
    if main_proc:
        dev_ds = StsJsonl(args.dev_data)
        dev_pairs = dev_ds.rows
        log.info("Dev rows (rank0 only): %d", len(dev_pairs))

    def collate_fn(batch):
        return collate(batch, tok, args.max_seq_len)

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

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            for g in optim.param_groups:
                g["lr"] = lr_at(step)

            optim.zero_grad(set_to_none=True)
            loss, stats = cosine_mse_step(
                model, batch, device,
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
                    "step %d  mse %.4f  pearson_batch %.3f  lr %.2e  elapsed %.0fs",
                    step, stats["mse"], stats["pearson_batch"],
                    lr_at(step), elapsed,
                )
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")

            if dev_pairs and step > 0 and step % args.val_every == 0 and main_proc:
                vstats = eval_dev(
                    adapter_for_save, tok, dev_pairs, device,
                    args.max_seq_len, args.batch_size,
                )
                vstats["step"] = step
                vstats["kind"] = "val"
                log.info(
                    "VAL step %d  spearman %.4f  pearson %.4f  n=%d",
                    step, vstats["val_spearman"], vstats["val_pearson"],
                    vstats["n_val"],
                )
                with log_path.open("a") as f:
                    f.write(json.dumps(vstats) + "\n")
                if vstats["val_spearman"] > best_metric:
                    best_metric = vstats["val_spearman"]
                    adapter_for_save.save_adapter(str(out_dir / "best_adapter.pt"))
                    log.info("  saved best (spearman %.4f)", best_metric)
                model.train()

            if is_distributed and step > 0 and step % args.val_every == 0:
                dist.barrier()

            step += 1

        if main_proc:
            adapter_for_save.save_adapter(str(out_dir / f"adapter_epoch{epoch+1}.pt"))

    # Final eval at end-of-training (covers tail of cosine schedule)
    if dev_pairs and main_proc:
        vstats = eval_dev(
            adapter_for_save, tok, dev_pairs, device,
            args.max_seq_len, args.batch_size,
        )
        vstats["step"] = step
        vstats["kind"] = "val_final"
        log.info("FINAL VAL  spearman %.4f  pearson %.4f", vstats["val_spearman"], vstats["val_pearson"])
        with log_path.open("a") as f:
            f.write(json.dumps(vstats) + "\n")
        if vstats["val_spearman"] > best_metric:
            best_metric = vstats["val_spearman"]
            adapter_for_save.save_adapter(str(out_dir / "best_adapter.pt"))
            log.info("  saved best (spearman %.4f)", best_metric)

    if main_proc:
        adapter_for_save.save_adapter(str(out_dir / "final_adapter.pt"))
        cfg_dict = {
            "args": vars(args),
            "best_val_spearman": best_metric,
            "total_steps": step,
            "world_size": world_size,
            "effective_batch": args.batch_size * world_size,
        }
        (out_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2, default=str))
        log.info("Done. best_val_spearman=%.4f  steps=%d", best_metric, step)

    ddp_cleanup(is_distributed)


if __name__ == "__main__":
    main()
