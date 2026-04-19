#!/usr/bin/env python3
"""Train SRT Adapter on JSONL data.

Usage:
    python scripts/train.py \
        --backbone Qwen/Qwen2.5-7B \
        --train-data data/all_train.jsonl \
        --val-data data/all_val.jsonl \
        --output-dir checkpoints/adapter_v1 \
        --batch-size 16 \
        --epochs 3 \
        --lr 3e-4

The backbone is frozen — only the lightweight adapter modules (~11M params)
are trained.  CE loss comes from the backbone's native LM head, so language
quality starts at pretrained level (~3.5 CE).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from srt.adapter import SRTAdapter
from srt.config import SRTConfig
from srt.data.dataset import SRTAdapterDataset, make_collate_fn
from srt.training.losses import compute_total_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("srt.train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SRT Adapter")
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B", help="HF model id")
    p.add_argument("--train-data", required=True, help="Path to train JSONL")
    p.add_argument("--val-data", default=None, help="Path to val JSONL")
    p.add_argument("--output-dir", default="checkpoints/adapter", help="Checkpoint dir")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--val-every", type=int, default=1000, help="Validate every N steps")
    p.add_argument("--log-every", type=int, default=100, help="Log every N steps")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default=None, help="Device (auto-detected if not set)")
    p.add_argument("--resume", default=None, help="Path to full training checkpoint to resume from")
    return p.parse_args()


def get_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(
    model: SRTAdapter, lr: float, warmup_steps: int, total_steps: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """AdamW on trainable params with linear warmup + cosine decay."""
    trainable = list(model.trainable_parameters())
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


@torch.no_grad()
def validate(model: SRTAdapter, dataloader: DataLoader, config: SRTConfig) -> dict[str, float]:
    """Run validation and return average metrics."""
    model.eval()
    totals: dict[str, float] = {}
    count = 0

    for batch in dataloader:
        batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        _, metrics = compute_total_loss(
            output,
            model.chain_predictor,
            batch["r_true"],
            batch["r_mask"],
            config.loss,
            attention_mask=batch["attention_mask"],
        )
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        count += 1

    model.train()
    model.backbone.eval()  # backbone must stay in eval mode always
    return {k: v / max(count, 1) for k, v in totals.items()}


def train(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    logger.info("Device: %s", device)

    # ── Config ────────────────────────────────────────────────────────
    config = SRTConfig(
        backbone_id=args.backbone,
        backbone_dtype=args.dtype,
    )
    config.training.batch_size = args.batch_size
    config.training.epochs = args.epochs
    config.training.lr = args.lr
    config.training.max_seq_len = args.max_seq_len
    config.training.grad_clip = args.grad_clip
    config.training.warmup_steps = args.warmup_steps
    config.training.val_every = args.val_every
    config.training.log_every = args.log_every

    # ── Model ─────────────────────────────────────────────────────────
    model = SRTAdapter(config)
    model = model.to(device)

    # Only put adapter params in training mode (backbone stays eval)
    model.train()
    model.backbone.eval()

    # ── Data ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collate_fn = make_collate_fn(pad_token_id=tokenizer.pad_token_id)

    train_ds = SRTAdapterDataset(
        args.train_data, tokenizer, args.max_seq_len, args.max_train_samples
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = None
    if args.val_data:
        val_ds = SRTAdapterDataset(
            args.val_data, tokenizer, args.max_seq_len, args.max_val_samples
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )

    # ── Optimizer ─────────────────────────────────────────────────────
    total_steps = len(train_loader) * args.epochs
    optimizer, scheduler = build_optimizer(model, args.lr, args.warmup_steps, total_steps)
    logger.info(
        "Training: %d steps (%d epochs × %d batches)",
        total_steps,
        args.epochs,
        len(train_loader),
    )

    # ── Output dir ────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        import dataclasses
        json.dump(dataclasses.asdict(config), f, indent=2, default=str)

    # ── Resume from checkpoint ────────────────────────────────────────
    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if "adapter_state" in ckpt:
            # Full training checkpoint
            missing, _ = model.load_state_dict(ckpt["adapter_state"], strict=False)
            non_bb = [k for k in missing if not k.startswith("backbone.")]
            if non_bb:
                logger.warning("Missing adapter keys: %s", non_bb)
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            global_step = ckpt["global_step"]
            start_epoch = ckpt["epoch"]
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            logger.info(
                "Resumed from checkpoint: step=%d, epoch=%d, best_val=%.4f",
                global_step, start_epoch, best_val_loss,
            )
        else:
            # Legacy: plain adapter weights only
            model.load_adapter(args.resume)
            logger.info("Loaded adapter weights (no optimizer state) from %s", args.resume)

    # ── Training loop ─────────────────────────────────────────────────
    log_file = open(out_dir / "train_log.jsonl", "a")

    for epoch in range(start_epoch, args.epochs):
        logger.info("=== Epoch %d/%d ===", epoch + 1, args.epochs)
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            # Compute combined loss
            loss, metrics = compute_total_loss(
                output,
                model.chain_predictor,
                batch["r_true"],
                batch["r_mask"],
                config.loss,
                attention_mask=batch["attention_mask"],
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.trainable_parameters(), args.grad_clip
                )
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += metrics["total"]
            epoch_steps += 1

            # Logging
            if global_step % args.log_every == 0:
                elapsed = time.time() - t0
                steps_per_sec = epoch_steps / elapsed
                lr_now = scheduler.get_last_lr()[0]

                # ── Diagnostic: divergence norms, r_hat stats, injection norms
                with torch.no_grad():
                    div_norms = [d.norm(dim=-1).mean().item() for d in output.divergences]
                    inj_norms = [inj.norm(dim=-1).mean().item() for inj in output.injections]
                    diag = {
                        "div_norm_mean": round(sum(div_norms) / max(len(div_norms), 1), 5),
                        "div_norms": [round(n, 5) for n in div_norms],
                        "inj_norms": [round(n, 5) for n in inj_norms],
                    }
                    if output.ben_output is not None:
                        rh = output.ben_output.r_hat
                        diag["r_hat_mean"] = round(rh.mean().item(), 5)
                        diag["r_hat_std"] = round(rh.std().item(), 5)
                        diag["r_hat_min"] = round(rh.min().item(), 5)
                        diag["r_hat_max"] = round(rh.max().item(), 5)

                log_entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "lr": lr_now,
                    "steps_per_sec": round(steps_per_sec, 2),
                    **{k: round(v, 4) for k, v in metrics.items()},
                    **diag,
                }
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()
                logger.info(
                    "step=%d  total=%.3f  ce=%.3f  chain=%.4f  bif=%.4f  lr=%.2e  %.1f step/s",
                    global_step,
                    metrics.get("total", 0),
                    metrics.get("ce", 0),
                    metrics.get("chain", 0),
                    metrics.get("bif", 0),
                    lr_now,
                    steps_per_sec,
                )
                logger.info(
                    "  diag: div_norms=%s  inj_norms=%s  r_hat=%.4f±%.4f [%.4f, %.4f]",
                    diag["div_norms"],
                    diag["inj_norms"],
                    diag.get("r_hat_mean", 0),
                    diag.get("r_hat_std", 0),
                    diag.get("r_hat_min", 0),
                    diag.get("r_hat_max", 0),
                )

            # Validation
            if val_loader and global_step % args.val_every == 0:
                val_metrics = validate(model, val_loader, config)
                val_loss = val_metrics.get("total", float("inf"))
                logger.info(
                    "VAL step=%d  total=%.3f  ce=%.3f  bif=%.4f",
                    global_step,
                    val_metrics.get("total", 0),
                    val_metrics.get("ce", 0),
                    val_metrics.get("bif", 0),
                )
                log_entry = {"step": global_step, "val": True, **val_metrics}
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save_adapter(str(out_dir / "best_adapter.pt"))
                    logger.info("New best val loss: %.4f", val_loss)

                # Save full training checkpoint for resumption
                torch.save({
                    "adapter_state": {
                        k: v for k, v in model.state_dict().items()
                        if k.startswith(model._ADAPTER_PREFIXES)
                    },
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                }, str(out_dir / "training_checkpoint.pt"))
                logger.info("Saved training checkpoint at step %d", global_step)

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(
            "Epoch %d complete — avg_loss=%.3f, %.1f min",
            epoch + 1,
            avg_epoch_loss,
            (time.time() - t0) / 60,
        )
        model.save_adapter(str(out_dir / f"adapter_epoch{epoch + 1}.pt"))

    log_file.close()
    model.save_adapter(str(out_dir / "final_adapter.pt"))
    logger.info("Training complete. Best val loss: %.4f", best_val_loss)


if __name__ == "__main__":
    train(parse_args())
