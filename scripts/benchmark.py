#!/usr/bin/env python3
"""Benchmark a trained SRT adapter on validation data.

Computes headline metrics, dumps per-token traces, and emits diagnostic
arrays needed for plots and the eventual paper / Substack post.

Usage:
    python scripts/benchmark.py \
        --adapter checkpoints/adapter_v3/best_adapter.pt \
        --val-data data/all_val.jsonl \
        --output-dir artifacts/benchmark \
        --max-samples 5000 \
        --trace-samples 25
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from srt.adapter import SRTAdapter
from srt.config import SRTConfig
from srt.data.dataset import SRTAdapterDataset, make_collate_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("srt.bench")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark SRT Adapter")
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--adapter", required=True, help="Path to adapter weights .pt")
    p.add_argument("--val-data", required=True, help="Path to val JSONL")
    p.add_argument("--output-dir", default="artifacts/benchmark")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--trace-samples", type=int, default=25,
                   help="Number of passages to dump per-token traces for")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default=None)
    p.add_argument("--ablate-rrm", action="store_true",
                   help="Zero out RRM injections at runtime (ablation study)")
    return p.parse_args()


def get_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation coefficient between two 1D tensors."""
    x = x.float()
    y = y.float()
    xm = x - x.mean()
    ym = y - y.mean()
    num = (xm * ym).sum()
    den = (xm.pow(2).sum().sqrt() * ym.pow(2).sum().sqrt()).clamp(min=1e-8)
    return float(num / den)


@torch.no_grad()
def benchmark(
    model: SRTAdapter,
    dataloader: DataLoader,
    device: torch.device,
    trace_samples: int,
) -> dict:
    """Run benchmark pass and collect metrics + traces."""
    model.eval()

    # Headline accumulators
    n_samples = 0
    n_tokens_masked = 0
    ce_sum = 0.0  # weighted by token count for true mean
    ce_token_count = 0

    # Aggregated tensors (concatenated across batches)
    r_hat_all: list[torch.Tensor] = []
    r_true_all: list[torch.Tensor] = []
    r_true_compressed_all: list[torch.Tensor] = []
    regime_pred_all: list[torch.Tensor] = []
    regime_true_all: list[torch.Tensor] = []

    # Community: (n_samples, K) soft assignments + ground-truth community_id per sample
    community_weights_all: list[torch.Tensor] = []
    community_id_all: list[int] = []

    # Diag norms per MAH layer (concat across batch & token positions, masked)
    div_norms_per_layer: dict[int, list[torch.Tensor]] = defaultdict(list)
    inj_norms_per_layer: dict[int, list[torch.Tensor]] = defaultdict(list)

    # Per-community CE accumulators
    ce_per_comm: dict[int, list[float]] = defaultdict(list)

    # Per-token traces for first trace_samples passages
    traces: list[dict] = []

    t0 = time.time()
    for batch_i, batch in enumerate(dataloader):
        # community_id is not in collate output — recover from dataset wrappers if needed
        comm_ids = batch.pop("community_id", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        if comm_ids is not None:
            comm_ids = comm_ids.tolist()

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        B, T = batch["input_ids"].shape
        att = batch["attention_mask"].bool()
        r_mask = batch["r_mask"].bool() & att

        # CE: use sample-level shift CE — recompute per-sample, not the global mean
        # The model returns a scalar mean ce_loss, so we use that for the running global mean.
        if out.ce_loss is not None:
            # Approximate token count for weighting
            n_target_tokens = att[:, 1:].sum().item()
            ce_sum += float(out.ce_loss) * n_target_tokens
            ce_token_count += n_target_tokens

        # r_hat / r_true at masked positions
        if r_mask.any():
            r_hat_m = out.ben_output.r_hat[r_mask]  # (N,)
            r_true_m = batch["r_true"][r_mask]
            r_true_compressed = r_true_m.sign() * (1.0 + r_true_m.abs()).log()
            r_hat_all.append(r_hat_m.detach().cpu())
            r_true_all.append(r_true_m.detach().cpu())
            r_true_compressed_all.append(r_true_compressed.detach().cpu())

            regime_pred = out.ben_output.regime_logits[r_mask].argmax(dim=-1)
            regime_true = (r_true_m > 0).long()
            regime_pred_all.append(regime_pred.detach().cpu())
            regime_true_all.append(regime_true.detach().cpu())
            n_tokens_masked += int(r_mask.sum())

        # Community assignments (one per sample)
        if out.community_output is not None:
            community_weights_all.append(
                out.community_output.weights.detach().float().cpu()
            )
            if comm_ids is not None:
                community_id_all.extend(comm_ids)
            else:
                community_id_all.extend([-1] * B)

        # Divergence / injection norms per MAH layer
        for li, div in enumerate(out.divergences):
            norms = div.norm(dim=-1)  # (B, T)
            masked = norms[att].detach().float().cpu()
            div_norms_per_layer[li].append(masked)
        for li, inj in enumerate(out.injections):
            norms = inj.norm(dim=-1)
            masked = norms[att].detach().float().cpu()
            inj_norms_per_layer[li].append(masked)

        # Per-token traces for first N passages in batch 0
        if batch_i == 0 and trace_samples > 0:
            tokenizer = AutoTokenizer.from_pretrained(model.config.backbone_id)
            n_to_take = min(trace_samples, B)
            for s in range(n_to_take):
                seq_len = int(att[s].sum())
                tokens = tokenizer.convert_ids_to_tokens(
                    batch["input_ids"][s, :seq_len].tolist()
                )
                trace = {
                    "tokens": tokens,
                    "r_hat": out.ben_output.r_hat[s, :seq_len].float().cpu().tolist(),
                    "r_true": batch["r_true"][s, :seq_len].float().cpu().tolist(),
                    "r_mask": batch["r_mask"][s, :seq_len].cpu().tolist(),
                    "regime_pred": out.ben_output.regime_logits[s, :seq_len]
                        .argmax(dim=-1).cpu().tolist(),
                    "community_id_true": comm_ids[s] if comm_ids else None,
                    "community_pred": int(out.community_output.weights[s].argmax())
                        if out.community_output else None,
                    "community_weights": out.community_output.weights[s]
                        .float().cpu().tolist() if out.community_output else None,
                }
                traces.append(trace)

        n_samples += B
        if (batch_i + 1) % 10 == 0:
            dt = time.time() - t0
            logger.info(
                "batch %d  samples=%d  rate=%.2f samp/s",
                batch_i + 1, n_samples, n_samples / dt,
            )

    # ── Aggregate ────────────────────────────────────────────────────
    r_hat_cat = torch.cat(r_hat_all) if r_hat_all else torch.tensor([])
    r_true_cat = torch.cat(r_true_all) if r_true_all else torch.tensor([])
    r_true_comp_cat = (torch.cat(r_true_compressed_all)
                       if r_true_compressed_all else torch.tensor([]))
    regime_pred_cat = torch.cat(regime_pred_all) if regime_pred_all else torch.tensor([])
    regime_true_cat = torch.cat(regime_true_all) if regime_true_all else torch.tensor([])

    metrics: dict = {}
    metrics["n_samples"] = n_samples
    metrics["n_tokens_masked"] = n_tokens_masked
    metrics["mean_ce"] = ce_sum / max(ce_token_count, 1)
    metrics["wall_time_sec"] = time.time() - t0

    if r_hat_cat.numel() > 0:
        metrics["pearson_r_hat_vs_r_true_raw"] = pearson(r_hat_cat, r_true_cat)
        metrics["pearson_r_hat_vs_r_true_compressed"] = pearson(r_hat_cat, r_true_comp_cat)
        metrics["r_hat_mean"] = float(r_hat_cat.mean())
        metrics["r_hat_std"] = float(r_hat_cat.std())
        metrics["r_hat_min"] = float(r_hat_cat.min())
        metrics["r_hat_max"] = float(r_hat_cat.max())
        metrics["r_hat_saturated_pos"] = float((r_hat_cat >= 0.99).float().mean())
        metrics["r_hat_saturated_neg"] = float((r_hat_cat <= -0.99).float().mean())

    if regime_pred_cat.numel() > 0:
        metrics["regime_accuracy"] = float(
            (regime_pred_cat == regime_true_cat).float().mean()
        )
        # Per-class breakdown
        sub_mask = regime_true_cat == 0
        sup_mask = regime_true_cat == 1
        if sub_mask.any():
            metrics["regime_accuracy_subcritical"] = float(
                (regime_pred_cat[sub_mask] == 0).float().mean()
            )
        if sup_mask.any():
            metrics["regime_accuracy_supercritical"] = float(
                (regime_pred_cat[sup_mask] == 1).float().mean()
            )
        metrics["regime_true_pos_frac"] = float(sup_mask.float().mean())

    # Per-layer norm stats
    metrics["divergence_norms_per_layer"] = {}
    for li, chunks in div_norms_per_layer.items():
        cat = torch.cat(chunks)
        metrics["divergence_norms_per_layer"][li] = {
            "mean": float(cat.mean()),
            "std": float(cat.std()),
            "min": float(cat.min()),
            "max": float(cat.max()),
            "median": float(cat.median()),
        }
    metrics["injection_norms_per_layer"] = {}
    for li, chunks in inj_norms_per_layer.items():
        cat = torch.cat(chunks)
        metrics["injection_norms_per_layer"][li] = {
            "mean": float(cat.mean()),
            "std": float(cat.std()),
            "min": float(cat.min()),
            "max": float(cat.max()),
            "median": float(cat.median()),
        }

    # Community: prototype × community_id co-occurrence matrix
    if community_weights_all and community_id_all:
        cw = torch.cat(community_weights_all)  # (N, K)
        ids = torch.tensor(community_id_all)
        unique_ids = sorted(set(community_id_all) - {-1})
        K = cw.shape[1]
        co = torch.zeros(len(unique_ids), K)
        for i, cid in enumerate(unique_ids):
            mask = ids == cid
            if mask.any():
                co[i] = cw[mask].mean(dim=0)
        metrics["community_protocol_activation"] = {
            "community_ids": unique_ids,
            "matrix": co.tolist(),  # (n_communities, K)
            "K": K,
        }
        # Cluster purity: for each community, fraction of mass on its top prototype
        purities = []
        for i, cid in enumerate(unique_ids):
            purities.append(float(co[i].max()))
        metrics["community_top_prototype_mass_mean"] = sum(purities) / len(purities)
        # Community separation: pairwise cosine distance between community profiles
        co_norm = co / co.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sims = co_norm @ co_norm.T  # (n, n)
        metrics["community_pairwise_cos_sim_mean"] = float(
            (sims.sum() - sims.diag().sum())
            / max(sims.numel() - sims.shape[0], 1)
        )

    # Per-community CE — recompute from raw if needed (skipped here, would need per-sample CE)

    return {"metrics": metrics, "traces": traces}


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s, dtype: %s", device, args.dtype)
    logger.info("Adapter: %s", args.adapter)
    logger.info("Val data: %s", args.val_data)

    # Build config matching training
    config = SRTConfig(backbone_id=args.backbone, backbone_dtype=args.dtype)

    # Load model
    logger.info("Building model...")
    model = SRTAdapter(config).to(device)
    state = torch.load(args.adapter, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("Loaded adapter. missing=%d unexpected=%d",
                len(missing), len(unexpected))

    # RRM ablation: monkey-patch inject() to return zeros
    if args.ablate_rrm:
        logger.info("ABLATION: zeroing RRM injections at runtime")
        original_inject = model.rrm.inject

        def zero_inject(meta_state, h):
            out = original_inject(meta_state, h)
            return torch.zeros_like(out)

        model.rrm.inject = zero_inject

    # Tokenizer + dataset
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = SRTAdapterDataset(
        args.val_data, tokenizer,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
    )

    # Custom collate that also forwards community_id
    base_collate = make_collate_fn(tokenizer.pad_token_id)

    # Subclass dataset to also emit community_id
    class DatasetWithComm(torch.utils.data.Dataset):
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            item = self.base[idx]
            row = self.base.samples[idx]
            cid = row.get("community_id", -1)
            if cid is None:
                cid = -1
            item["community_id"] = torch.tensor(int(cid))
            return item

    def collate_extended(batch):
        comm_ids = torch.stack([b.pop("community_id") for b in batch])
        out = base_collate(batch)
        out["community_id"] = comm_ids
        return out

    wrapped = DatasetWithComm(dataset)
    dataloader = DataLoader(
        wrapped,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_extended,
        num_workers=2,
    )

    logger.info("Running benchmark on %d samples...", len(dataset))
    result = benchmark(model, dataloader, device, args.trace_samples)

    # Write outputs
    metrics_path = out_dir / "metrics.json"
    traces_path = out_dir / "traces.json"
    with open(metrics_path, "w") as f:
        json.dump(result["metrics"], f, indent=2)
    with open(traces_path, "w") as f:
        json.dump(result["traces"], f, indent=2)

    logger.info("Wrote %s", metrics_path)
    logger.info("Wrote %s", traces_path)

    # Print headline summary
    m = result["metrics"]
    print("\n" + "=" * 60)
    print("HEADLINE METRICS")
    print("=" * 60)
    print(f"Samples evaluated:   {m['n_samples']}")
    print(f"Tokens with r_true:  {m['n_tokens_masked']}")
    print(f"Mean CE:             {m['mean_ce']:.4f}")
    if "pearson_r_hat_vs_r_true_compressed" in m:
        print(f"Pearson r̂ vs r_true (raw):        {m['pearson_r_hat_vs_r_true_raw']:.4f}")
        print(f"Pearson r̂ vs r_true (compressed): "
              f"{m['pearson_r_hat_vs_r_true_compressed']:.4f}")
    if "regime_accuracy" in m:
        print(f"Regime accuracy (overall):        {m['regime_accuracy']:.4f}")
        print(f"  subcritical:                    "
              f"{m.get('regime_accuracy_subcritical', float('nan')):.4f}")
        print(f"  supercritical:                  "
              f"{m.get('regime_accuracy_supercritical', float('nan')):.4f}")
        print(f"  true supercritical fraction:    {m['regime_true_pos_frac']:.4f}")
    if "r_hat_mean" in m:
        print(f"r̂ distribution: mean={m['r_hat_mean']:.3f} "
              f"std={m['r_hat_std']:.3f} "
              f"min={m['r_hat_min']:.3f} max={m['r_hat_max']:.3f}")
        print(f"  saturated +1: {m['r_hat_saturated_pos']:.4f}  "
              f"saturated -1: {m['r_hat_saturated_neg']:.4f}")
    if "community_top_prototype_mass_mean" in m:
        print(f"Community top-prototype mass:     "
              f"{m['community_top_prototype_mass_mean']:.4f}")
        print(f"Community pairwise cos sim:       "
              f"{m['community_pairwise_cos_sim_mean']:.4f}")
    print("=" * 60)
    print(f"Wall time: {m['wall_time_sec']:.1f}s")


if __name__ == "__main__":
    main()
