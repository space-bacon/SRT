#!/usr/bin/env python3
"""Consolidated Phase-A read-only held-out probe.

Loads the (optionally sharded) backbone + adapter ONCE and evaluates every
read-only SRT head on a held-out set in a single pass:

  * Regime head      : ECE, AUROC, Brier, base rate (reliability bins)
  * Bifurcation r_hat : MAE, Pearson r vs r_true (masked positions)
  * Community head    : NMI vs community_id labels (KMeans on encoded)
  * Divergence path   : per-layer mean activation norm (alive check)

Designed for the Qwen3-235B Phase-A checkpoint, but works on any backbone.
Use --device-map auto for sharded backbones (pins SRT heads to --device).

Usage:
    python scripts/phaseA_probe.py \
        --backbone Qwen/Qwen3-235B-A22B-FP8 \
        --adapter checkpoints/qwen3_235b_phaseA_bs128/best_adapter.pt \
        --val-data data/all_val.jsonl \
        --device-map auto --device cuda:0 \
        --max-samples 2000 \
        --out artifacts/regime_calibration/qwen3_235b_phaseA.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srt.adapter import SRTAdapter  # noqa: E402
from srt.config import SRTConfig  # noqa: E402
from srt.data.dataset import SRTAdapterDataset, make_collate_fn  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("srt.phaseA_probe")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--adapter", required=True)
    p.add_argument("--val-data", required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda:0",
                   help="Device for SRT heads (sharded) or whole model.")
    p.add_argument("--device-map", default=None,
                   help="Pass 'auto' to shard the backbone across GPUs.")
    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-bins", type=int, default=15)
    p.add_argument("--n-clusters", type=int, default=0,
                   help="KMeans clusters for community NMI (0 = #unique ids).")
    p.add_argument("--out", required=True)
    return p.parse_args()


def ece_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int):
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
    return weighted, bins


def auroc_mwu(probs: np.ndarray, labels: np.ndarray) -> float:
    pos = probs[labels == 1]
    neg = probs[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = probs.argsort().argsort() + 1
    rank_sum_pos = ranks[labels == 1].sum()
    u = rank_sum_pos - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    config = SRTConfig(backbone_id=args.backbone, backbone_dtype=args.dtype)

    logger.info("Building model (device_map=%s) ...", args.device_map)
    if args.device_map:
        model = SRTAdapter(config, device_map=args.device_map)
        model.set_head_device(args.device)
    else:
        model = SRTAdapter(config).to(args.device, dtype=dtype)
    model.load_adapter(args.adapter)
    model.eval()
    model.backbone.eval()
    logger.info("Loaded adapter: %s", args.adapter)

    head_device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = SRTAdapterDataset(args.val_data, tokenizer,
                           max_seq_len=args.max_seq_len,
                           max_samples=args.max_samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=make_collate_fn(tokenizer.pad_token_id),
                        num_workers=2)

    regime_probs: list[np.ndarray] = []
    regime_labels: list[np.ndarray] = []
    rhat_pred: list[np.ndarray] = []
    rhat_true: list[np.ndarray] = []
    comm_vecs: list[np.ndarray] = []
    comm_ids: list[np.ndarray] = []
    div_norm_sums: dict[int, float] = {}
    div_norm_count = 0

    n_batches = 0
    for bi, batch in enumerate(loader):
        batch = {k: v.to(head_device) for k, v in batch.items()}
        # labels=None → skip CE (avoids fp32 logit OOM; probe needs heads only)
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=None,
            read_only=True,
        )

        attn = batch["attention_mask"].bool()

        # ── Regime + r_hat ────────────────────────────────────────────
        if output.ben_output is not None:
            rl = output.ben_output.regime_logits.float()  # (B,T,2)
            p_super = torch.softmax(rl, dim=-1)[..., 1]    # (B,T)
            r_true = batch["r_true"].float()
            r_mask = batch["r_mask"].bool() & attn
            if r_mask.any():
                regime_probs.append(p_super[r_mask].cpu().numpy())
                regime_labels.append((r_true[r_mask] > 0).cpu().numpy().astype(np.int8))
                rhat = output.ben_output.r_hat.float()
                rhat_pred.append(rhat[r_mask].cpu().numpy())
                rhat_true.append(r_true[r_mask].cpu().numpy())

        # ── Community encoded (B,d) per-sample ────────────────────────
        if output.community_output is not None and "community_id" in batch:
            enc = output.community_output.encoded.float()  # (B,d)
            comm_vecs.append(enc.cpu().numpy())
            comm_ids.append(batch["community_id"].cpu().numpy())

        # ── Divergence per-layer alive norms ──────────────────────────
        if output.divergences:
            for li, d in enumerate(output.divergences):
                # (B,T,h) → mean L2 over alive tokens
                dn = d.float().norm(dim=-1)  # (B,T)
                m = attn.to(dn.device)
                val = float((dn * m).sum() / m.sum().clamp_min(1))
                div_norm_sums[li] = div_norm_sums.get(li, 0.0) + val
            div_norm_count += 1

        n_batches += 1
        if (bi + 1) % 10 == 0:
            tok = sum(len(x) for x in regime_probs)
            logger.info("batch %d  regime_tokens=%d  comm_samples=%d",
                        bi + 1, tok, sum(len(x) for x in comm_ids))

    result: dict = {"adapter": args.adapter, "backbone": args.backbone,
                    "n_batches": n_batches}

    # ── Regime metrics ────────────────────────────────────────────────
    if regime_probs:
        probs = np.concatenate(regime_probs)
        labels = np.concatenate(regime_labels)
        ece, bins = ece_curve(probs, labels, args.n_bins)
        result["regime"] = {
            "n_tokens": int(len(labels)),
            "base_rate": float(labels.mean()),
            "ece": float(ece),
            "brier": float(((probs - labels) ** 2).mean()),
            "auroc": auroc_mwu(probs, labels),
            "n_bins": args.n_bins,
            "bins": bins,
        }

    # ── r_hat regression metrics ──────────────────────────────────────
    if rhat_pred:
        pred = np.concatenate(rhat_pred)
        true = np.concatenate(rhat_true)
        mae = float(np.abs(pred - true).mean())
        if pred.std() > 1e-8 and true.std() > 1e-8:
            pear = float(np.corrcoef(pred, true)[0, 1])
        else:
            pear = float("nan")
        result["r_hat"] = {"n": int(len(pred)), "mae": mae,
                           "pearson": pear,
                           "pred_mean": float(pred.mean()),
                           "true_mean": float(true.mean())}

    # ── Divergence norms ──────────────────────────────────────────────
    if div_norm_count:
        result["divergence_norms"] = {
            str(li): s / div_norm_count for li, s in sorted(div_norm_sums.items())
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Persist raw community arrays + partial result BEFORE clustering so a
    # late failure (e.g. missing sklearn) never discards the forward pass.
    if comm_vecs:
        X = np.concatenate(comm_vecs, axis=0)
        y = np.concatenate(comm_ids, axis=0)
        npz_path = out_path.with_suffix(".community.npz")
        np.savez_compressed(npz_path, X=X, y=y)
        logger.info("Saved raw community arrays to %s", npz_path)
    out_path.write_text(json.dumps(result, indent=2))

    # ── Community NMI ─────────────────────────────────────────────────
    if comm_vecs:
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import (normalized_mutual_info_score,
                                         adjusted_rand_score)
            uniq = np.unique(y)
            k = args.n_clusters if args.n_clusters > 0 else len(uniq)
            k = max(2, min(k, len(X)))
            logger.info("Community NMI: %d samples, %d unique ids, k=%d",
                        len(X), len(uniq), k)
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            pred = km.fit_predict(X)
            result["community"] = {
                "n_samples": int(len(X)),
                "n_unique_ids": int(len(uniq)),
                "k": int(k),
                "nmi": float(normalized_mutual_info_score(y, pred)),
                "ari": float(adjusted_rand_score(y, pred)),
            }
            out_path.write_text(json.dumps(result, indent=2))
        except Exception as e:  # noqa: BLE001
            logger.warning("Community clustering skipped: %s. "
                           "Raw arrays saved; compute NMI locally.", e)

    # ── Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"PHASE-A HELD-OUT PROBE — {args.adapter}")
    print("=" * 64)
    if "regime" in result:
        r = result["regime"]
        print(f"  Regime    tokens={r['n_tokens']}  base={r['base_rate']:.4f}")
        print(f"            ECE={r['ece']:.4f}  Brier={r['brier']:.4f}  AUROC={r['auroc']:.4f}")
    if "r_hat" in result:
        r = result["r_hat"]
        print(f"  r_hat     MAE={r['mae']:.4f}  Pearson={r['pearson']:.4f}  "
              f"(pred {r['pred_mean']:.3f} / true {r['true_mean']:.3f})")
    if "community" in result:
        c = result["community"]
        print(f"  Community NMI={c['nmi']:.4f}  ARI={c['ari']:.4f}  "
              f"(n={c['n_samples']}, ids={c['n_unique_ids']}, k={c['k']})")
    if "divergence_norms" in result:
        norms = result["divergence_norms"]
        s = "  ".join(f"L{li}={v:.3f}" for li, v in norms.items())
        print(f"  Div norms {s}")
    print("=" * 64)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
