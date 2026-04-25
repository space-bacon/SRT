#!/usr/bin/env python3
"""Instrument-channel evaluation for SRT Adapter (Tier-1 item D + E).

Runs one inference pass over val data and emits the artifacts needed to
demonstrate two SRT-only capabilities:

  D. Per-token r̂ heatmap data  →  artifacts/instrument/<tag>/traces.jsonl
     (one passage per line, with tokens + r_hat + regime per position)

  E. Community-vector retrieval eval  →  artifacts/instrument/<tag>/community_metrics.json
     - within-community vs between-community mean cosine
     - their ratio (the v3 baseline collapsed to ≈ 1.000)
     - k-NN community recall@1, @5, @10 (probe-classifier-free)

The rendering of D into HTML is the responsibility of
scripts/render_heatmap.py.

Usage:
    python scripts/instrument_eval.py \\
        --adapter checkpoints/adapter_v3/best_adapter.pt \\
        --val-data data/all_val.jsonl \\
        --tag v3 \\
        --max-samples 2000 \\
        --trace-samples 60
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
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
logger = logging.getLogger("srt.instrument")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--adapter", required=True)
    p.add_argument("--val-data", required=True)
    p.add_argument("--tag", required=True, help="e.g. v3, v4_step5000")
    p.add_argument("--output-root", default="artifacts/instrument")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--trace-samples", type=int, default=60,
                   help="Passages to dump full per-token traces for")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default=None)
    return p.parse_args()


def get_device(req: str | None) -> torch.device:
    if req:
        return torch.device(req)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def run_eval(
    model: SRTAdapter,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    out_dir: Path,
    trace_samples: int,
) -> dict:
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    traces_path = out_dir / "traces.jsonl"
    vectors_chunks: list[np.ndarray] = []
    ids_chunks: list[np.ndarray] = []

    n_traced = 0
    t0 = time.time()
    n_samples = 0

    with traces_path.open("w") as ftr:
        for batch_i, batch in enumerate(dataloader):
            comm_ids = batch.pop("community_id").tolist()
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            B = batch["input_ids"].shape[0]
            att = batch["attention_mask"].bool()

            # Per-sample community vectors (always collected)
            if out.community_output is not None:
                vec = out.community_output.vector.detach().float().cpu().numpy()
                vectors_chunks.append(vec)
                ids_chunks.append(np.asarray(comm_ids, dtype=np.int64))

            # Per-token traces only for first trace_samples passages
            for s in range(B):
                if n_traced >= trace_samples:
                    break
                seq_len = int(att[s].sum())
                token_ids = batch["input_ids"][s, :seq_len].tolist()
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                trace = {
                    "tokens": tokens,
                    "r_hat": out.ben_output.r_hat[s, :seq_len].float().cpu().tolist(),
                    "r_true": batch["r_true"][s, :seq_len].float().cpu().tolist(),
                    "r_mask": [bool(x) for x in batch["r_mask"][s, :seq_len].cpu().tolist()],
                    "regime_pred": out.ben_output.regime_logits[s, :seq_len]
                        .argmax(dim=-1).cpu().tolist(),
                    "community_id_true": int(comm_ids[s]) if comm_ids[s] is not None else None,
                    "community_pred": int(out.community_output.weights[s].argmax())
                        if out.community_output is not None
                        and out.community_output.weights is not None
                        else None,
                }
                ftr.write(json.dumps(trace) + "\n")
                n_traced += 1

            n_samples += B
            if (batch_i + 1) % 10 == 0:
                rate = n_samples / (time.time() - t0)
                logger.info("batch %d  samples=%d  %.1f samp/s",
                            batch_i + 1, n_samples, rate)

    # Dump community vectors
    vectors = np.concatenate(vectors_chunks, axis=0)  # (N, d)
    cids = np.concatenate(ids_chunks, axis=0)         # (N,)
    np.savez(out_dir / "community_vectors.npz", vectors=vectors, ids=cids)
    logger.info("Wrote %d community vectors (d=%d)", vectors.shape[0], vectors.shape[1])

    # ── Community separation metrics ────────────────────────────────
    metrics = community_metrics(vectors, cids)
    metrics["n_samples"] = int(n_samples)
    metrics["n_traced"] = int(n_traced)
    metrics["wall_time_sec"] = round(time.time() - t0, 2)

    with (out_dir / "community_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def community_metrics(vectors: np.ndarray, ids: np.ndarray) -> dict:
    """Within / between cosine + k-NN community recall.

    Args:
        vectors: (N, d) per-sample community vectors (NOT prototypes).
        ids: (N,) integer community ids (-1 = unknown, dropped).

    Returns:
        Dict of metrics.
    """
    keep = ids >= 0
    vectors = vectors[keep]
    ids = ids[keep]
    N = len(ids)
    logger.info("Computing community metrics on N=%d vectors, "
                "%d distinct ids", N, len(np.unique(ids)))

    if N < 2:
        return {"error": "not enough samples"}

    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-8)
    v = vectors / norms

    # Pairwise cosine; for very large N this would be infeasible —
    # we cap N at 5000 by random subsample to keep memory bounded.
    if N > 5000:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, 5000, replace=False)
        v_sub, ids_sub = v[idx], ids[idx]
        N_sub = 5000
    else:
        v_sub, ids_sub, N_sub = v, ids, N

    sim = v_sub @ v_sub.T  # (N_sub, N_sub)
    eye = np.eye(N_sub, dtype=bool)
    same = ids_sub[:, None] == ids_sub[None, :]
    diff = ~same

    # Within = same-id, off-diagonal
    within_mask = same & ~eye
    between_mask = diff
    within_cos = float(sim[within_mask].mean()) if within_mask.any() else float("nan")
    between_cos = float(sim[between_mask].mean()) if between_mask.any() else float("nan")
    ratio = within_cos / between_cos if between_cos != 0 else float("nan")

    # k-NN community recall: for each anchor, its top-k neighbors
    # (excluding self) — what fraction share its community?
    sim_no_self = sim.copy()
    np.fill_diagonal(sim_no_self, -np.inf)
    recalls = {}
    for k in (1, 5, 10):
        if k >= N_sub:
            continue
        topk = np.argpartition(-sim_no_self, k, axis=1)[:, :k]  # (N_sub, k)
        neighbors_match = (ids_sub[topk] == ids_sub[:, None]).any(axis=1) \
            if k == 1 else (ids_sub[topk] == ids_sub[:, None]).mean(axis=1)
        if k == 1:
            # recall@1 = fraction whose nearest neighbor matches
            recalls[f"recall@{k}"] = float(neighbors_match.mean())
        else:
            # recall@k = mean fraction of top-k that match
            recalls[f"recall@{k}"] = float(neighbors_match.mean())

    # Random baseline = 1 / num_classes (as a comparable line)
    n_classes = len(np.unique(ids_sub))
    recalls["random_baseline_recall"] = 1.0 / n_classes

    return {
        "n_subsample": int(N_sub),
        "n_classes": int(n_classes),
        "within_class_cos_mean": within_cos,
        "between_class_cos_mean": between_cos,
        "within_over_between_ratio": float(ratio),
        **recalls,
    }


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    out_dir = Path(args.output_root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Tag: %s  device: %s  out: %s", args.tag, device, out_dir)

    config = SRTConfig(backbone_id=args.backbone, backbone_dtype=args.dtype)
    logger.info("Building model...")
    model = SRTAdapter(config).to(device)
    state = torch.load(args.adapter, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("Loaded adapter: missing=%d unexpected=%d",
                len(missing), len(unexpected))
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected[:5])

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = SRTAdapterDataset(
        args.val_data, tokenizer,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
    )
    base_collate = make_collate_fn(tokenizer.pad_token_id)

    class WithComm(torch.utils.data.Dataset):
        def __init__(self, b): self.b = b
        def __len__(self): return len(self.b)
        def __getitem__(self, i):
            it = self.b[i]
            row = self.b.samples[i]
            cid = row.get("community_id", -1)
            if cid is None:
                cid = -1
            it["community_id"] = torch.tensor(int(cid))
            return it

    def collate_with_comm(batch):
        cids = torch.stack([b.pop("community_id") for b in batch])
        out = base_collate(batch)
        out["community_id"] = cids
        return out

    loader = DataLoader(
        WithComm(base),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_with_comm,
        num_workers=2,
    )

    metrics = run_eval(model, loader, tokenizer, device, out_dir, args.trace_samples)

    # Summary
    print("\n" + "=" * 60)
    print(f"INSTRUMENT EVAL — {args.tag}")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:40s} {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
