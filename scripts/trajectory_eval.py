"""Trajectory metrics over the community encoder space.

Treats identity as a *trajectory* through the 64-D community encoder space
rather than an assignment over discrete prototypes. For each multi-sentence
document we compute:

  - **path length**: sum of L2 distances between successive sentence vectors
  - **curvature**: mean angle between successive direction vectors
  - **volume**: log-determinant of the per-document covariance (using a small
    ridge so the determinant is well-defined for K < D)
  - **anisotropy**: ratio of largest to smallest covariance eigenvalue

Inputs are taken from artifacts/archetype_probe/generations.jsonl by default
(986 archetype-conditioned sentences from bare Qwen). Each archetype acts as
a "document" of ~30 sentences, which is enough to estimate a covariance.

Usage:
    python3 scripts/trajectory_eval.py --adapter <ckpt> --tag <name>
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from srt.adapter import SRTAdapter
from srt.config import SRTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("trajectory_eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--adapter", required=True)
    p.add_argument("--generations", default="artifacts/archetype_probe/generations.jsonl")
    p.add_argument("--tag", required=True)
    p.add_argument("--output-root", default="artifacts/trajectory")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--ridge", type=float, default=1e-4,
                   help="ridge added to covariance for log-det stability")
    return p.parse_args()


def trajectory_metrics(vectors: np.ndarray, ridge: float) -> dict[str, float]:
    """Compute trajectory metrics for one document (sequence of vectors).

    Args:
        vectors: (N, D) per-sentence community vectors in document order.
        ridge: small constant added to covariance diagonal for stability.

    Returns:
        Dict with path_length, mean_curvature, log_det_cov, anisotropy.
    """
    N, D = vectors.shape
    if N < 2:
        return {
            "path_length": 0.0, "mean_curvature": 0.0,
            "log_det_cov": float("nan"), "anisotropy": float("nan"),
            "n_points": int(N),
        }

    # Path length: sum of L2 step sizes
    steps = np.diff(vectors, axis=0)  # (N-1, D)
    step_norms = np.linalg.norm(steps, axis=-1)
    path_length = float(step_norms.sum())

    # Curvature: mean angle between successive step directions
    if N >= 3:
        u = steps[:-1] / (np.linalg.norm(steps[:-1], axis=-1, keepdims=True) + 1e-8)
        v = steps[1:] / (np.linalg.norm(steps[1:], axis=-1, keepdims=True) + 1e-8)
        cos_angle = (u * v).sum(axis=-1).clip(-1.0, 1.0)
        mean_curvature = float(np.arccos(cos_angle).mean())
    else:
        mean_curvature = 0.0

    # Volume: log-det of covariance with ridge
    cov = np.cov(vectors, rowvar=False)  # (D, D)
    cov = cov + ridge * np.eye(D)
    sign, logdet = np.linalg.slogdet(cov)
    log_det_cov = float(logdet) if sign > 0 else float("nan")

    # Anisotropy: ratio of largest to smallest eigenvalue
    eigvals = np.linalg.eigvalsh(cov)
    anisotropy = float(eigvals[-1] / max(eigvals[0], ridge * 0.5))

    return {
        "path_length": path_length,
        "mean_curvature": mean_curvature,
        "log_det_cov": log_det_cov,
        "anisotropy": anisotropy,
        "n_points": int(N),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: list[dict] = []
    with Path(args.generations).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    logger.info("Loaded %d sentences", len(rows))

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = SRTConfig(backbone_id=args.backbone, backbone_dtype=args.dtype)
    model = SRTAdapter(config).to(device)
    state = torch.load(args.adapter, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("Loaded adapter; missing=%d unexpected=%d", len(missing), len(unexpected))
    model.eval()

    vectors_all: list[np.ndarray] = []
    arch_all: list[int] = []
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i : i + args.batch_size]
        texts = [r["text"] for r in batch]
        enc = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_seq_len,
        ).to(device)
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        v = out.community_output.vector.detach().float().cpu().numpy()
        vectors_all.append(v)
        arch_all.extend(r["archetype_id"] for r in batch)
        if (i // args.batch_size) % 10 == 0:
            logger.info("processed %d / %d", i + len(batch), len(rows))

    V = np.concatenate(vectors_all, axis=0)
    A = np.asarray(arch_all, dtype=np.int64)

    # Per-archetype trajectory metrics
    by_arch: dict[int, list[np.ndarray]] = defaultdict(list)
    for vec, a in zip(V, A):
        by_arch[int(a)].append(vec)

    per_archetype: dict[int, dict[str, float]] = {}
    for a, vecs in by_arch.items():
        per_archetype[a] = trajectory_metrics(np.stack(vecs), ridge=args.ridge)

    # Aggregate
    keys = ["path_length", "mean_curvature", "log_det_cov", "anisotropy"]
    agg = {}
    for k in keys:
        vals = [m[k] for m in per_archetype.values()
                if m[k] == m[k]]  # filter nan
        agg[f"mean_{k}"] = float(np.mean(vals)) if vals else float("nan")
        agg[f"std_{k}"] = float(np.std(vals)) if vals else float("nan")

    # Global trajectory across ALL points (treats whole corpus as one sequence)
    global_metrics = trajectory_metrics(V, ridge=args.ridge)

    results = {
        "tag": args.tag,
        "n_sentences": int(len(V)),
        "n_archetypes": int(len(by_arch)),
        "ridge": args.ridge,
        "aggregate_per_archetype": agg,
        "global_trajectory": global_metrics,
        "per_archetype": {int(a): m for a, m in per_archetype.items()},
    }

    out_dir = Path(args.output_root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    np.savez(out_dir / "vectors.npz", V=V, A=A)
    logger.info("Wrote %s", out_dir / "results.json")
    logger.info(
        "mean_path=%.3f mean_curv=%.3f mean_logdet=%.3f mean_aniso=%.1f",
        agg["mean_path_length"], agg["mean_mean_curvature"],
        agg["mean_log_det_cov"], agg["mean_anisotropy"],
    )


if __name__ == "__main__":
    main()
