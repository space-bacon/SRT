"""Archetype-conditioned community-head probe.

Tests whether v7's 32 community prototypes carry features that align with
Lancaster's 33 archetypes (paired with the Lexicon of Synthetic Interiority).

Procedure:
  1. Load generations.jsonl produced by scripts/archetype_generate.py.
     Each row: {archetype_id, archetype_name, topic, text}.
  2. Load v7 SRTAdapter from --adapter checkpoint.
  3. For each text, run a forward pass and capture
     CommunityOutput.weights (B, 32) softmax over prototypes
     CommunityOutput.vector  (B, 64) mixture vector in community space
  4. Group by archetype. Compute:
       - per-archetype mean weight vector (33, 32)
       - per-archetype mean 64D vector    (33, 64)
       - pairwise cosine separation between archetype means (33, 33)
       - within-archetype variance / between-archetype variance ratio
       - recall@1 and recall@5 against archetype-mean centroids
       - argmax-prototype distribution per archetype
  5. Dump results to artifacts/archetype_probe/<tag>/results.json + .npz.

The ground truth is the *generation condition* (which archetype prompt
produced the text), not a hand-tag. Random-baseline recall@1 = 1/33 ≈ 0.030.
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
logger = logging.getLogger("archetype_probe")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--adapter", required=True, help="Path to best_adapter.pt")
    p.add_argument("--generations", default="artifacts/archetype_probe/generations.jsonl")
    p.add_argument("--tag", required=True, help="e.g. v7_step6000")
    p.add_argument("--output-root", default="artifacts/archetype_probe")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    return p.parse_args()


def cosine_sim_matrix(x: np.ndarray) -> np.ndarray:
    n = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    return n @ n.T


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load generations
    rows: list[dict] = []
    with Path(args.generations).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    logger.info("Loaded %d generated sentences", len(rows))
    archetype_ids = sorted({r["archetype_id"] for r in rows})
    n_arch = len(archetype_ids)
    id2name = {r["archetype_id"]: r["archetype_name"] for r in rows}
    logger.info("Archetypes covered: %d (expected 33)", n_arch)

    # Load backbone + adapter
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
    model.backbone.eval()

    # Run forward pass batched
    weights_all: list[np.ndarray] = []  # (N, 32) — may stay empty in trajectory mode
    vectors_all: list[np.ndarray] = []  # (N, 64)
    arch_all: list[int] = []
    has_weights = True  # set False if community_output.weights is None (v8a)

    K = config.community.num_prototypes  # 32
    D = config.community.d_community     # 64

    for i in range(0, len(rows), args.batch_size):
        batch = rows[i : i + args.batch_size]
        texts = [r["text"] for r in batch]
        enc = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_seq_len,
        ).to(device)
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        if out.community_output is None:
            raise RuntimeError("Adapter returned no community_output; check checkpoint.")
        if out.community_output.weights is None:
            has_weights = False
        else:
            w = out.community_output.weights.detach().float().cpu().numpy()  # (B, 32)
            weights_all.append(w)
        v = out.community_output.vector.detach().float().cpu().numpy()   # (B, 64)
        vectors_all.append(v)
        arch_all.extend(r["archetype_id"] for r in batch)
        if (i // args.batch_size) % 10 == 0:
            logger.info("processed %d / %d", i + len(batch), len(rows))

    W = np.concatenate(weights_all, axis=0) if has_weights else None
    V = np.concatenate(vectors_all, axis=0)  # (N, 64)
    A = np.asarray(arch_all, dtype=np.int64)
    N = len(A)
    if has_weights:
        logger.info("Collected: weights %s, vectors %s", W.shape, V.shape)
    else:
        logger.info("Collected: vectors %s (trajectory mode — no prototype weights)", V.shape)

    # Group means
    arch_index = {a: i for i, a in enumerate(archetype_ids)}
    mean_W = np.zeros((n_arch, K), dtype=np.float64) if has_weights else None
    mean_V = np.zeros((n_arch, D), dtype=np.float64)
    counts = np.zeros(n_arch, dtype=np.int64)
    within_var_W = np.zeros(n_arch, dtype=np.float64) if has_weights else None
    within_var_V = np.zeros(n_arch, dtype=np.float64)

    # Pass 1: means
    for a in archetype_ids:
        ai = arch_index[a]
        mask = A == a
        counts[ai] = int(mask.sum())
        if counts[ai] == 0:
            continue
        if has_weights:
            mean_W[ai] = W[mask].mean(axis=0)
        mean_V[ai] = V[mask].mean(axis=0)

    # Pass 2: within-archetype variance (mean squared distance to own centroid)
    for a in archetype_ids:
        ai = arch_index[a]
        mask = A == a
        if counts[ai] == 0:
            continue
        if has_weights:
            diff_W = W[mask] - mean_W[ai]
            within_var_W[ai] = (diff_W ** 2).sum(axis=-1).mean()
        diff_V = V[mask] - mean_V[ai]
        within_var_V[ai] = (diff_V ** 2).sum(axis=-1).mean()

    # Between-archetype variance (mean squared distance between centroid pairs)
    grand_V = mean_V.mean(axis=0)
    between_var_V = ((mean_V - grand_V) ** 2).sum(axis=-1).mean()
    sep_ratio_V = float(between_var_V / (within_var_V.mean() + 1e-12))
    if has_weights:
        grand_W = mean_W.mean(axis=0)
        between_var_W = ((mean_W - grand_W) ** 2).sum(axis=-1).mean()
        sep_ratio_W = float(between_var_W / (within_var_W.mean() + 1e-12))
        cos_W = cosine_sim_matrix(mean_W)
        off_diag_W = float(cos_W[~np.eye(n_arch, dtype=bool)].mean())
    else:
        sep_ratio_W = None
        cos_W = None
        off_diag_W = None

    # Pairwise cosine separation in 64-D vector space
    cos_V = cosine_sim_matrix(mean_V)
    off_diag_V = cos_V[~np.eye(n_arch, dtype=bool)].mean()

    # recall@k against archetype-mean centroids in the 64D space (cosine)
    V_norm = V / (np.linalg.norm(V, axis=-1, keepdims=True) + 1e-8)
    M_norm = mean_V / (np.linalg.norm(mean_V, axis=-1, keepdims=True) + 1e-8)
    sims = V_norm @ M_norm.T  # (N, 33)
    pred_rank = np.argsort(-sims, axis=-1)  # (N, 33)
    true_idx = np.array([arch_index[a] for a in A])  # (N,)
    rank_of_true = (pred_rank == true_idx[:, None]).argmax(axis=-1)
    recall_at_1 = float((rank_of_true == 0).mean())
    recall_at_5 = float((rank_of_true < 5).mean())
    recall_at_10 = float((rank_of_true < 10).mean())

    # Top prototype per archetype (which of the 32 fires most under each).
    # Only available in prototype mode.
    if has_weights:
        top_proto = mean_W.argmax(axis=-1)  # (33,)
        proto_archetypes: dict[int, list[str]] = defaultdict(list)
        for ai, a in enumerate(archetype_ids):
            proto_archetypes[int(top_proto[ai])].append(id2name[a])
        n_unique_top_proto = len(set(top_proto.tolist()))
    else:
        top_proto = None
        proto_archetypes = None
        n_unique_top_proto = None

    results = {
        "tag": args.tag,
        "mode": "prototype" if has_weights else "trajectory",
        "n_generations": int(N),
        "n_archetypes": int(n_arch),
        "K_prototypes": K if has_weights else None,
        "D_community": D,
        "random_baseline_recall_at_1": round(1.0 / n_arch, 4),
        "recall_at_1": round(recall_at_1, 4),
        "recall_at_5": round(recall_at_5, 4),
        "recall_at_10": round(recall_at_10, 4),
        "separation_ratio_weights_32": (
            round(sep_ratio_W, 4) if sep_ratio_W is not None else None
        ),
        "separation_ratio_vectors_64": round(sep_ratio_V, 4),
        "mean_offdiag_cosine_weights": (
            round(off_diag_W, 4) if off_diag_W is not None else None
        ),
        "mean_offdiag_cosine_vectors": round(float(off_diag_V), 4),
        "n_unique_top_prototypes": n_unique_top_proto,
        "top_prototype_per_archetype": (
            {id2name[a]: int(top_proto[arch_index[a]]) for a in archetype_ids}
            if top_proto is not None else None
        ),
        "archetypes_per_top_prototype": (
            {int(p): names for p, names in sorted(proto_archetypes.items())}
            if proto_archetypes is not None else None
        ),
    }

    out_dir = Path(args.output_root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    npz_kwargs = {
        "V": V, "A": A, "mean_V": mean_V, "cos_V": cos_V,
        "archetype_ids": np.asarray(archetype_ids),
    }
    if has_weights:
        npz_kwargs.update({"W": W, "mean_W": mean_W, "cos_W": cos_W})
    np.savez(out_dir / "vectors.npz", **npz_kwargs)
    logger.info("Wrote %s", out_dir / "results.json")
    logger.info(
        "recall@1=%.3f (baseline %.3f)  recall@5=%.3f  sep_ratio(64D)=%.3f%s",
        recall_at_1, 1.0 / n_arch, recall_at_5, sep_ratio_V,
        f"  unique_top_proto={n_unique_top_proto}/{K}" if has_weights else "  [trajectory mode]",
    )


if __name__ == "__main__":
    main()
