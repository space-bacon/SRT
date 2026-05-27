"""Evaluate the NLA target representation as a sentence encoder on MTEB STS.

The NLA target is: forward a tokenized sentence through the frozen backbone,
take ``hidden_states[layer]`` at the *last valid token*. This is the same
representation the AV was trained to verbalize.

We evaluate on the public MTEB STS tasks via direct HF dataset loads (the
``mteb`` package is currently broken against transformers >= 5.0). The
reported metric is Spearman(cosine(h_a, h_b), gold) * 100, matching the
MTEB leaderboard convention.

Usage::

    python scripts/evals/mteb_sts_nla_target.py \\
        --backbone Qwen/Qwen2.5-7B --layer 20 \\
        --tasks biosses stsb sickr sts17 \\
        --batch-size 8 --max-len 128

Memory note: Qwen2.5-7B in bf16 needs ~14 GB. On M2 Ultra (MPS) this fits
in 64 GB unified memory comfortably.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr, pearsonr
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger("mteb_sts_nla_target")


# ---------------------------------------------------------------------------
# Task registry: (hf_repo, hf_config, split, s1_col, s2_col, score_col)
# These mirror MTEB v1 task definitions and use the mirrored HF datasets
# maintained by the MTEB team.
# ---------------------------------------------------------------------------
TASKS = {
    "biosses": ("mteb/biosses-sts", None, "test", "sentence1", "sentence2", "score"),
    "stsb": ("mteb/stsbenchmark-sts", None, "test", "sentence1", "sentence2", "score"),
    "sickr": ("mteb/sickr-sts", None, "test", "sentence1", "sentence2", "score"),
    "sts12": ("mteb/sts12-sts", None, "test", "sentence1", "sentence2", "score"),
    "sts13": ("mteb/sts13-sts", None, "test", "sentence1", "sentence2", "score"),
    "sts14": ("mteb/sts14-sts", None, "test", "sentence1", "sentence2", "score"),
    "sts15": ("mteb/sts15-sts", None, "test", "sentence1", "sentence2", "score"),
    "sts16": ("mteb/sts16-sts", None, "test", "sentence1", "sentence2", "score"),
    "sts17": ("mteb/sts17-crosslingual-sts", "en-en", "test", "sentence1", "sentence2", "score"),
    "sts22": ("mteb/sts22-crosslingual-sts", "en", "test", "sentence1", "sentence2", "score"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--layer", type=int, nargs="+", default=[20],
                   help="one or more layer indices; model is loaded once and re-used")
    p.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    p.add_argument("--tasks", nargs="+", default=["biosses", "stsb", "sickr", "sts17"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detect)")
    p.add_argument("--out", type=Path, default=Path("artifacts/mteb_sts_nla_target.json"))
    p.add_argument(
        "--pool",
        nargs="+",
        default=["last"],
        choices=("last", "mean"),
        help="one or more pools; last = NLA-canonical last-valid-token, mean = mean over valid tokens",
    )
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _pick_device(req: str | None) -> str:
    if req:
        return req
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def encode_multi(
    model,
    tok,
    sentences: list[str],
    *,
    layers: list[int],
    pools: list[str],
    batch_size: int,
    max_len: int,
    device: str,
) -> dict[tuple[int, str], np.ndarray]:
    """Single forward pass per batch; emit (layer, pool) -> (N, d) arrays."""
    out: dict[tuple[int, str], list[np.ndarray]] = {(L, p): [] for L in layers for p in pools}
    n = len(sentences)
    for i in range(0, n, batch_size):
        batch = sentences[i : i + batch_size]
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)
        fwd = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )
        attn = enc["attention_mask"]  # (B, T)
        last_idx = attn.sum(dim=1) - 1  # (B,)
        B = attn.shape[0]
        idx_rng = torch.arange(B, device=device)
        for L in layers:
            h = fwd.hidden_states[L]  # (B, T, d)
            for p in pools:
                if p == "last":
                    vecs = h[idx_rng, last_idx]
                elif p == "mean":
                    mask = attn.unsqueeze(-1).to(h.dtype)
                    vecs = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
                else:
                    raise ValueError(p)
                out[(L, p)].append(vecs.to(torch.float32).cpu().numpy())
        if (i // batch_size) % 10 == 0:
            logger.info("  encode %d / %d", min(i + batch_size, n), n)
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (a * b).sum(axis=1)


def eval_task(name: str, model, tok, args, device: str) -> list[dict]:
    repo, cfg, split, c1, c2, cs = TASKS[name]
    logger.info("[%s] loading %s / %s / %s", name, repo, cfg or "-", split)
    ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
    s1 = [str(x) for x in ds[c1]]
    s2 = [str(x) for x in ds[c2]]
    gold = np.asarray(ds[cs], dtype=np.float64)
    t0 = time.time()
    e1 = encode_multi(model, tok, s1,
                      layers=args.layer, pools=args.pool,
                      batch_size=args.batch_size, max_len=args.max_len, device=device)
    e2 = encode_multi(model, tok, s2,
                      layers=args.layer, pools=args.pool,
                      batch_size=args.batch_size, max_len=args.max_len, device=device)
    dt = time.time() - t0
    rows = []
    for L in args.layer:
        for p in args.pool:
            sims = cosine_rows(e1[(L, p)], e2[(L, p)])
            rho = float(spearmanr(sims, gold).statistic)
            r = float(pearsonr(sims, gold).statistic)
            logger.info("[%s L%d %s] n=%d  rho=%.4f  pearson=%.4f",
                        name, L, p, len(gold), rho, r)
            rows.append({
                "task": name, "layer": L, "pool": p,
                "n": int(len(gold)),
                "spearman": rho, "pearson": r,
                "spearman_x100": rho * 100,
            })
    logger.info("[%s] (%d (L,pool) combos) %.1fs total", name, len(rows), dt)
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()
    device = _pick_device(args.device)
    logger.info("device=%s  backbone=%s  layer=%d  pool=%s  dtype=%s",
                device, args.backbone, args.layer, args.pool, args.dtype)

    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(args.backbone, dtype=_dtype(args.dtype))
    model.to(device)
    model.eval()
    for L in args.layer:
        if L > model.config.num_hidden_layers:
            raise SystemExit(
                f"--layer {L} exceeds num_hidden_layers={model.config.num_hidden_layers}"
            )

    all_rows: list[dict] = []
    for t in args.tasks:
        if t not in TASKS:
            raise SystemExit(f"unknown task {t!r}; known: {sorted(TASKS)}")
        all_rows.extend(eval_task(t, model, tok, args, device))

    # Aggregate per (layer, pool)
    from collections import defaultdict
    agg: dict[tuple[int, str], list[float]] = defaultdict(list)
    for r in all_rows:
        agg[(r["layer"], r["pool"])].append(r["spearman"])
    summary_combos = [
        {"layer": L, "pool": p,
         "mean_spearman": float(np.mean(vals)),
         "mean_spearman_x100": float(np.mean(vals) * 100),
         "n_tasks": len(vals)}
        for (L, p), vals in sorted(agg.items())
    ]
    summary = {
        "backbone": args.backbone,
        "layers": args.layer,
        "pools": args.pool,
        "dtype": args.dtype,
        "max_len": args.max_len,
        "per_task": all_rows,
        "per_combo": summary_combos,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    logger.info("=" * 60)
    for row in summary_combos:
        logger.info("L%-2d %-4s  mean rho_x100 = %.2f  (over %d tasks)",
                    row["layer"], row["pool"], row["mean_spearman_x100"], row["n_tasks"])
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
