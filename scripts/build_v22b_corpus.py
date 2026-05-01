#!/usr/bin/env python3
"""Build v22b corpus = average of multi-teacher cosines on the v20 pool.

v21a used a single teacher (mxbai-embed-large-v1). Single-teacher
distillation inherits that teacher's idiosyncrasies. v22b averages
two strong but architecturally-different teachers to denoise:

    teacher_1 = mixedbread-ai/mxbai-embed-large-v1            (English-strong)
    teacher_2 = intfloat/multilingual-e5-large-instruct        (multilingual-strong)

Each pair gets score = mean(cos_t1, cos_t2), clipped to [0, 1].
Output schema is the same as v21a corpus.

Usage:
    python scripts/build_v22b_corpus.py \\
        --in-train data/supervised_sts_v20/train.jsonl \\
        --in-dev   data/supervised_sts_v20/dev.jsonl \\
        --output-dir data/supervised_sts_v22b \\
        --teachers mixedbread-ai/mxbai-embed-large-v1 intfloat/multilingual-e5-large-instruct \\
        --batch-size 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _encode_pairs(rows: list[dict], teacher: SentenceTransformer, batch_size: int) -> torch.Tensor:
    s1 = [r["sentence1"] for r in rows]
    s2 = [r["sentence2"] for r in rows]
    e1 = teacher.encode(s1, batch_size=batch_size, convert_to_tensor=True,
                        show_progress_bar=True, normalize_embeddings=True)
    e2 = teacher.encode(s2, batch_size=batch_size, convert_to_tensor=True,
                        show_progress_bar=True, normalize_embeddings=True)
    return (e1 * e2).sum(dim=-1).float().cpu()


def _hist(scores: list[float], n_bins: int = 10) -> None:
    bins = [0] * n_bins
    for s in scores:
        b = min(int(s * n_bins), n_bins - 1)
        bins[b] += 1
    n = max(1, sum(bins))
    for i, c in enumerate(bins):
        center = (i + 0.5) / n_bins
        bar = "#" * int(50 * c / n)
        print(f"  {center:.2f} | {c:>7} | {bar}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-train", required=True)
    p.add_argument("--in-dev", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--teachers", nargs="+", required=True,
                   help="Two or more sentence-transformers model ids")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--max-seq-length", type=int, default=256)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    for split, in_path in [("train", args.in_train), ("dev", args.in_dev)]:
        rows = _read_jsonl(Path(in_path))
        print(f"[v22b] {split}: {len(rows)} pairs from {in_path}")
        accum = torch.zeros(len(rows), dtype=torch.float32)
        with torch.inference_mode():
            for i, name in enumerate(args.teachers):
                print(f"[v22b] {split}: scoring with teacher {i + 1}/{len(args.teachers)} = {name}")
                t = SentenceTransformer(name, device=args.device,
                                        model_kwargs={"torch_dtype": torch_dtype})
                t.max_seq_length = args.max_seq_length
                t.eval()
                accum += _encode_pairs(rows, t, args.batch_size)
                del t
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

        avg = (accum / len(args.teachers)).clamp(min=0.0, max=1.0).tolist()
        out = []
        for r, sc in zip(rows, avg):
            out.append({
                "sentence1": r["sentence1"],
                "sentence2": r["sentence2"],
                "score": float(sc),
                "source": f"{r.get('source', 'unk').rstrip('-teacher')}-multi",
            })
        out_path = out_dir / f"{split}.jsonl"
        _write_jsonl(out_path, out)
        print(f"[v22b] {split}: wrote {out_path} ({len(out)} rows). score histogram:")
        _hist(avg)


if __name__ == "__main__":
    main()
