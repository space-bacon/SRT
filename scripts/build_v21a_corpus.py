#!/usr/bin/env python3
"""Build v21a corpus by RE-SCORING the v20 pair pool with a teacher.

v20 lesson: NLI categorical labels {entail=1.0, neutral=0.5, contra=0.0}
are NOT similarity labels. They warp graded English STS.

v21a fix: keep the same diverse pair pool (STSB + ~200K NLI premise/
hypothesis pairs) but replace every label with the teacher embedding's
cosine similarity. Teacher cosines are CONTINUOUS values aligned with
how STS itself is evaluated, so CoSENT trains on graded similarity
rather than NLI's categorical notion.

Default teacher: `mixedbread-ai/mxbai-embed-large-v1` (335M, MTEB-STS top tier).
Override with --teacher to swap in `intfloat/multilingual-e5-large-instruct`
or any other sentence-transformers compatible model.

Output schema (unchanged):
    {"sentence1": str, "sentence2": str, "score": float in [0,1], "source": str}

Score = max(0, cos(teacher(s1), teacher(s2))). Negative cosines are
clipped to 0 because labels feed into a [0, 1] regression-style space.
The CoSENT loss only cares about pairwise ordering, so the clipping
just compresses the lowest scores; ranking is preserved.

Usage:
    python scripts/build_v21a_corpus.py \\
        --in-train data/supervised_sts_v20/train.jsonl \\
        --in-dev   data/supervised_sts_v20/dev.jsonl \\
        --output-dir data/supervised_sts_v21a \\
        --teacher mixedbread-ai/mxbai-embed-large-v1 \\
        --batch-size 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _rescore(rows: list[dict], teacher: SentenceTransformer, batch_size: int) -> list[dict]:
    s1_all = [r["sentence1"] for r in rows]
    s2_all = [r["sentence2"] for r in rows]

    e1 = teacher.encode(
        s1_all,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    e2 = teacher.encode(
        s2_all,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Cosine = dot for normalized vectors. Clamp to [0, 1].
    cos = (e1 * e2).sum(dim=-1).clamp(min=0.0, max=1.0).cpu().tolist()

    out: list[dict] = []
    for r, sc in zip(rows, cos):
        out.append(
            {
                "sentence1": r["sentence1"],
                "sentence2": r["sentence2"],
                "score": float(sc),
                "source": f"{r.get('source', 'unk')}-teacher",
            }
        )
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _hist(rows: list[dict], n_bins: int = 10) -> list[tuple[float, int]]:
    bins = [0] * n_bins
    for r in rows:
        b = min(int(r["score"] * n_bins), n_bins - 1)
        bins[b] += 1
    return [((i + 0.5) / n_bins, c) for i, c in enumerate(bins)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-train", required=True)
    p.add_argument("--in-dev", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--teacher", default="mixedbread-ai/mxbai-embed-large-v1")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v21a] loading teacher: {args.teacher} (device={args.device}, dtype={args.dtype})")
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    teacher = SentenceTransformer(args.teacher, device=args.device, model_kwargs={"torch_dtype": torch_dtype})
    teacher.max_seq_length = 256
    teacher.eval()

    for split, in_path in [("train", args.in_train), ("dev", args.in_dev)]:
        rows = _read_jsonl(Path(in_path))
        print(f"[v21a] {split}: {len(rows)} pairs from {in_path}")
        with torch.inference_mode():
            scored = _rescore(rows, teacher, args.batch_size)
        out_path = out_dir / f"{split}.jsonl"
        _write_jsonl(out_path, scored)
        print(f"[v21a] {split}: wrote {out_path} ({len(scored)} rows)")
        print(f"[v21a] {split} score histogram (10 bins):")
        for center, count in _hist(scored):
            bar = "#" * int(50 * count / max(1, len(scored)))
            print(f"  {center:.2f} | {count:>7} | {bar}")


if __name__ == "__main__":
    main()
