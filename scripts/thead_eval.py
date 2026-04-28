#!/usr/bin/env python3
"""T-Head evaluation — extract Takens delay-embedding readouts on a probe set.

For each prompt in a JSONL battery, run the frozen backbone with
`output_hidden_states=True`, take the hidden-state stream at a chosen layer,
pass it through `srt.modules.thead.TakensHead`, and dump the per-token
Lyapunov estimate and recurrence density along with metadata.

This is a v10 prototype eval. The T-Head is *not* part of the v9 SRTAdapter
forward pass; this script reads the backbone directly so the readouts can be
computed without touching the adapter codepath.

Usage
-----
On a CUDA box (preferred for Qwen-7B):

    python3 scripts/thead_eval.py \\
        --backbone Qwen/Qwen2.5-7B \\
        --battery data/probes/separatrix_illusion_v1.jsonl \\
        --output artifacts/thead/v1/readouts.jsonl \\
        --layer-index 16 \\
        --max-seq-len 128

Local smoke test (small backbone, no GPU needed):

    python3 scripts/thead_eval.py \\
        --backbone sshleifer/tiny-gpt2 \\
        --battery data/probes/separatrix_illusion_v1.jsonl \\
        --output artifacts/thead/smoke/readouts.jsonl \\
        --layer-index -1 \\
        --max-seq-len 64

Output schema (one JSON line per battery item):

    {
      "id": "sep_001",
      "concept": "entanglement",
      "domain": "physics",
      "branch": "technical" | "mystical" | "bedrock" | "prompt_only",
      "n_tokens": 47,
      "layer_index": 16,
      "lyapunov_mean": 0.13,
      "lyapunov_std": 0.41,
      "lyapunov_per_token": [...],
      "recurrence_mean": 1.4,
      "recurrence_max": 5,
      "recurrence_per_token": [...]
    }

For separatrix-style batteries (with technical/mystical/bedrock continuations)
each prompt produces four rows: prompt-only and prompt + each continuation.
For generic batteries (single `text` field) each prompt produces one row.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from srt.modules.thead import TakensHead, THeadConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("thead_eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--battery", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--layer-index", type=int, default=16,
                   help="Hidden-state layer to read. -1 = last. "
                        "Qwen2.5-7B has 28 layers; 16 is mid-late.")
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default=None)
    # T-Head hyperparams (defaults match THeadConfig)
    p.add_argument("--embedding-dim", type=int, default=4)
    p.add_argument("--delay", type=int, default=2)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--recurrence-eps", type=float, default=1.0)
    p.add_argument("--project-to", type=int, default=32)
    p.add_argument("--limit", type=int, default=0,
                   help="Limit number of battery items (0 = all).")
    return p.parse_args()


def get_device(req: str | None) -> torch.device:
    if req:
        return torch.device(req)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def expand_battery_item(item: dict) -> list[tuple[str, str]]:
    """Return list of (branch, text) pairs for one battery item.

    Separatrix items expand to 4 branches; generic items expand to 1.
    """
    if "technical_continuation" in item and "mystical_continuation" in item:
        prompt = item["prompt"]
        return [
            ("prompt_only", prompt),
            ("technical", prompt + item["technical_continuation"]),
            ("mystical", prompt + item["mystical_continuation"]),
            ("bedrock", prompt + item["bedrock_continuation"]),
        ]
    return [("prompt_only", item.get("text", item.get("prompt", "")))]


@torch.no_grad()
def main() -> int:
    args = parse_args()
    device = get_device(args.device)
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    load_dtype = dtype_map[args.dtype]
    log.info("device=%s dtype=%s backbone=%s layer=%d",
             device, args.dtype, args.backbone, args.layer_index)

    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    log.info("loading backbone (frozen, no adapter)")
    model = AutoModelForCausalLM.from_pretrained(
        args.backbone, torch_dtype=load_dtype, output_hidden_states=True,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    d_hidden = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    layer_idx = args.layer_index if args.layer_index >= 0 else n_layers + args.layer_index
    log.info("backbone layers=%d, reading layer %d (d_hidden=%d)",
             n_layers, layer_idx, d_hidden)

    thead_cfg = THeadConfig(
        embedding_dim=args.embedding_dim,
        delay=args.delay,
        knn_k=args.knn_k,
        recurrence_eps=args.recurrence_eps,
        project_to=args.project_to,
    )
    thead = TakensHead(thead_cfg, d_hidden=d_hidden).to(device).to(load_dtype)
    thead.eval()
    log.info("T-Head config=%s params=%d", thead_cfg,
             sum(p.numel() for p in thead.parameters()))

    items = [json.loads(l) for l in Path(args.battery).read_text().splitlines() if l.strip()]
    if args.limit > 0:
        items = items[: args.limit]
    log.info("loaded %d battery items", len(items))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    with out_path.open("w") as fout:
        for item in items:
            for branch, text in expand_battery_item(item):
                if not text:
                    continue
                enc = tok(text, return_tensors="pt", truncation=True,
                          max_length=args.max_seq_len, padding=False)
                ids = enc["input_ids"].to(device)
                att = enc["attention_mask"].to(device)
                T = int(att.sum())
                if T < thead_cfg.embedding_dim * thead_cfg.delay + 1:
                    log.warning("skip too-short sequence id=%s branch=%s T=%d",
                                item.get("id"), branch, T)
                    continue

                out = model(input_ids=ids, attention_mask=att,
                            output_hidden_states=True)
                # hidden_states: tuple of length n_layers+1 (incl. embeddings)
                h = out.hidden_states[layer_idx + 1]  # +1 to skip embed layer
                # truncate to real tokens; T-Head ops are O(T^2) so keep tight
                h = h[:, :T, :]

                t_out = thead(h)
                lyap = t_out.lyapunov[0].float().cpu()
                rec = t_out.recurrence[0].float().cpu()

                row: dict[str, Any] = {
                    "id": item.get("id"),
                    "concept": item.get("concept"),
                    "domain": item.get("domain"),
                    "label": item.get("label"),
                    "branch": branch,
                    "n_tokens": T,
                    "layer_index": layer_idx,
                    "lyapunov_mean": float(lyap.mean()),
                    "lyapunov_std": float(lyap.std()),
                    "lyapunov_per_token": lyap.tolist(),
                    "recurrence_mean": float(rec.mean()),
                    "recurrence_max": float(rec.max()),
                    "recurrence_per_token": rec.tolist(),
                }
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                n_rows += 1

            if (n_rows % 25) == 0:
                log.info("rows written: %d", n_rows)

    log.info("done: %d rows -> %s", n_rows, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
