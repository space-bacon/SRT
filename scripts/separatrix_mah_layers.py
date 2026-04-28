#!/usr/bin/env python3
"""Per-MAH-layer separatrix readout.

Mirrors scripts/separatrix_eval.py but records the MAH peak-divergence on
the *continuation* slice for **every** MAH layer (v8a: L7, L14, L21), for
every branch (technical, mystical, bedrock). This lets us build a per-layer
depth profile of MAH bedrock-vs-mystic vs T-Head bedrock-vs-tech (see
docs/THEAD_SEPARATRIX_V1_FINDINGS.md, "Depth profile").
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

from srt.adapter import SRTAdapter
from srt.config import SRTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sep_mah_layers")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--adapter", required=True)
    p.add_argument("--probe", default="data/probes/separatrix_illusion_v1.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--mah-window", type=int, default=8)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def sliding_peak(x: torch.Tensor, window: int) -> float:
    if x.numel() == 0:
        return 0.0
    if x.numel() <= window:
        return float(x.mean())
    pooled = x.unfold(0, window, 1).mean(dim=-1)
    return float(pooled.max())


@torch.no_grad()
def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    cfg = SRTConfig(backbone_id=args.backbone, backbone_dtype=args.dtype)
    cfg.community.use_prototypes = False
    model = SRTAdapter(cfg).to(device)
    model.load_adapter(args.adapter)
    model.eval()

    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    items = [json.loads(l) for l in Path(args.probe).read_text().splitlines() if l.strip()]
    log.info("loaded %d items", len(items))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_layers_seen = None
    with out_path.open("w") as fout:
        for item in items:
            prompt = item["prompt"]
            n_prompt = int(tok(prompt, add_special_tokens=False,
                               return_tensors="pt")["input_ids"].shape[1])
            branches = {
                "technical": prompt + item["technical_continuation"],
                "mystical": prompt + item["mystical_continuation"],
                "bedrock": prompt + item["bedrock_continuation"],
            }
            row = {"id": item["id"], "concept": item["concept"],
                   "domain": item["domain"], "n_prompt_tokens": n_prompt,
                   "branches": {}}
            for name, text in branches.items():
                enc = tok(text, return_tensors="pt", truncation=True,
                          max_length=args.max_seq_len, padding=False)
                ids = enc["input_ids"].to(device)
                att = enc["attention_mask"].to(device)
                T = int(att.sum())
                o = model(input_ids=ids, attention_mask=att)
                if not o.divergences:
                    log.error("no MAH divergences in adapter output")
                    return 2
                if n_layers_seen is None:
                    n_layers_seen = len(o.divergences)
                    log.info("MAH layers in adapter: %d", n_layers_seen)
                per_layer = {}
                for li, div in enumerate(o.divergences):
                    norms = div.norm(dim=-1)[0].float().cpu()  # (T,)
                    full = sliding_peak(norms, args.mah_window)
                    cont = sliding_peak(norms[n_prompt:], args.mah_window) if n_prompt < T else full
                    per_layer[f"L{li}"] = {
                        "mah_peak_full": full,
                        "mah_peak_continuation": cont,
                        "mah_mean_continuation": float(norms[n_prompt:].mean()) if n_prompt < T else float(norms.mean()),
                    }
                row["branches"][name] = {"n_tokens": T, "mah_layers": per_layer}
            fout.write(json.dumps(row) + "\n"); fout.flush()
            n_written += 1
    log.info("done: %d rows -> %s", n_written, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
