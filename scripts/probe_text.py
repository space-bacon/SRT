#!/usr/bin/env python3
"""Probe arbitrary text through a trained SRT-Adapter and emit an HTML heatmap.

Unlike instrument_eval.py (which evaluates over a labelled val set), this
script takes raw input text, tokenizes it, runs one forward pass, and
renders per-token r̂ + regime + community-prototype prediction directly
to a self-contained HTML file.

Usage (single string):
    python scripts/probe_text.py \\
        --adapter checkpoints/adapter_v4/best_adapter.pt \\
        --text "the vaccine is safe and effective" \\
        --out artifacts/probes/vaccine.html

Usage (file, one passage per line):
    python scripts/probe_text.py \\
        --adapter checkpoints/adapter_v4/best_adapter.pt \\
        --text-file probes/inputs.txt \\
        --out artifacts/probes/batch.html
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_heatmap import render_html  # noqa: E402

from srt.adapter import SRTAdapter
from srt.config import SRTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("srt.probe")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--adapter", required=True, type=Path)
    p.add_argument("--text", default=None,
                   help="Single passage to probe. Mutually exclusive with --text-file.")
    p.add_argument("--text-file", default=None, type=Path,
                   help="File with one passage per line.")
    p.add_argument("--out", required=True, type=Path,
                   help="Output HTML path.")
    p.add_argument("--title", default=None,
                   help="HTML page title (defaults to adapter basename).")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default=None)
    args = p.parse_args()
    if (args.text is None) == (args.text_file is None):
        p.error("Provide exactly one of --text or --text-file.")
    return args


def get_device(req: str | None) -> torch.device:
    if req:
        return torch.device(req)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_passages(args: argparse.Namespace) -> list[str]:
    if args.text is not None:
        return [args.text]
    lines = [
        line.rstrip("\n")
        for line in args.text_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise SystemExit(f"No non-empty lines in {args.text_file}")
    return lines


@torch.no_grad()
def probe(
    model: SRTAdapter,
    tokenizer,
    passages: list[str],
    device: torch.device,
    max_seq_len: int,
) -> list[dict]:
    """Return a list of trace dicts in the schema render_heatmap expects."""
    model.eval()
    traces: list[dict] = []

    for passage in passages:
        enc = tokenizer(
            passage,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        seq_len = int(attention_mask.sum())

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
        )

        token_ids = input_ids[0, :seq_len].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        r_hat = out.ben_output.r_hat[0, :seq_len].float().cpu().tolist()
        regime = out.ben_output.regime_logits[0, :seq_len].argmax(dim=-1).cpu().tolist()
        community_pred = (
            int(out.community_output.weights[0].argmax())
            if out.community_output is not None
            else None
        )

        traces.append({
            "tokens": tokens,
            "r_hat": r_hat,
            # No ground truth at probe time — fill in compatible placeholders
            "r_true": [0.0] * seq_len,
            "r_mask": [False] * seq_len,
            "regime_pred": regime,
            "community_id_true": None,
            "community_pred": community_pred,
        })

        # Console summary for quick inspection
        rh_arr = torch.tensor(r_hat)
        logger.info(
            "passage (%d tok)  r̂ mean=%.3f  max=%.3f @ '%s'  community_pred=%s",
            seq_len,
            float(rh_arr.mean()),
            float(rh_arr.max()),
            tokens[int(rh_arr.argmax())].replace("Ġ", " ").replace("▁", " "),
            community_pred,
        )

    return traces


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    title = args.title or f"SRT-Adapter probe — {args.adapter.name}"

    config = SRTConfig(backbone_id=args.backbone, backbone_dtype=args.dtype)
    logger.info("Building model on %s ...", device)
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

    passages = load_passages(args)
    logger.info("Probing %d passage(s)...", len(passages))
    traces = probe(model, tokenizer, passages, device, args.max_seq_len)

    html_str = render_html(traces, title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_str, encoding="utf-8")
    print(f"Wrote {args.out}  ({len(traces)} passages, {args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
