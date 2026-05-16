"""Backbone self-sampler — generate the corpus-free target population.

Runs unconditional sampling from the frozen backbone, extracts hidden
states at ``cfg.extraction_layer``, and writes a parquet shard with
per-position activation vectors plus the generated text.

Usage:
    python scripts/sample_targets.py \\
        --backbone Qwen/Qwen2.5-7B \\
        --layer 20 \\
        --num-sequences 1000 \\
        --seq-len 256 \\
        --out artifacts/nla/targets_qwen7b_L20.parquet

For the N0 smoke this script also supports ``--num-sequences 16`` to
produce a tiny shard usable by tests/bench_n0.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("sample_targets")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--num-sequences", type=int, default=1000)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--format",
        choices=("parquet", "pt", "jsonl"),
        default="pt",
        help="parquet requires pyarrow; pt is a single torch.save dict; jsonl writes text only",
    )
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModelForCausalLM.from_pretrained(args.backbone, dtype=_dtype(args.dtype))
    model.to(device)
    model.eval()

    if not (0 < args.layer <= model.config.num_hidden_layers):
        raise SystemExit(f"--layer {args.layer} out of range for {args.backbone}")

    bos = tok.bos_token_id or model.config.bos_token_id or 0
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else model.config.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sequences: list[str] = []
    activations: list[torch.Tensor] = []  # each (T_valid, d) on CPU, trimmed to first EOS

    todo = args.num_sequences
    while todo > 0:
        bs = min(args.batch_size, todo)
        input_ids = torch.full((bs, 1), bos, dtype=torch.long, device=device)
        with torch.no_grad():
            out_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.seq_len,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
            # Build attention mask that keeps every token up to and including the
            # first EOS per row. Required because pad_token_id == eos_token_id for
            # Qwen, so naive `ids != pad_id` would drop the legitimate EOS *and*
            # leave true pad positions unmasked when no EOS was emitted.
            B, T = out_ids.shape
            is_eos = out_ids == eos_id
            pos = torch.arange(T, device=out_ids.device).unsqueeze(0).expand(B, T)
            first_eos = torch.where(
                is_eos.any(dim=-1, keepdim=True),
                is_eos.float().argmax(dim=-1, keepdim=True).long(),
                torch.full((B, 1), T, device=out_ids.device, dtype=torch.long),
            )
            attn_mask = (pos <= first_eos).long()
            valid_len = attn_mask.sum(dim=-1)  # (B,)
            fwd = model(
                input_ids=out_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            h = fwd.hidden_states[args.layer].detach().to(torch.float32).cpu()  # (B, T, d)

        for i in range(bs):
            n = int(valid_len[i].item())
            sequences.append(tok.decode(out_ids[i, :n], skip_special_tokens=True))
            activations.append(h[i, :n].clone())
        todo -= bs
        logger.info("sampled %d / %d", args.num_sequences - todo, args.num_sequences)

    if args.format == "jsonl":
        with args.out.open("w") as f:
            for s in sequences:
                f.write(json.dumps({"text": s}) + "\n")
    elif args.format == "pt":
        torch.save(
            {
                "sequences": sequences,
                "activations": activations,  # list of (T_i, d) float32 cpu tensors
                "meta": {
                    "backbone_id": args.backbone,
                    "extraction_layer": args.layer,
                    "d": model.config.hidden_size,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.seed,
                },
            },
            args.out,
        )
    elif args.format == "parquet":
        try:
            import pyarrow as pa  # type: ignore[import-not-found]
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError as e:
            raise SystemExit("--format parquet requires pyarrow") from e
        table = pa.table(
            {
                "text": sequences,
                "activations": [a.numpy().tolist() for a in activations],
                "layer": [args.layer] * len(sequences),
            }
        )
        pq.write_table(table, args.out)

    logger.info("wrote %s (%d sequences)", args.out, len(sequences))


if __name__ == "__main__":
    main()
