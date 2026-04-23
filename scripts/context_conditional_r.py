#!/usr/bin/env python3
"""Context-conditional r̂ probe (proxy for Tier-1 A).

For each contested topic, compare r̂ on a target token in two registers:
  - factual passage  (low expected reflexivity)
  - charged passage  (high expected reflexivity)

Same token, same backbone, only the surrounding context changes. If
SRT-Adapter is doing real work, Δr̂ = r̂_charged - r̂_factual should be
positive and meaningfully larger than zero on average.

Reads probes/inputs.txt (paired lines: factual then charged per topic)
and emits a table + summary stats.

Usage:
    python scripts/context_conditional_r.py \\
        --adapter checkpoints/adapter_v5/best_adapter.pt \\
        --prompts probes/inputs.txt \\
        --out artifacts/context_conditional/v5_step17000.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from srt.adapter import SRTAdapter  # noqa: E402
from srt.config import SRTConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("srt.ctxr")


# Topic -> (target_substring, label).  Substring match is used to find the
# token position in the tokenized passage. Case-insensitive.
TOPICS = [
    ("vaccine",  "vaccine"),
    ("border",   "border"),
    ("trans",    "Trans"),
    ("freedom",  "Freedom"),
    ("climate",  "Climate"),
    ("CRT",      "Critical"),
    ("election", "election"),
    ("gender",   "Gender"),
    ("Israel",   "Israel"),
    ("Fed",      "Federal"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--adapter", required=True, type=Path)
    p.add_argument("--prompts", required=True, type=Path,
                   help="Paired-lines file: factual line then charged line, "
                        "one pair per topic, blank line between pairs.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default=None)
    return p.parse_args()


def get_device(req: str | None) -> torch.device:
    if req:
        return torch.device(req)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_pairs(text: str) -> list[tuple[str, str]]:
    """Return list of (factual, charged) pairs."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) % 2 != 0:
        raise SystemExit(f"Expected even number of non-empty lines, got {len(lines)}")
    return [(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]


def find_target_position(tokens: list[str], target: str) -> int | None:
    """Return the index of the first token whose decoded form contains target."""
    target_lower = target.lower()
    for i, tok in enumerate(tokens):
        cleaned = tok.replace("Ġ", "").replace("▁", "").lower()
        if target_lower.startswith(cleaned) and len(cleaned) >= 3:
            # First subword token of target — return this position
            return i
        if cleaned == target_lower:
            return i
    return None


@torch.no_grad()
def r_hat_at_target(
    model: SRTAdapter,
    tokenizer,
    text: str,
    target: str,
    device: torch.device,
    max_seq_len: int,
) -> dict | None:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False,
                    truncation=True, max_length=max_seq_len)
    input_ids = enc["input_ids"].to(device)
    seq_len = int(input_ids.size(1))
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    pos = find_target_position(tokens, target)
    if pos is None:
        return None

    out = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=None,
    )
    r_hat = out.ben_output.r_hat[0].float().cpu()
    div_norm = (out.divergences[-1].norm(dim=-1)[0].float().cpu()
                if out.divergences else torch.zeros(seq_len))
    community_pred = (
        int(out.community_output.weights[0].argmax())
        if out.community_output is not None else None
    )
    return {
        "tok_at_target": tokens[pos].replace("Ġ", " ").replace("▁", " "),
        "pos": pos,
        "seq_len": seq_len,
        "r_hat_at_target": float(r_hat[pos]),
        "r_hat_mean_seq":  float(r_hat.mean()),
        "r_hat_max_seq":   float(r_hat.max()),
        "div_norm_at_target": float(div_norm[pos]),
        "community_pred": community_pred,
    }


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    config = SRTConfig(backbone_id=args.backbone, backbone_dtype=args.dtype)
    logger.info("Building model on %s ...", device)
    model = SRTAdapter(config).to(device)
    state = torch.load(args.adapter, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("Loaded adapter: missing=%d unexpected=%d",
                len(missing), len(unexpected))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pairs = parse_pairs(args.prompts.read_text(encoding="utf-8"))
    if len(pairs) != len(TOPICS):
        logger.warning("Got %d pairs but %d topics defined; truncating to min",
                       len(pairs), len(TOPICS))
    n = min(len(pairs), len(TOPICS))

    rows: list[dict] = []
    for i in range(n):
        factual_text, charged_text = pairs[i]
        topic, target = TOPICS[i]

        f = r_hat_at_target(model, tokenizer, factual_text, target,
                            device, args.max_seq_len)
        c = r_hat_at_target(model, tokenizer, charged_text, target,
                            device, args.max_seq_len)
        if f is None or c is None:
            logger.warning("[%s] target=%r not found in %s",
                           topic, target, "factual" if f is None else "charged")
            continue

        delta_target = c["r_hat_at_target"] - f["r_hat_at_target"]
        delta_mean   = c["r_hat_mean_seq"]  - f["r_hat_mean_seq"]
        rows.append({
            "topic": topic,
            "target": target,
            "factual": f,
            "charged": c,
            "delta_r_hat_at_target": delta_target,
            "delta_r_hat_mean_seq":  delta_mean,
        })
        logger.info("[%s] tok=%r  factual=%.3f  charged=%.3f  Δ=%+.3f  "
                    "(mean Δ=%+.3f)  community %s→%s",
                    topic, f["tok_at_target"],
                    f["r_hat_at_target"], c["r_hat_at_target"], delta_target,
                    delta_mean, f["community_pred"], c["community_pred"])

    n_pos = sum(1 for r in rows if r["delta_r_hat_at_target"] > 0)
    n_pos_mean = sum(1 for r in rows if r["delta_r_hat_mean_seq"] > 0)
    summary = {
        "n_topics_evaluated": len(rows),
        "mean_delta_r_hat_at_target": (
            sum(r["delta_r_hat_at_target"] for r in rows) / max(len(rows), 1)
        ),
        "mean_delta_r_hat_mean_seq": (
            sum(r["delta_r_hat_mean_seq"] for r in rows) / max(len(rows), 1)
        ),
        "n_positive_delta_at_target": n_pos,
        "n_positive_delta_mean_seq": n_pos_mean,
        "rows": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print(f"CONTEXT-CONDITIONAL r̂ PROBE — {args.adapter.name}")
    print("=" * 64)
    print(f"  topics evaluated                      {len(rows)}")
    print(f"  mean Δ r̂ at target token              {summary['mean_delta_r_hat_at_target']:+.4f}")
    print(f"  mean Δ r̂ over full passage            {summary['mean_delta_r_hat_mean_seq']:+.4f}")
    print(f"  topics with positive Δ at target      {n_pos}/{len(rows)}")
    print(f"  topics with positive Δ over passage   {n_pos_mean}/{len(rows)}")
    print("=" * 64)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
