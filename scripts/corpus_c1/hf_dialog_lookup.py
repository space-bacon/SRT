"""Harvest dialog/reply-bearing datasets from HuggingFace into C1 jsonl format.

Targets datasets where text exhibits *replies* — multi-turn conversation,
question→answer threading, post→comment chains. These supply the
discourse/speech-act/meaning-negotiation signal called for in the C1 spec.

Datasets supported:
    wildchat       allenai/WildChat-1M             CC-BY-4.0 (open)
    oasst2         OpenAssistant/oasst2            Apache-2.0 (open)
    hh-rlhf        Anthropic/hh-rlhf               MIT (open)
    lmsys-chat     lmsys/lmsys-chat-1m             gated; needs HF_TOKEN
    sharegpt       anon8231489123/ShareGPT_Vicuna_unfiltered  CC-BY (community)

Usage:
    python scripts/corpus_c1/hf_dialog_lookup.py --dataset wildchat --max-conversations 50000

Streams the dataset (no full download), formats each conversation as
ROLE-tagged text, then runs the same chunking pipeline as build.py and
writes data/corpus_c1/jsonl/<dataset>.jsonl directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import unicodedata
from pathlib import Path

# Reuse chunker from build.py
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build import (  # noqa: E402
    CHUNK_MAX,
    CHUNK_MIN,
    CHUNK_TARGET,
    JSONL_DIR,
    chunk_text,
    normalise,
    sha1_hex,
    token_count,
)

DATASETS = {
    "wildchat": {
        "hf_id": "allenai/WildChat-1M",
        "split": "train",
        "license": "CC-BY-4.0",
        "school": "dialog-wildchat",
        "weight": 1.1,
        "tier": 2,
        "venue": "WildChat-1M",
        "url": "https://huggingface.co/datasets/allenai/WildChat-1M",
        "field": "conversation",  # list[{role, content}]
    },
    "oasst2": {
        "hf_id": "OpenAssistant/oasst2",
        "split": "train",
        "license": "Apache-2.0",
        "school": "dialog-oasst",
        "weight": 1.2,
        "tier": 2,
        "venue": "OpenAssistant oasst2",
        "url": "https://huggingface.co/datasets/OpenAssistant/oasst2",
        "field": "_oasst_tree",  # special: rebuild trees from messages
    },
    "hh-rlhf": {
        "hf_id": "Anthropic/hh-rlhf",
        "split": "train",
        "license": "MIT",
        "school": "dialog-hh",
        "weight": 1.0,
        "tier": 2,
        "venue": "Anthropic HH-RLHF",
        "url": "https://huggingface.co/datasets/Anthropic/hh-rlhf",
        "field": "_hh_text",  # special: chosen field is "Human:/Assistant:" string
    },
    "lmsys-chat": {
        "hf_id": "lmsys/lmsys-chat-1m",
        "split": "train",
        "license": "LMSYS-Chat-1M-Dataset-License (research)",
        "school": "dialog-lmsys",
        "weight": 1.1,
        "tier": 2,
        "venue": "LMSYS-Chat-1M",
        "url": "https://huggingface.co/datasets/lmsys/lmsys-chat-1m",
        "field": "conversation",
    },
    "sharegpt": {
        "hf_id": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "split": "train",
        "license": "ShareGPT-community (CC-BY-attribution)",
        "school": "dialog-sharegpt",
        "weight": 1.0,
        "tier": 2,
        "venue": "ShareGPT Vicuna unfiltered",
        "url": "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered",
        "field": "conversations",  # list[{from, value}]
    },
}


def fmt_role(role: str) -> str:
    r = (role or "").lower()
    if r in ("user", "human", "prompter"):
        return "USER"
    if r in ("assistant", "gpt", "bot"):
        return "ASSISTANT"
    if r == "system":
        return "SYSTEM"
    return role.upper() if role else "USER"


def conv_to_text(turns: list[dict], role_key: str, content_key: str) -> str:
    parts: list[str] = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        role = t.get(role_key, "user")
        content = t.get(content_key, "") or ""
        content = content.strip()
        if not content:
            continue
        parts.append(f"{fmt_role(role)}: {content}")
    return "\n\n".join(parts)


def iter_conversations(spec: dict, max_n: int, min_lang: str | None):
    """Yield (conv_id, formatted_text, meta) tuples for a streaming dataset."""
    from datasets import load_dataset

    ds = load_dataset(spec["hf_id"], split=spec["split"], streaming=True)
    field = spec["field"]
    n = 0
    for i, row in enumerate(ds):
        if n >= max_n:
            return
        text = ""
        meta: dict = {}
        if field == "conversation":
            turns = row.get("conversation") or []
            if min_lang and row.get("language") and row["language"] != min_lang:
                continue
            text = conv_to_text(turns, "role", "content")
            meta = {
                "conv_id": row.get("conversation_hash") or row.get("conversation_id") or f"row{i}",
                "lang": row.get("language", ""),
                "model": row.get("model", ""),
                "turn": row.get("turn") or len(turns),
            }
        elif field == "conversations":
            turns = row.get("conversations") or []
            text = conv_to_text(turns, "from", "value")
            meta = {"conv_id": row.get("id") or f"row{i}"}
        elif field == "_hh_text":
            # Anthropic HH-RLHF: rows have 'chosen' and 'rejected' both formatted
            # as "\n\nHuman: ...\n\nAssistant: ...". Use chosen only.
            raw = row.get("chosen") or ""
            text = raw.replace("Human:", "USER:").replace("Assistant:", "ASSISTANT:").strip()
            meta = {"conv_id": f"hh{i}"}
        elif field == "_oasst_tree":
            # OpenAssistant rows are individual messages; reconstruct trees by
            # streaming all rows for the split, then walking root→deepest
            # path. To stay streaming-friendly we instead emit each *thread*
            # as we encounter the leaf. Simpler: just emit each message with
            # role tag joined to its parent's role tag if available.
            # Approach: maintain a small in-memory tree of recently-seen
            # messages keyed by message_id; emit when we see a leaf-ish row
            # (no children expected: all messages emitted, dedup by message_id).
            # For simplicity, emit each row as ROLE: text alone.
            role = row.get("role", "")
            content = (row.get("text") or "").strip()
            if not content:
                continue
            text = f"{fmt_role(role)}: {content}"
            meta = {
                "conv_id": row.get("message_tree_id") or row.get("message_id") or f"row{i}",
                "lang": row.get("lang", ""),
                "msg_id": row.get("message_id", ""),
                "parent_id": row.get("parent_id", ""),
            }
        else:
            raise NotImplementedError(field)

        if not text:
            continue
        n += 1
        yield meta["conv_id"], text, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--max-conversations", type=int, default=50000)
    ap.add_argument(
        "--min-lang",
        default=None,
        help="if set, keep only rows with this language code (e.g. 'English')",
    )
    ap.add_argument(
        "--output-name",
        default=None,
        help="basename for output jsonl (default: dataset key)",
    )
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    name = args.output_name or args.dataset.replace("-", "_")
    out_path = JSONL_DIR / f"{name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_chunks: set[str] = set()
    n_rows = 0
    n_convs = 0
    n_tokens = 0

    print(f"Streaming {spec['hf_id']} → {out_path}", file=sys.stderr)
    if args.dataset == "lmsys-chat":
        if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
            print(
                "WARN: LMSYS-Chat-1M is gated. Set HF_TOKEN env var (huggingface-cli login) "
                "and accept the dataset terms at the dataset page.",
                file=sys.stderr,
            )

    with out_path.open("w", encoding="utf-8") as fh:
        for conv_id, raw_text, meta in iter_conversations(
            spec, args.max_conversations, args.min_lang
        ):
            text = normalise(raw_text)
            if token_count(text) < CHUNK_MIN:
                continue
            doc_sha = sha1_hex(text)
            chunks = chunk_text(text)
            emitted = 0
            for ci, chunk in enumerate(chunks):
                csha = sha1_hex(chunk)
                if csha in seen_chunks:
                    continue
                seen_chunks.add(csha)
                ntok = token_count(chunk)
                row = {
                    "text": chunk,
                    "source_id": f"{name}_{conv_id}",
                    "source_url": spec["url"],
                    "title": f"{spec['venue']} conversation {conv_id}",
                    "venue": spec["venue"],
                    "author": [],
                    "school": spec["school"],
                    "tier": spec["tier"],
                    "year": None,
                    "license": spec["license"],
                    "weight": spec["weight"],
                    "doc_sha1": doc_sha,
                    "chunk_idx": ci,
                    "chunk_sha1": csha,
                    "n_tokens_ws": ntok,
                    "meta": meta,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_tokens += ntok
                emitted += 1
            if emitted:
                n_convs += 1
            n_rows += 1
            if n_rows % 1000 == 0:
                print(
                    f"  read={n_rows} kept_convs={n_convs} chunks={len(seen_chunks)} ~tokens={n_tokens:,}",
                    file=sys.stderr,
                )

    print(
        f"DONE {args.dataset}: read={n_rows} kept_convs={n_convs} "
        f"chunks={len(seen_chunks)} approx_tokens={n_tokens:,} → {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
