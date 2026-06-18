#!/usr/bin/env python3
"""Maximise HF discoverability for all RiverRider SRT model cards.

For every repo it ensures the YAML frontmatter carries the metadata the Hub
indexes for its filter/discovery lists:

- ``base_model`` + ``base_model_relation: adapter`` -> appears under each base
  model's *Adapters* tab and the ``base_model:adapter:<id>`` filter.
- ``pipeline_tag`` -> task filter.
- ``language`` / ``license`` -> language and license filters.
- ``library_name`` -> library filter.
- ``tags`` -> ``?other=<tag>`` filters.

Existing card bodies are preserved. Run after ``hf auth login``.

Usage:
    python scripts/hf_update_cards.py            # apply to all repos
    python scripts/hf_update_cards.py --dry-run  # print planned metadata only
"""
from __future__ import annotations

import argparse
import os

from huggingface_hub import ModelCard, ModelCardData

QWEN25 = "Qwen/Qwen2.5-7B"

# repo_id -> (base_model list, human-readable title)
MODELS: dict[str, tuple[list[str], str]] = {
    "RiverRider/srt-adapter-v8a": ([QWEN25], "SRT Adapter v8a"),
    "RiverRider/srt-adapter-v18": ([QWEN25], "SRT Adapter v18"),
    "RiverRider/srt-adapter-v21a": ([QWEN25], "SRT Adapter v21a"),
    "RiverRider/srt-adapter-v22c_a050": ([QWEN25], "SRT Adapter v22c_a050"),
    "RiverRider/srt-adapter-v1.0": ([QWEN25], "SRT Adapter v1.0"),
    "RiverRider/zooL4nD3r-v0.1": ([QWEN25], "zooL4nD3r v0.1"),
    "RiverRider/srt-nla-av-v1": ([QWEN25], "SRT NLA Activation Verbalizer v1"),
    "RiverRider/srt-nla-av-llama32-3b": (
        ["meta-llama/Llama-3.2-3B"],
        "SRT NLA Activation Verbalizer (Llama-3.2-3B)",
    ),
    "RiverRider/srt-nla-av-gemma2-2b-v1": (
        ["google/gemma-2-2b"],
        "SRT NLA Activation Verbalizer (Gemma-2-2B)",
    ),
    "RiverRider/srt-adapter-qwen3-235b": (
        ["Qwen/Qwen3-235B-A22B", "Qwen/Qwen3-235B-A22B-FP8"],
        "SRT Adapter Qwen3-235B-A22B (Phase-A, read-only)",
    ),
}

COMMON_TAGS = [
    "srt",
    "semiotic-reflexive-transformer",
    "introspection",
    "interpretability",
    "adapter",
    "frozen-backbone",
    "custom_code",
]

DEFAULT_BODY = (
    "# {title}\n\n"
    "A Semiotic-Reflexive Transformer (SRT) side-channel adapter on a frozen "
    "backbone. See <https://github.com/space-bacon/SRT> for code and usage.\n"
)


def build_card(repo_id: str, bases: list[str], title: str) -> ModelCard:
    try:
        card = ModelCard.load(repo_id)
    except Exception:
        card = ModelCard(DEFAULT_BODY.format(title=title))

    if not (card.content or "").strip() or card.content.strip() == "---":
        card = ModelCard(DEFAULT_BODY.format(title=title))

    data: ModelCardData = card.data or ModelCardData()

    data.base_model = bases
    data.base_model_relation = "adapter"
    data.pipeline_tag = "feature-extraction"
    data.library_name = "srt-adapter"
    data.license = "apache-2.0"
    data.language = ["en"]

    existing = list(data.tags or [])
    merged: list[str] = []
    for t in existing + COMMON_TAGS:
        if t and not t.startswith("base_model") and t not in merged:
            merged.append(t)
    data.tags = merged

    card.data = data
    return card


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", help="restrict to these repo ids")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    repos = args.only or list(MODELS)

    for repo_id in repos:
        bases, title = MODELS[repo_id]
        card = build_card(repo_id, bases, title)
        print(f"\n=== {repo_id} ===")
        print(card.data.to_yaml())
        if args.dry_run:
            continue
        card.push_to_hub(
            repo_id,
            token=token,
            commit_message="Add discovery metadata (base_model adapter relation, tags, pipeline)",
        )
        print(f"pushed: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
