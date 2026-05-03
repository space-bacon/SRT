#!/usr/bin/env python3
"""Encode sentences with a released SRT-Adapter and print pairwise cosine sims.

Uses `community_output.encoded` (mean-pooled, L2-normalized) as the sentence
embedding.

Usage:
    python encode_sentences.py --sentences \
        "A man is playing guitar." \
        "Someone is performing music." \
        "The cat sat on the mat."

First run downloads Qwen/Qwen2.5-7B (~15 GB).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from srt.adapter import SRTAdapter
from srt.config import (
    BENConfig,
    CommunityConfig,
    LossConfig,
    MAHConfig,
    RRMConfig,
    SRTConfig,
)


def build_config(config_path: Path) -> SRTConfig:
    raw = json.loads(config_path.read_text())
    return SRTConfig(
        backbone_id=raw["backbone_id"],
        backbone_dtype=raw["backbone_dtype"],
        mah_layer_indices=list(raw["mah_layer_indices"]),
        rrm_inject_indices=list(raw["rrm_inject_indices"]),
        community_layer_idx=raw["community_layer_idx"],
        num_mah_layers=raw["num_mah_layers"],
        mah=MAHConfig(**raw["mah"]),
        rrm=RRMConfig(**raw["rrm"]),
        ben=BENConfig(**raw["ben"]),
        community=CommunityConfig(**raw["community"]),
        loss=LossConfig(
            **{k: v for k, v in raw["loss"].items() if k in LossConfig.__dataclass_fields__}
        ),
    )


@torch.no_grad()
def encode(model, tok, sentences, device, max_seq_len=128):
    enc = tok(sentences, padding=True, truncation=True,
              max_length=max_seq_len, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    out = model(input_ids=input_ids, attention_mask=attn)
    encoded = out.community_output.encoded  # (B, T, d_community) or (B, d_community)
    if encoded.dim() == 3:
        mask = attn.unsqueeze(-1).to(encoded.dtype)
        encoded = (encoded * mask).sum(1) / mask.sum(1).clamp_min(1.0)
    return F.normalize(encoded, p=2, dim=-1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="RiverRider/srt-adapter-v1.0")
    p.add_argument("--sentences", nargs="+", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--max-seq-len", type=int, default=128)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = Path(hf_hub_download(args.repo, "config.json"))
    config = build_config(cfg_path)
    weights = hf_hub_download(args.repo, "adapter.safetensors")
    print(f"Loading {config.backbone_id} (frozen) and adapter from {args.repo} ...", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(config.backbone_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = SRTAdapter(config).to(device)
    from safetensors.torch import load_file
    model.load_state_dict(load_file(weights, device=device), strict=False)
    model.eval()

    embs = encode(model, tok, args.sentences, device, args.max_seq_len)
    sims = embs @ embs.T

    print("\n=== Sentences ===")
    for i, s in enumerate(args.sentences):
        print(f"  [{i}] {s}")

    print("\n=== Cosine similarity matrix ===")
    n = len(args.sentences)
    print("      " + " ".join(f"  [{j}] " for j in range(n)))
    for i in range(n):
        row = " ".join(f"{sims[i, j].item():+.4f}" for j in range(n))
        print(f"  [{i}] {row}")


if __name__ == "__main__":
    main()
