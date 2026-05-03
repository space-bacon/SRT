#!/usr/bin/env python3
"""Load a released SRT-Adapter checkpoint from Hugging Face and score a passage.

Usage:
    python load_and_score.py --text "Vaccine mandates are an obvious public health win."
    python load_and_score.py --repo RiverRider/srt-adapter-v8a --text "..."

First run downloads Qwen/Qwen2.5-7B (~15 GB) from HuggingFace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="RiverRider/srt-adapter-v1.0",
                    help="Hugging Face model repo containing config.json + adapter.safetensors")
    ap.add_argument("--text", required=True, help="Passage to score.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-seq-len", type=int, default=512)
    args = ap.parse_args()

    print(f"[load] repo:     {args.repo}")
    cfg_path = Path(hf_hub_download(args.repo, "config.json"))
    config = build_config(cfg_path)

    print(f"[load] backbone: {config.backbone_id} ({config.backbone_dtype})")
    adapter_path = hf_hub_download(args.repo, "adapter.safetensors")
    model = SRTAdapter(config).to(args.device)

    from safetensors.torch import load_file
    state = load_file(adapter_path, device=args.device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    tok = AutoTokenizer.from_pretrained(config.backbone_id)
    enc = tok(args.text, return_tensors="pt", truncation=True,
              max_length=args.max_seq_len).to(args.device)

    with torch.no_grad():
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)

    print("\n=== SRT-Adapter readouts ===")
    print(f"input tokens:                  {enc.input_ids.shape[1]}")
    print(f"backbone vocab logits shape:   {tuple(out.logits.shape)}")

    if out.community_output is not None:
        cv = out.community_output.vector[0]
        print(f"community vector ({cv.shape[0]}-D): "
              f"norm={cv.norm().item():.3f}  "
              f"first 5 dims={[round(x, 3) for x in cv[:5].tolist()]}")

    for i, d in enumerate(out.divergences):
        print(f"divergence layer {i} mean ||d||:  {d.norm(dim=-1).mean().item():.3f}")

    if out.ben_output is not None:
        r_hat = out.ben_output.r_hat[0]
        regime_prob_super = torch.softmax(out.ben_output.regime_logits[0], dim=-1)[:, 1]
        print(f"reflexivity r_hat:             "
              f"mean={r_hat.mean().item():+.3f}  "
              f"min={r_hat.min().item():+.3f}  "
              f"max={r_hat.max().item():+.3f}")
        print(f"P(supercritical):              "
              f"mean={regime_prob_super.mean().item():.3f}  "
              f"max={regime_prob_super.max().item():.3f}")

    print("\nSee paper.pdf §3 for what each readout means and §5 for headline numbers.")


if __name__ == "__main__":
    main()
