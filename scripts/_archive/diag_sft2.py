"""Inline replacement for diagnose_sft.py (which had a numerical bug).

Uses the proven curate_bon math: F.normalize → MSE → 1 - mse/2.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from srt.nla import ActivationReconstructor, ActivationVerbalizer, NLAConfig


def _first_eos_mask(ids: torch.Tensor, eos: int) -> torch.Tensor:
    B, T = ids.shape
    is_eos = (ids == eos)
    pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
    fe = torch.where(
        is_eos.any(-1, keepdim=True),
        is_eos.float().argmax(-1, keepdim=True).long(),
        torch.full((B, 1), T, device=ids.device, dtype=torch.long),
    )
    return (pos <= fe).long()


@torch.no_grad()
def fve_of(av, ar, eos, v, do_sample: bool, k: int, max_new: int) -> float:
    B0 = 8 if not do_sample else 2  # K=32 → 64 effective
    fves = []
    for i in range(0, v.size(0), B0):
        vb = v[i : i + B0]
        if k > 1:
            vb = vb.repeat_interleave(k, dim=0)
        ids = av.generate(
            vb,
            max_new_tokens=max_new,
            do_sample=do_sample,
            temperature=1.2 if do_sample else 1.0,
            top_p=0.95 if do_sample else 1.0,
        )
        attn = _first_eos_mask(ids, eos)
        vhat = ar.reconstruct(ids, attention_mask=attn).float()
        vt = vb.float()
        if torch.isnan(vhat).any() or torch.isinf(vhat).any():
            print(f"  !!! NaN/Inf vhat batch {i}")
            vhat = torch.nan_to_num(vhat, nan=0.0, posinf=0.0, neginf=0.0)
        mse = ((F.normalize(vhat, dim=-1) - F.normalize(vt, dim=-1)) ** 2).sum(-1)
        f = 1.0 - mse / 2.0
        if k > 1:
            f = f.view(-1, k).max(dim=1).values
        fves.append(f.cpu())
    return float(torch.cat(fves).mean().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, type=Path)
    ap.add_argument("--warm", required=True, type=Path)
    ap.add_argument("--sft", required=True, type=Path)
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = NLAConfig(
        backbone_id=args.backbone, backbone_dtype="bfloat16",
        extraction_layer=args.layer, max_new_tokens=args.max_new,
    )
    bb = AutoModelForCausalLM.from_pretrained(args.backbone, torch_dtype=torch.bfloat16)
    for p in bb.parameters():
        p.requires_grad = False
    bb.to(dev).eval()
    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    av = ActivationVerbalizer(cfg, backbone=bb, tokenizer=tok).to(dev)
    ar = ActivationReconstructor(cfg, backbone=bb).to(dev)
    eos = tok.eos_token_id

    obj = torch.load(args.targets, map_location="cpu", weights_only=False)
    pool = torch.stack([a[-1] for a in obj["activations"]]).to(dev)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(10000, generator=g).tolist()[: args.n]
    V = pool[perm]
    print(f"val N={V.size(0)} d={V.size(1)} norm_mean={V.float().norm(dim=-1).mean():.2f}")

    def load(path):
        sd = torch.load(path, map_location=dev, weights_only=False)
        sd = sd["trainable"] if isinstance(sd, dict) and "trainable" in sd else sd
        av.load_state_dict(sd, strict=False)
        av.eval()
        print(f"  loaded {path}  proj.norm={av.proj.weight.float().norm().item():.3f}  "
              f"prefix.norm={av.prefix_embeds.float().norm().item():.3f}")

    print("\n=== WARM ==="); load(args.warm)
    t = time.time(); fg = fve_of(av, ar, eos, V, False, 1, args.max_new)
    print(f"  warm.greedy   = {fg:.4f}  ({time.time()-t:.0f}s)")
    t = time.time(); fb = fve_of(av, ar, eos, V, True, args.k, args.max_new)
    print(f"  warm.bon_K{args.k}  = {fb:.4f}  ({time.time()-t:.0f}s)")

    print("\n=== SFT ==="); load(args.sft)
    t = time.time(); sg = fve_of(av, ar, eos, V, False, 1, args.max_new)
    print(f"  sft.greedy    = {sg:.4f}  ({time.time()-t:.0f}s)")
    t = time.time(); sb = fve_of(av, ar, eos, V, True, args.k, args.max_new)
    print(f"  sft.bon_K{args.k}   = {sb:.4f}  ({time.time()-t:.0f}s)")

    print(f"\n=== DELTAS ===")
    print(f"  greedy:    sft - warm = {sg-fg:+.4f}")
    print(f"  K={args.k}:    sft - warm = {sb-fb:+.4f}")
    print(f"  decoding gap (sft):   K - greedy = {sb-sg:+.4f}")


if __name__ == "__main__":
    main()
