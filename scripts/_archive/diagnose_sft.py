"""Diagnose SFT iter-1 plateau.

Measures, on the same 256 held-out targets:
  1. warm-start AV  + greedy        → baseline
  2. warm-start AV  + temp=1.2 K=32 → BoN ceiling (matches curation)
  3. SFT'd  AV      + greedy        → deterministic policy (val number)
  4. SFT'd  AV      + temp=1.2 K=32 → policy's BoN ceiling

If (4) >> (3) the policy *knows* the high-fve manifold but greedy
collapses to a low-mode → fix is decoding (temp/topp) or KD with a
distribution target. If (4) ≈ (3) the policy never learned → fix is
training (lr, schedule, more epochs).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from srt.nla import ActivationReconstructor, ActivationVerbalizer, NLAConfig
from srt.nla.targets_check import assert_targets_healthy


def _load_targets(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return torch.stack([a[-1] for a in obj["activations"]])


def _first_eos_mask(ids: torch.Tensor, eos: int) -> torch.Tensor:
    B, T = ids.shape
    is_eos = ids == eos
    pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
    first = torch.where(
        is_eos.any(-1, keepdim=True),
        is_eos.float().argmax(-1, keepdim=True).long(),
        torch.full((B, 1), T, device=ids.device, dtype=torch.long),
    )
    return (pos <= first).long()


@torch.no_grad()
def measure(av, ar, eos, v, batch, max_new, do_sample, temperature, top_p, k):
    """Return mean fve_nrm. For k>1 returns mean of per-target best."""
    fve = []
    for i in range(0, v.size(0), batch):
        vb = v[i : i + batch]
        if k > 1:
            vb_rep = vb.repeat_interleave(k, dim=0)
        else:
            vb_rep = vb
        ids = av.generate(
            vb_rep, max_new_tokens=max_new,
            do_sample=do_sample, temperature=temperature, top_p=top_p,
        )
        attn = _first_eos_mask(ids, eos)
        vhat = ar.reconstruct(ids, attention_mask=attn).float()
        vt = vb_rep.float()
        mse = ((F.normalize(vhat, -1) - F.normalize(vt, -1)) ** 2).sum(-1)
        f = 1.0 - mse / 2.0
        if k > 1:
            f = f.view(-1, k).max(dim=1).values
        fve.append(f.cpu())
    return torch.cat(fve).mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, type=Path)
    ap.add_argument("--warm", required=True, type=Path)
    ap.add_argument("--sft", required=True, type=Path)
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n", type=int, default=256, help="val targets (same 256 across all 4)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temp", type=float, default=1.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    cfg = NLAConfig(backbone_id=args.backbone, backbone_dtype=args.dtype,
                    extraction_layer=args.layer, max_new_tokens=args.max_new)

    backbone = AutoModelForCausalLM.from_pretrained(args.backbone, torch_dtype=dtype)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    av = ActivationVerbalizer(cfg, backbone=backbone, tokenizer=tok).to(device)
    ar = ActivationReconstructor(cfg, backbone=backbone).to(device)
    eos = tok.eos_token_id

    pool = _load_targets(args.targets)
    assert_targets_healthy(args.targets, pool)
    pool = pool.to(device)

    # Same val split as train_nla_sft.py: torch.Generator(seed=0).randperm,
    # take first n_val = 5% then first val_vectors=256 of that. With 10000
    # pairs, n_val=500, val_vectors=256 → first 256 of the permutation.
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(10000, generator=g).tolist()[: args.n]
    v = pool[perm]  # (N, d)
    print(f"val targets: N={v.size(0)}  d={v.size(1)}")

    results = {}

    def run(label: str, ckpt: Path, do_sample: bool, k: int):
        t0 = time.time()
        # reload ckpt
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        sd = sd["trainable"] if isinstance(sd, dict) and "trainable" in sd else sd
        miss, unexp = av.load_state_dict(sd, strict=False)
        av.eval()
        f = measure(av, ar, eos, v, args.batch, args.max_new,
                    do_sample=do_sample, temperature=args.temp, top_p=args.top_p, k=k)
        dt = time.time() - t0
        print(f"  {label}: fve={f:.4f}  (took {dt:.1f}s)")
        results[label] = f

    print("\n== WARM-START AV ==")
    run("warm.greedy",    args.warm, do_sample=False, k=1)
    run("warm.bon_K32",   args.warm, do_sample=True,  k=args.k)

    print("\n== SFT iter-1 AV ==")
    run("sft.greedy",     args.sft,  do_sample=False, k=1)
    run("sft.bon_K32",    args.sft,  do_sample=True,  k=args.k)

    print("\n== SUMMARY ==")
    for k, v_ in results.items():
        print(f"  {k:18s}  {v_:.4f}")

    print("\n== DIAGNOSIS ==")
    sft_gap = results["sft.bon_K32"] - results["sft.greedy"]
    sft_delta = results["sft.greedy"] - results["warm.greedy"]
    bon_delta = results["sft.bon_K32"] - results["warm.bon_K32"]
    print(f"  sft K32 - sft greedy   = {sft_gap:+.4f}   (decoding gap)")
    print(f"  sft greedy - warm greedy = {sft_delta:+.4f}   (greedy improvement)")
    print(f"  sft K32 - warm K32     = {bon_delta:+.4f}   (manifold improvement)")
    if results["sft.bon_K32"] > results["warm.bon_K32"] + 0.02:
        print("  → policy MOVED. Continue BoN+SFT iteration.")
    if sft_gap > 0.10:
        print("  → big decoding gap. Try low-temp sampling or distillation with sampling.")
    if sft_delta < 0.02 and sft_gap < 0.05:
        print("  → policy did NOT move and greedy ≈ K32. SFT signal too weak: raise lr.")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
