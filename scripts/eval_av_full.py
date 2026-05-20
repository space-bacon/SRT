"""Unbiased greedy fve_nrm eval on full target pool, bucketed by BoN K=32 ceiling."""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from srt.nla.config import NLAConfig
from srt.nla.verbalizer import ActivationVerbalizer
from srt.nla.reconstructor import ActivationReconstructor


def _load_targets(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    acts = obj["activations"]
    return torch.stack([a[-1] for a in acts])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--targets", required=True, type=Path)
    ap.add_argument("--bon", type=Path, default=None,
                    help="BoN curated jsonl for bucketing by fve_best")
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-random", type=int, default=1000)
    ap.add_argument("--n-bucket", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda"
    dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    cfg = NLAConfig(
        backbone_id=args.backbone,
        backbone_dtype=args.dtype,
        extraction_layer=args.layer,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"loading backbone {args.backbone}...", flush=True)
    backbone = AutoModelForCausalLM.from_pretrained(args.backbone, torch_dtype=dt)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    av = ActivationVerbalizer(cfg, backbone=backbone, tokenizer=tok).to(device)
    ar = ActivationReconstructor(cfg, backbone=backbone).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt["trainable"] if isinstance(ckpt, dict) and "trainable" in ckpt else ckpt
    miss, unexp = av.load_state_dict(state, strict=False)
    print(
        f"loaded ckpt val={ckpt.get('val_fve_nrm','?')} step={ckpt.get('step','?')} "
        f"missing={len(miss)} unexpected={len(unexp)}",
        flush=True,
    )

    pool = _load_targets(args.targets).to(device)
    N = pool.shape[0]
    print(f"N targets = {N}", flush=True)

    eos_id = backbone.config.eos_token_id or tok.eos_token_id
    pad_id = tok.pad_token_id

    rng = np.random.default_rng(args.seed)
    eval_idx = rng.choice(N, size=args.n_random, replace=False)

    print(f"--- unbiased greedy eval on {args.n_random} random targets ---", flush=True)

    @torch.no_grad()
    def run(ids):
        vt = pool[ids]
        fves = []
        for i in range(0, vt.size(0), args.batch_size):
            v = vt[i : i + args.batch_size]
            gen_ids = av.generate(v, max_new_tokens=args.max_new_tokens, do_sample=False)
            B, T = gen_ids.shape
            is_eos = gen_ids == eos_id
            pos = torch.arange(T, device=gen_ids.device).unsqueeze(0).expand(B, T)
            first_eos = torch.where(
                is_eos.any(dim=-1, keepdim=True),
                is_eos.float().argmax(dim=-1, keepdim=True).long(),
                torch.full((B, 1), T, device=gen_ids.device, dtype=torch.long),
            )
            attn = (pos <= first_eos).long()
            v_hat = ar.reconstruct(gen_ids, attention_mask=attn).float()
            v_tgt = v.float()
            import torch.nn.functional as F
            v_hat_n = F.normalize(v_hat, dim=-1)
            v_tgt_n = F.normalize(v_tgt, dim=-1)
            mse_per = ((v_hat_n - v_tgt_n) ** 2).sum(dim=-1)
            fve = 1.0 - mse_per / 2.0
            fves.extend(fve.cpu().tolist())
        return np.array(fves)

    fves = run(list(eval_idx))
    print(
        f"OVERALL (N={len(fves)}): mean={fves.mean():.4f} median={np.median(fves):.4f}",
        flush=True,
    )
    for p in [10, 25, 50, 75, 90]:
        print(f"  p{p:2d}={np.percentile(fves, p):.4f}")
    print(f"  frac>=0.85: {(fves >= 0.85).mean():.3f}")
    print(f"  frac>=0.50: {(fves >= 0.50).mean():.3f}")

    if args.bon is None:
        return

    print(f"--- bucketed eval by BoN K=32 ceiling ({args.n_bucket}/bucket) ---", flush=True)
    rows = [json.loads(l) for l in open(args.bon)]
    fve_best = {r["target_idx"]: r["fve_best"] for r in rows}
    sorted_idx = sorted(range(N), key=lambda i: fve_best.get(i, 0.0))
    third = N // 3
    buckets = {
        "bot33 (hardest)":   sorted_idx[:third],
        "mid33 (medium)":    sorted_idx[third : 2 * third],
        "top33 (easiest)":   sorted_idx[2 * third :],
    }
    for name, ids in buckets.items():
        sub = list(rng.choice(ids, size=args.n_bucket, replace=False))
        fves = run(sub)
        ceil = np.mean([fve_best[i] for i in sub])
        print(
            f"  {name}: greedy mean={fves.mean():.4f} median={np.median(fves):.4f}  "
            f"frac>=.85={(fves >= 0.85).mean():.3f}  (BoN ceil={ceil:.4f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
