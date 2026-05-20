"""N0 round-trip benchmark — plumbing smoke for SRT-NLA.

Samples ``--n`` random unit vectors in the backbone's hidden dimension,
runs ``ActivationVerbalizer.generate`` → ``ActivationReconstructor.reconstruct``,
and reports ``fve_nrm`` against the input vectors.

With no training, fve is expected to be ~0 (random direction). The goal
is to verify the inject → generate → re-encode → measure loop executes
end-to-end on the chosen backbone.

Usage:
    python scripts/bench_nla_n0.py --backbone Qwen/Qwen2.5-7B --n 100
    python scripts/bench_nla_n0.py --backbone HuggingFaceTB/SmolLM-135M --n 16 --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from srt.nla import (
    ActivationReconstructor,
    ActivationVerbalizer,
    NLAConfig,
)
from srt.nla.loss import cosine_similarity, fraction_variance_explained, mse_nrm

logger = logging.getLogger("bench_nla_n0")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="optional JSON report")
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    backbone = AutoModelForCausalLM.from_pretrained(
        args.backbone, torch_dtype=_dtype(args.dtype) if device == "cuda" else torch.float32
    )
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.to(device)
    backbone.eval()
    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    cfg = NLAConfig(
        backbone_id=args.backbone,
        backbone_dtype=args.dtype if device == "cuda" else "float32",
        extraction_layer=min(args.layer, backbone.config.num_hidden_layers),
        max_new_tokens=args.max_new_tokens,
    )

    av = ActivationVerbalizer(cfg, backbone=backbone, tokenizer=tok).to(device)
    ar = ActivationReconstructor(cfg, backbone=backbone).to(device)

    d = backbone.config.hidden_size
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    v_all = F.normalize(torch.randn(args.n, d, generator=g), dim=-1).to(device)

    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    mses, fves, coss = [], [], []
    samples: list[str] = []
    t0 = time.time()
    for i in range(0, args.n, args.batch_size):
        v = v_all[i : i + args.batch_size]
        ids = av.generate(v, max_new_tokens=args.max_new_tokens, do_sample=False)
        # Keep up to and including the first EOS so AR's last-token pool
        # reads the natural sentence terminator rather than the token
        # before it (Qwen sets pad_token_id == eos_token_id).
        B, T = ids.shape
        is_eos = ids == tok.eos_token_id
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        first_eos = torch.where(
            is_eos.any(dim=-1, keepdim=True),
            is_eos.float().argmax(dim=-1, keepdim=True).long(),
            torch.full((B, 1), T, device=ids.device, dtype=torch.long),
        )
        attn = (pos <= first_eos).long()
        v_hat = ar.reconstruct(ids, attention_mask=attn).float()
        v_f = v.float()
        mses.append(mse_nrm(v_hat, v_f).item())
        fves.append(fraction_variance_explained(v_hat, v_f).item())
        coss.append(cosine_similarity(v_hat, v_f).item())
        if len(samples) < 4:
            samples.extend(tok.batch_decode(ids, skip_special_tokens=True))

    report = {
        "backbone": args.backbone,
        "layer": cfg.extraction_layer,
        "d": d,
        "n": args.n,
        "mse_nrm": sum(mses) / len(mses),
        "fve_nrm": sum(fves) / len(fves),
        "cosine": sum(coss) / len(coss),
        "elapsed_s": round(time.time() - t0, 2),
        "device": device,
        "sample_decodes": samples[:4],
    }
    logger.info("N0 bench:\n%s", json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
