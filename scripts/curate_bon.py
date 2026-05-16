"""Best-of-N curation for SRT-NLA Phase C (distillation).

For each target vector in the pool, sample K candidate descriptions from
the current AV at high temperature, score each by mse_nrm of
``AR.reconstruct``, and keep top-1. Output JSONL of
``(target_idx, gold_ids, mse_best, fve_best, mse_mean, fve_mean)``
records suitable for supervised fine-tuning.

Rationale: REINFORCE plateaus near fve_nrm=0.62 because one scalar
reward per sequence cannot move 12.7M params reliably. The soft-embedding
bridge (N2-v1..v3) failed because softmax(logits/tau) @ E_token lies
outside the manifold of real token embeddings, so its gradient drives the
discrete policy off the rails. BoN curation produces gold *one-hot* token
sequences that are *verified* to give low mse, then SFT teaches AV to
reproduce them deterministically. K=32 best-of-N typically gives ~+1-2
sigma reward above the mean.

Usage:
    python scripts/curate_bon.py \\
        --targets artifacts/nla/targets_q7b_L20_10k.pt \\
        --init-from artifacts/nla/n1i_v2_best/av_step002500.pt \\
        --k 32 --batch-size 4 --temperature 1.2 --top-p 0.95 \\
        --out artifacts/nla/curated/bon_iter1.jsonl
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

from srt.nla import ActivationReconstructor, ActivationVerbalizer, NLAConfig

logger = logging.getLogger("curate_bon")

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", required=True, type=Path)
    p.add_argument("--init-from", type=Path, default=None,
                   help="AV checkpoint to seed BoN sampling from. Without "
                        "this, BoN draws from the identity-init AV which "
                        "produces near-random text — useless.")
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--k", type=int, default=32,
                   help="Candidates per target.")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Targets per batch (effective generate batch is B*K).")
    p.add_argument("--temperature", type=float, default=1.2,
                   help="Sampling temperature. Higher = more diverse "
                        "candidates = better BoN gains; too high = noise.")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on number of targets (for smoke test).")
    p.add_argument("--log-every", type=int, default=10,
                   help="Log progress every N batches.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def _first_eos_mask(gen_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    B, T = gen_ids.shape
    is_eos = gen_ids == eos_id
    pos = torch.arange(T, device=gen_ids.device).unsqueeze(0).expand(B, T)
    first_eos = torch.where(
        is_eos.any(dim=-1, keepdim=True),
        is_eos.float().argmax(dim=-1, keepdim=True).long(),
        torch.full((B, 1), T, device=gen_ids.device, dtype=torch.long),
    )
    return (pos <= first_eos).long()


def _load_targets(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    acts = obj["activations"]
    return torch.stack([a[-1] for a in acts])  # (N, d) last-token pool


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = NLAConfig(
        backbone_id=args.backbone,
        backbone_dtype=args.dtype,
        extraction_layer=args.layer,
        max_new_tokens=args.max_new_tokens,
    )

    logger.info("loading backbone %s on %s", args.backbone, device)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.backbone, torch_dtype=_DTYPE_MAP[args.dtype]
    )
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    av = ActivationVerbalizer(cfg, backbone=backbone, tokenizer=tok).to(device)
    ar = ActivationReconstructor(cfg, backbone=backbone).to(device)

    if args.init_from is not None:
        logger.info("warm-start AV from %s", args.init_from)
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        state = ckpt["trainable"] if isinstance(ckpt, dict) and "trainable" in ckpt else ckpt
        missing, unexpected = av.load_state_dict(state, strict=False)
        logger.info("loaded %d params; missing=%d unexpected=%d",
                    len(state), len(missing), len(unexpected))

    pool = _load_targets(args.targets).to(device)
    N = pool.size(0)
    if args.limit:
        N = min(N, args.limit)
    logger.info("curating BoN over N=%d targets, K=%d (effective gen batch=%d)",
                N, args.k, args.batch_size * args.k)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_fp = args.out.open("w")

    t0 = time.time()
    fve_best_sum = 0.0
    fve_mean_sum = 0.0
    seen = 0
    batches = 0

    for i in range(0, N, args.batch_size):
        idx = list(range(i, min(i + args.batch_size, N)))
        v = pool[idx]  # (B, d)
        B = v.size(0)
        v_rep = v.repeat_interleave(args.k, dim=0)  # (B*K, d)

        gen_ids = av.generate(
            v_rep,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )  # (B*K, T)
        T = gen_ids.size(1)
        attn = _first_eos_mask(gen_ids, tok.eos_token_id)

        v_hat = ar.reconstruct(gen_ids, attention_mask=attn).float()  # (B*K, d)
        v_tgt = v_rep.float()

        v_hat_n = F.normalize(v_hat, dim=-1)
        v_tgt_n = F.normalize(v_tgt, dim=-1)
        mse_per = ((v_hat_n - v_tgt_n) ** 2).sum(dim=-1)  # (B*K,)
        mse_per = mse_per.view(B, args.k)

        best_k = mse_per.argmin(dim=1)  # (B,)
        gen_ids_BK = gen_ids.view(B, args.k, T)
        attn_BK = attn.view(B, args.k, T)

        for b in range(B):
            kb = int(best_k[b].item())
            ids = gen_ids_BK[b, kb].tolist()
            a = attn_BK[b, kb].tolist()
            L = int(sum(a))
            ids_trim = ids[:L]
            mse_best = float(mse_per[b, kb].item())
            mse_mean = float(mse_per[b].mean().item())
            rec = {
                "target_idx": idx[b],
                "gold_ids": ids_trim,
                "n_tokens": L,
                "mse_best": mse_best,
                "fve_best": 1.0 - mse_best / 2.0,
                "mse_mean": mse_mean,
                "fve_mean": 1.0 - mse_mean / 2.0,
            }
            out_fp.write(json.dumps(rec) + "\n")
        out_fp.flush()

        batches += 1
        seen += B
        fve_best_sum += float((1.0 - mse_per.min(dim=1).values / 2.0).sum().item())
        fve_mean_sum += float((1.0 - mse_per.mean(dim=1) / 2.0).sum().item())

        if batches % args.log_every == 0 or seen >= N:
            elapsed = time.time() - t0
            eta = elapsed * (N - seen) / max(seen, 1)
            logger.info(
                "progress %d/%d  fve_best=%.4f fve_mean=%.4f  elapsed=%.0fm eta=%.0fm",
                seen, N,
                fve_best_sum / seen, fve_mean_sum / seen,
                elapsed / 60, eta / 60,
            )

    out_fp.close()
    logger.info("done. wrote %s (%d records)", args.out, seen)


if __name__ == "__main__":
    main()
