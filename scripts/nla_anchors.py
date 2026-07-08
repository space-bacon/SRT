"""Backbone-generic NLA anchors: replay ceiling / NN baseline / random floor.

Establishes the honest reference frame (standing rule: never report fve
without a centered metric and a retrieval baseline) for any backbone,
including multimodal hosts (gemma-4) via ``load_frozen_backbone``.

Inputs are the same files the trainers consume: a ``sample_targets.py``
``.pt`` and a ``build_gold_pairs.py`` JSONL.

    python scripts/nla_anchors.py \\
        --backbone google/gemma-4-31B-it --layer 47 \\
        --targets artifacts/nla/gemma4/targets_L47_seq64_10k.pt \\
        --pairs   artifacts/nla/gemma4/gold_pairs_seq64.jsonl \\
        --num-queries 100 --pool-size 2000 \\
        --out artifacts/nla/gemma4/anchors_L47.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import statistics as st
from pathlib import Path

import torch
import torch.nn.functional as F

from srt.nla import load_frozen_backbone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("nla_anchors")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--targets", required=True, type=Path)
    p.add_argument("--pairs", required=True, type=Path)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--num-queries", type=int, default=100)
    p.add_argument("--pool-size", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def fve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + F.cosine_similarity(a.float(), b.float(), dim=-1))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bb, tok = load_frozen_backbone(args.backbone, args.dtype, device=device)
    pad_id = tok.pad_token_id

    obj = torch.load(args.targets, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "targets" in obj:
        targets = obj["targets"].float()
    else:
        targets = torch.stack([a[-1] for a in obj["activations"]]).float()
    pairs = [json.loads(l) for l in args.pairs.open() if l.strip()]
    log.info("targets N=%d  pairs N=%d", targets.size(0), len(pairs))

    random.shuffle(pairs)
    M = min(args.num_queries, len(pairs) // 2)
    queries = pairs[:M]
    pool_rows = pairs[M : M + args.pool_size]
    Vq = targets[torch.tensor([r["target_idx"] for r in queries])].to(device)
    Vp = targets[torch.tensor([r["target_idx"] for r in pool_rows])].to(device)
    mu = torch.cat([Vq, Vp]).mean(0)

    @torch.no_grad()
    def encode_texts(id_lists: list[list[int]]) -> torch.Tensor:
        out = []
        for i in range(0, len(id_lists), args.batch_size):
            chunk = id_lists[i : i + args.batch_size]
            T = max(len(x) for x in chunk)
            ids = torch.full((len(chunk), T), pad_id, dtype=torch.long)
            attn = torch.zeros((len(chunk), T), dtype=torch.long)
            for j, x in enumerate(chunk):
                ids[j, : len(x)] = torch.tensor(x)
                attn[j, : len(x)] = 1
            o = bb(input_ids=ids.to(device), attention_mask=attn.to(device),
                   output_hidden_states=True, use_cache=False)
            last = (attn.sum(1) - 1).clamp(min=0)
            out.append(
                o.hidden_states[args.layer][torch.arange(len(chunk)), last].float()
            )
        return torch.cat(out)

    # 1. Replay ceiling — re-encode each query's own gold text.
    Hq = encode_texts([r["gold_ids"] for r in queries])
    replay_raw = fve(Hq, Vq)
    replay_cen = fve(Hq - mu, Vq - mu)

    # 2. NN-retrieval baseline — nearest pool target by centered cosine,
    #    then honestly re-encode the neighbour's TEXT and score vs query v.
    Vq_c = F.normalize(Vq - mu, dim=-1)
    Vp_c = F.normalize(Vp - mu, dim=-1)
    nn_idx = (Vq_c @ Vp_c.T).argmax(dim=1)
    Hnn = encode_texts([pool_rows[int(j)]["gold_ids"] for j in nn_idx])
    nn_raw = fve(Hnn, Vq)
    nn_cen = fve(Hnn - mu, Vq - mu)

    # 3. Random floor — shuffle the NN encodings against the queries.
    perm = torch.randperm(M)
    rand_raw = fve(Hnn[perm], Vq)
    rand_cen = fve(Hnn[perm] - mu, Vq - mu)

    def _s(t: torch.Tensor) -> dict:
        vals = t.cpu().tolist()
        return {"mean": st.mean(vals), "median": st.median(vals),
                "p10": sorted(vals)[max(0, int(0.1 * len(vals)) - 1)],
                "p90": sorted(vals)[min(len(vals) - 1, int(0.9 * len(vals)))]}

    result = {
        "backbone": args.backbone,
        "layer": args.layer,
        "num_queries": M,
        "pool_size": len(pool_rows),
        "mu_norm": float(mu.norm()),
        "replay": {"raw": _s(replay_raw), "centered": _s(replay_cen)},
        "nn_retrieval": {"raw": _s(nn_raw), "centered": _s(nn_cen)},
        "random_floor": {"raw": _s(rand_raw), "centered": _s(rand_cen)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    log.info("anchors -> %s", args.out)
    log.info("||mu||=%.1f  replay_cen=%.3f  nn_cen=%.3f  floor_cen=%.3f",
             result["mu_norm"], result["replay"]["centered"]["mean"],
             result["nn_retrieval"]["centered"]["mean"],
             result["random_floor"]["centered"]["mean"])


if __name__ == "__main__":
    main()
