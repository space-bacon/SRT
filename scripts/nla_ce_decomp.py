"""CE decomposition: does the draft help, and does the prefix poison it?

For M val rows, mean per-token CE of gold under four contexts:
  A  [prefix(v), gold]                (plain AV regime, phase-5)
  B  [prefix(v), draft, sep, gold]    (draft-conditioned regime, phase-6)
  C  [BOS, draft, sep, gold]          (pure ICL, no injection)
  D  [BOS, gold]                      (unconditional)

Decision matrix:
  C << D and B >> C  -> ICL works, injection prefix poisons it: reorder or
                        detach the prefix in draft mode.
  C ~= D             -> the backbone cannot exploit NN drafts for gold
                        prediction: draft idea needs a different scaffold.
  B << A             -> drafts help under injection after all: keep training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from srt.nla import ActivationVerbalizer, NLAConfig, load_frozen_backbone


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--av-ckpt", required=True, type=Path)
    p.add_argument("--targets", required=True, type=Path)
    p.add_argument("--pairs", required=True, type=Path)
    p.add_argument("--backbone", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--num-prefix-tokens", type=int, default=16)
    p.add_argument("--separator", default="\n\u2192 ")
    p.add_argument("--num-vectors", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone, tok = load_frozen_backbone(args.backbone, args.dtype, device=device)
    cfg = NLAConfig(
        backbone_id=args.backbone, backbone_dtype=args.dtype,
        extraction_layer=args.layer, num_prefix_tokens=args.num_prefix_tokens,
    )
    av = ActivationVerbalizer(cfg, backbone=backbone, tokenizer=tok).to(device)
    ck = torch.load(args.av_ckpt, map_location=device, weights_only=False)
    state = ck["trainable"] if "trainable" in ck else ck
    own = av.state_dict()
    av.load_state_dict({k: v for k, v in state.items()
                        if k in own and own[k].shape == v.shape}, strict=False)
    av.eval()

    obj = torch.load(args.targets, map_location="cpu", weights_only=False)
    pool = torch.stack([a[-1] for a in obj["activations"]]).float().to(device)
    pairs = [json.loads(l) for l in args.pairs.open() if l.strip()]

    # trainer-identical split + NN drafts
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(pairs), generator=g).tolist()
    n_val = max(1, int(len(pairs) * 0.05))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    val_pairs = [pairs[i] for i in val_idx[: args.num_vectors]]
    train_tidx = torch.tensor([pairs[i]["target_idx"] for i in train_idx])
    train_pool = pool[train_tidx.to(device)]
    mu = train_pool.mean(0)
    tp = F.normalize(train_pool - mu, dim=-1)

    sep_ids = tok(args.separator, add_special_tokens=False)["input_ids"]
    bos = tok.bos_token_id
    embed = backbone.get_input_embeddings()

    @torch.no_grad()
    def gold_ce(prefix_embeds: torch.Tensor | None, ctx_ids: list[int],
                gold_ids: list[int]) -> float:
        """Mean per-token CE of gold given [prefix?][ctx][gold]."""
        ids = torch.tensor([list(ctx_ids) + list(gold_ids)], device=device)
        tok_emb = embed(ids).to(av._backbone_dtype)
        if prefix_embeds is not None:
            inputs = torch.cat([prefix_embeds, tok_emb], dim=1)
        else:
            inputs = tok_emb
        out = backbone(inputs_embeds=inputs,
                       attention_mask=torch.ones(inputs.shape[:2], dtype=torch.long,
                                                 device=device),
                       use_cache=False)
        P = inputs.size(1) - ids.size(1)
        logits = out.logits[:, P - 1 + len(ctx_ids): -1].float() if P > 0 else \
                 out.logits[:, len(ctx_ids) - 1: -1].float()
        tgt = torch.tensor([gold_ids], device=device)
        return float(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                     tgt.reshape(-1)).item())

    sums = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    n = 0
    for r in val_pairs:
        v = pool[r["target_idx"]].unsqueeze(0)
        gold = r["gold_ids"][:64]
        j = int((tp @ F.normalize(pool[r["target_idx"]] - mu, dim=-1)).argmax())
        draft = pairs[train_idx[j]]["gold_ids"][:64]
        prefix = av._inject_prefix(v)
        sums["A"] += gold_ce(prefix, [], gold)
        sums["B"] += gold_ce(prefix, list(draft) + sep_ids, gold)
        sums["C"] += gold_ce(None, [bos] + list(draft) + sep_ids, gold)
        sums["D"] += gold_ce(None, [bos], gold)
        n += 1
        if n % 16 == 0:
            print(f"[{n}] " + "  ".join(f"{k}={s/n:.4f}" for k, s in sums.items()),
                  flush=True)
    print("FINAL  " + "  ".join(f"{k}={s/n:.4f}" for k, s in sums.items()))
    print("A=prefix+gold  B=prefix+draft+gold  C=BOS+draft+gold  D=BOS+gold")


if __name__ == "__main__":
    main()
