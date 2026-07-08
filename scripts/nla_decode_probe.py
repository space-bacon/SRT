"""Quick decode probe: what does a trained AV actually generate?

Loads an AV checkpoint, decodes N val vectors (greedy + K sampled), prints
the text alongside centered scores, and dumps JSONL. The fast way to tell
degenerate-basin generation from a measurement bug.

    python scripts/nla_decode_probe.py \\
        --av-ckpt artifacts/nla/gemma4/ce_seq64_np16/best_av.pt \\
        --targets artifacts/nla/gemma4/targets_L47_seq64_10k.pt \\
        --pairs   artifacts/nla/gemma4/gold_pairs_seq64.jsonl \\
        --backbone google/gemma-4-31B-it --layer 47 \\
        --num-prefix-tokens 16 --prepend-bos --num-vectors 8 --K 4 \\
        --out artifacts/nla/gemma4/decode_probe.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from srt.nla import ActivationVerbalizer, NLAConfig, load_frozen_backbone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("decode_probe")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--av-ckpt", required=True, type=Path)
    p.add_argument("--targets", required=True, type=Path)
    p.add_argument("--pairs", required=True, type=Path)
    p.add_argument("--backbone", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--num-prefix-tokens", type=int, default=16)
    p.add_argument("--num-inject-slots", type=int, default=1)
    p.add_argument("--prepend-bos", action="store_true")
    p.add_argument("--num-vectors", type=int, default=8)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone, tok = load_frozen_backbone(args.backbone, args.dtype, device=device)
    cfg = NLAConfig(
        backbone_id=args.backbone, backbone_dtype=args.dtype,
        extraction_layer=args.layer,
        num_prefix_tokens=args.num_prefix_tokens,
        num_inject_slots=args.num_inject_slots,
    )
    av = ActivationVerbalizer(cfg, backbone=backbone, tokenizer=tok).to(device)
    ck = torch.load(args.av_ckpt, map_location=device, weights_only=False)
    state = ck["trainable"] if isinstance(ck, dict) and "trainable" in ck else ck
    own = av.state_dict()
    compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    av.load_state_dict(compatible, strict=False)
    av.eval()
    log.info("AV loaded %d/%d keys (ckpt step %s)", len(compatible), len(state),
             ck.get("step") if isinstance(ck, dict) else "?")

    obj = torch.load(args.targets, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "targets" in obj:
        pool = obj["targets"].float()
    else:
        pool = torch.stack([a[-1] for a in obj["activations"]]).float()
    pairs = [json.loads(l) for l in args.pairs.open() if l.strip()]
    by_tidx = {r["target_idx"]: r for r in pairs}
    mu = pool.mean(0).to(device)

    torch.manual_seed(args.seed)
    idxs = torch.randperm(len(pairs))[: args.num_vectors].tolist()

    @torch.no_grad()
    def score(texts: list[str], v: torch.Tensor) -> list[float]:
        out = []
        bos = tok.bos_token_id
        for t in texts:
            body = tok(t, add_special_tokens=False)["input_ids"] or [tok.pad_token_id]
            ids = torch.tensor([([bos] + body) if args.prepend_bos else body],
                               device=device)
            o = backbone(input_ids=ids, output_hidden_states=True, use_cache=False)
            h = o.hidden_states[args.layer][0, -1].float()
            out.append(float(0.5 * (1 + F.cosine_similarity(
                (h - mu).unsqueeze(0), (v - mu).unsqueeze(0), dim=-1))))
        return out

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for i in idxs:
            r = pairs[i]
            v = pool[r["target_idx"]].to(device)
            gold = tok.decode(r["gold_ids"], skip_special_tokens=True)
            g_ids = av.generate(v.unsqueeze(0), max_new_tokens=args.max_new_tokens,
                                do_sample=False)
            texts = [tok.decode(g_ids[0], skip_special_tokens=True)]
            if args.K > 1:
                s_ids = av.generate(v.unsqueeze(0).expand(args.K, -1),
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=True, temperature=1.0)
                texts += [tok.decode(x, skip_special_tokens=True) for x in s_ids]
            scores = score(texts, v)
            rec = {"target_idx": r["target_idx"], "gold": gold,
                   "greedy": {"text": texts[0], "cen": scores[0]},
                   "sampled": [{"text": t, "cen": s}
                               for t, s in zip(texts[1:], scores[1:])]}
            f.write(json.dumps(rec) + "\n")
            log.info("tidx %d\n  GOLD  : %.60s\n  GREEDY (%.3f): %.60s\n  BEST-S (%.3f): %.60s",
                     r["target_idx"], gold, scores[0], texts[0],
                     max(scores[1:]) if len(scores) > 1 else float("nan"),
                     texts[1 + int(torch.tensor(scores[1:]).argmax())] if len(scores) > 1 else "")
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
