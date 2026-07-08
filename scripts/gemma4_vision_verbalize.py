"""Verbalize gemma-4's visual percept: image-position activations → text.

The vision transition after the greedy-gap test. Extract the L{layer}
hidden state at the IMAGE soft-token positions of a frozen
``google/gemma-4-31B-it`` (mean over the ~266 image positions — the
representation the cross-modal study showed aligns with word space at
~80% depth), then decode it with the trained Activation Verbalizer.

With ``--draft`` mode (default), the AV receives the nearest TEXT
neighbour from the target pool as a draft (retrieval-then-edit), exactly
matching the train-time conditioning of ``train_nla_draft.py``. This is
the full cross-modal claim in one artifact: an image's activation,
verbalized through a text-trained decoder, with a text retrieval draft.

    python scripts/gemma4_vision_verbalize.py \\
        --av-ckpt artifacts/nla/gemma4/draft_seq64_np16/best_av.pt \\
        --targets artifacts/nla/gemma4/targets_L47_seq64_10k.pt \\
        --pairs   artifacts/nla/gemma4/gold_pairs_seq64.jsonl \\
        --layer 47 --images artifacts/nla/gemma4/stereo/control.png \\
        --out artifacts/nla/gemma4/vision_verbalize.json
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
log = logging.getLogger("vision_verbalize")

MID = "google/gemma-4-31B-it"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default=MID)
    p.add_argument("--av-ckpt", required=True, type=Path)
    p.add_argument("--targets", required=True, type=Path,
                   help="text target pool (for mu + retrieval drafts)")
    p.add_argument("--pairs", required=True, type=Path,
                   help="gold pairs (draft text lookup)")
    p.add_argument("--layer", type=int, default=47)
    p.add_argument("--images", nargs="+", required=True, type=Path)
    p.add_argument("--num-prefix-tokens", type=int, default=16)
    p.add_argument("--num-inject-slots", type=int, default=1)
    p.add_argument("--max-draft-len", type=int, default=48)
    p.add_argument("--separator", default="\n\u2192 ")
    p.add_argument("--no-draft", action="store_true",
                   help="decode from v alone (no retrieval draft)")
    p.add_argument("--K", type=int, default=8,
                   help="sampled candidates; oracle-reranked vs v")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from PIL import Image
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(args.backbone)
    backbone, tok = load_frozen_backbone(args.backbone, args.dtype, device=device)

    cfg = NLAConfig(
        backbone_id=args.backbone,
        backbone_dtype=args.dtype,
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
    log.info("AV loaded: %d/%d keys", len(compatible), len(state))

    obj = torch.load(args.targets, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "targets" in obj:
        pool = obj["targets"].float()
    else:
        pool = torch.stack([a[-1] for a in obj["activations"]]).float()
    pool = pool.to(device)
    pairs = [json.loads(l) for l in args.pairs.open() if l.strip()]
    by_tidx = {r["target_idx"]: r for r in pairs}
    mu = pool.mean(0)
    pool_c = F.normalize(pool - mu, dim=-1)
    sep_ids = tok(args.separator, add_special_tokens=False)["input_ids"]

    image_token_id = getattr(backbone.config, "image_token_id", None)
    assert image_token_id is not None, "backbone config lacks image_token_id"

    @torch.no_grad()
    def image_v(path: Path) -> torch.Tensor:
        """Mean L{layer} hidden over the image soft-token positions."""
        img = Image.open(path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Describe this image."},
        ]}]
        enc = proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(device)
        out = backbone(**enc, output_hidden_states=True, use_cache=False)
        mask = enc["input_ids"][0] == image_token_id
        return out.hidden_states[args.layer][0][mask].float().mean(0)

    @torch.no_grad()
    def reencode_last(texts: list[str]) -> torch.Tensor:
        bos = tok.bos_token_id
        hs = []
        for t in texts:
            body = tok(t, add_special_tokens=False)["input_ids"] or [tok.pad_token_id]
            # gemma-4 is BOS-sensitive: replay_cen 0.62 without BOS, 0.999 with.
            ids = torch.tensor([[bos] + body], device=device)
            o = backbone(input_ids=ids, output_hidden_states=True, use_cache=False)
            hs.append(o.hidden_states[args.layer][0, -1].float())
        return torch.stack(hs)

    results = []
    for path in args.images:
        v = image_v(path)
        v_c = F.normalize((v - mu), dim=-1)
        sims = pool_c @ v_c
        nn_j = int(sims.argmax().item())
        nn_row = by_tidx.get(nn_j)
        draft_text = None
        ctx = None
        if not args.no_draft and nn_row is not None:
            dids = nn_row["gold_ids"][: args.max_draft_len]
            draft_text = tok.decode(dids, skip_special_tokens=True)
            ctx = torch.tensor([list(dids) + list(sep_ids)],
                               dtype=torch.long, device=device)

        vb = v.unsqueeze(0)
        gen_g = av.generate(vb, max_new_tokens=args.max_new_tokens,
                            do_sample=False, context_ids=ctx)
        texts = [tok.decode(gen_g[0], skip_special_tokens=True)]
        if args.K > 1:
            v_rep = vb.expand(args.K, -1)
            ctx_rep = ctx.expand(args.K, -1) if ctx is not None else None
            gen_k = av.generate(v_rep, max_new_tokens=args.max_new_tokens,
                                do_sample=True, temperature=1.0, context_ids=ctx_rep)
            texts += [tok.decode(g, skip_special_tokens=True) for g in gen_k]
        H = reencode_last(texts)
        scores = 0.5 * (1.0 + F.cosine_similarity(H - mu, (v - mu).expand_as(H), dim=-1))
        best = int(scores.argmax().item())
        rec = {
            "image": str(path),
            "nn_draft": draft_text,
            "nn_centered_cos": float(sims[nn_j].item()),
            "greedy": {"text": texts[0], "cen_fve": float(scores[0].item())},
            "best_of_K": {"text": texts[best], "cen_fve": float(scores[best].item()),
                          "K": len(texts)},
            "all": [{"text": t, "cen_fve": float(s.item())}
                    for t, s in zip(texts, scores)],
        }
        results.append(rec)
        log.info("%s\n  draft : %s\n  greedy: %s (%.3f)\n  best  : %s (%.3f)",
                 path.name, draft_text, texts[0], scores[0], texts[best], scores[best])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
