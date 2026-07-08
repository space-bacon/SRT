"""Augment the Sunstone demo cache with open-vocabulary caption retrievals.

Adds ``top_captions`` to each category in the demo's ``gallery.json``: the
nearest COCO captions (from the 10k caption pool encoded at L47) to the
category's mean image state, under per-modality centering. Runs once on a
GPU box; the Space stays CPU-only.

The image-side mean here is the mean over ALL gallery images (150 natural
photos) — a far better modality anchor than synthetic probes.

    python scripts/augment_gallery_captions.py \\
        --gallery demo/cross_modal_space/cached/gallery.json \\
        --caption-targets artifacts/nla/gemma4/targets_caps_L47.pt \\
        --layer 47 --top-k 3
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from srt.nla import load_frozen_backbone

MID = "google/gemma-4-31B-it"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gallery", required=True)
    p.add_argument("--caption-targets", required=True)
    p.add_argument("--layer", type=int, default=47)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--processor-cache", default=None)
    args = p.parse_args()

    from PIL import Image
    from transformers import AutoProcessor

    proc_kw = {"cache_dir": args.processor_cache} if args.processor_cache else {}
    proc = AutoProcessor.from_pretrained(MID, **proc_kw)
    bb, tok = load_frozen_backbone(MID, "bfloat16", device="cuda")
    img_tok = bb.config.image_token_id

    obj = torch.load(args.caption_targets, map_location="cpu", weights_only=False)
    pool = torch.stack([a[-1] for a in obj["activations"]]).float().cuda()
    caps = obj["sequences"]
    mu_txt = pool.mean(0)
    pool_c = F.normalize(pool - mu_txt, dim=-1)
    print(f"caption pool: {pool.size(0)}", flush=True)

    with open(args.gallery) as f:
        G = json.load(f)
    cache_dir = os.path.dirname(os.path.abspath(args.gallery))

    @torch.no_grad()
    def image_v(path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Describe this image."}]}]
        enc = proc.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=True, return_dict=True,
                                       return_tensors="pt").to("cuda")
        out = bb(**enc, output_hidden_states=True, use_cache=False)
        mask = enc["input_ids"][0] == img_tok
        return out.hidden_states[args.layer][0][mask].float().mean(0)

    # per-image embeddings (raw L47 space) for every gallery image
    per_cat_vs: dict[str, list[torch.Tensor]] = {}
    all_vs = []
    for c in G["categories"]:
        vs = [image_v(os.path.join(cache_dir, t)) for t in c["thumbs"]]
        per_cat_vs[c["category"]] = vs
        all_vs += vs
        print(f"  encoded {c['category']} ({len(vs)} images)", flush=True)
    mu_img = torch.stack(all_vs).mean(0)
    print(f"||mu_img||={float(mu_img.norm()):.1f} over {len(all_vs)} images", flush=True)

    for c in G["categories"]:
        v = torch.stack(per_cat_vs[c["category"]]).mean(0)
        sims = pool_c @ F.normalize(v - mu_img, dim=-1)
        top = sims.topk(args.top_k)
        c["top_captions"] = [
            [caps[int(top.indices[i])], round(float(0.5 * (1 + top.values[i])), 3)]
            for i in range(args.top_k)
        ]
        print(f"{c['category']:>12} -> {c['top_captions'][0][1]:.3f} | "
              f"{c['top_captions'][0][0][:80]}", flush=True)

    G["caption_pool"] = {
        "n": int(pool.size(0)),
        "source": "sentence-transformers/coco-captions (deduped)",
        "layer": args.layer,
        "note": "Open-vocabulary retrieval: category mean L47 image state vs "
                "10k COCO captions, per-modality centered. Zero training.",
    }
    with open(args.gallery, "w") as f:
        json.dump(G, f, indent=2)
    print(f"updated {args.gallery}", flush=True)


if __name__ == "__main__":
    main()
