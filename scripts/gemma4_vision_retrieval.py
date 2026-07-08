"""Cross-modal retrieval decode on gemma-4: image → L47 state → nearest pool text.

Per-modality centering built in (the hard-learned Sunstone lesson): the image
query is centered by a mean over synthetic probe images (mu_img), the text
pool by its own mean (mu_txt). Without this every image retrieves the single
most image-like text (the ASCII-art attractor).

Pool = any sample_targets.py .pt (use a caption-style corpus for images that
depict objects/shapes; a forum-text pool has nothing correct to retrieve).

    python scripts/gemma4_vision_retrieval.py \\
        --targets artifacts/nla/gemma4/targets_caps_L47.pt \\
        --images artifacts/nla/gemma4/stereo/control.png ... \\
        --layer 47 --top-k 3 \\
        --out artifacts/nla/gemma4/vision_caps_retrieval.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from srt.nla import load_frozen_backbone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("vision_retrieval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default="google/gemma-4-31B-it")
    p.add_argument("--targets", required=True, type=Path,
                   help="text pool .pt from sample_targets.py")
    p.add_argument("--images", nargs="+", required=True, type=Path)
    p.add_argument("--layer", type=int, default=47)
    p.add_argument("--num-probe-images", type=int, default=24,
                   help="synthetic images for the image-side mean")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--processor-cache", default=None,
                   help="alternate cache_dir for AutoProcessor (stale-NFS workaround)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from PIL import Image, ImageDraw
    from transformers import AutoProcessor

    proc_kw = {"cache_dir": args.processor_cache} if args.processor_cache else {}
    proc = AutoProcessor.from_pretrained(args.backbone, **proc_kw)
    backbone, tok = load_frozen_backbone(args.backbone, args.dtype, device=device)
    image_token_id = getattr(backbone.config, "image_token_id", None)
    assert image_token_id is not None, "backbone config lacks image_token_id"

    obj = torch.load(args.targets, map_location="cpu", weights_only=False)
    pool = torch.stack([a[-1] for a in obj["activations"]]).float().to(device)
    seqs = obj["sequences"]
    mu_txt = pool.mean(0)
    pool_c = F.normalize(pool - mu_txt, dim=-1)
    log.info("pool N=%d  ||mu_txt||=%.1f", pool.size(0), float(mu_txt.norm()))

    @torch.no_grad()
    def image_v(img) -> torch.Tensor:
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

    # image-side mean from synthetic probes: flats, shapes, sparse noise
    random.seed(args.seed)
    probes = []
    for i in range(args.num_probe_images):
        rc = lambda: tuple(random.randrange(256) for _ in range(3))
        im = Image.new("RGB", (256, 256), rc())
        d = ImageDraw.Draw(im)
        if i % 4 == 1:
            d.ellipse([64, 64, 192, 192], fill=rc())
        elif i % 4 == 2:
            d.rectangle([48, 80, 208, 176], fill=rc())
        elif i % 4 == 3:
            px = im.load()
            for x in range(0, 256, 4):
                for y in range(0, 256, 4):
                    px[x, y] = rc()
        probes.append(image_v(im))
    mu_img = torch.stack(probes).mean(0)
    log.info("||mu_img||=%.1f (from %d probes)", float(mu_img.norm()), len(probes))

    results = []
    for path in args.images:
        img = Image.open(path).convert("RGB")
        v = image_v(img)
        sims = pool_c @ F.normalize(v - mu_img, dim=-1)
        top = sims.topk(args.top_k)
        rec = {"image": str(path), "retrievals": [
            {"cen_fve": float(0.5 * (1 + top.values[i])),
             "text": seqs[int(top.indices[i])]}
            for i in range(args.top_k)
        ]}
        results.append(rec)
        log.info("%s", path.name)
        for r in rec["retrievals"]:
            log.info("  %.3f | %.90s", r["cen_fve"], r["text"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
