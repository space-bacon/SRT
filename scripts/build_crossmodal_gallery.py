"""Offline builder for the cross-modal read-out demo gallery.

gemma-4-31B is 62.5 GB, so the demo Space cannot host it. This runs ONCE on the
GPU box to precompute everything the lightweight Space needs: for a curated set
of images, the text-trained SRT community head's assignment to discourse
communities and its nearest concept words, plus upscaled thumbnails. The Space
then loads only this cache (no model, CPU-only).

Pipeline (mirrors scripts/cross_modal_readout.py):
  - text community centroids from held-out discourse passages (35 communities)
  - word-anchor embeddings ("a photo of a {word}") for a concrete-noun vocab
  - per image: community encoder over the image soft-tokens -> top-k communities
    (softmax over centroid cosine) + top-k nearest words; save a thumbnail

Output: <out-dir>/gallery.json + <out-dir>/thumbs/*.png

Usage (venv transformers>=5):
    python scripts/build_crossmodal_gallery.py \
        --ckpt checkpoints/gemma4_readout/readout_selected.pt \
        --heldout data/phase1_heldout.jsonl --per-class 5 \
        --out-dir demo/cached_crossmodal
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F

MID = "google/gemma-4-31B-it"

# CIFAR-100 fine classes curated for an evocative, varied semiotic map.
CIFAR100_PICKS = ["bicycle", "bus", "motorcycle", "train", "rocket", "keyboard",
                  "telephone", "clock", "bottle", "cup", "mushroom", "orange",
                  "sunflower", "rose", "bridge", "house", "castle", "mountain",
                  "forest", "sea"]
# extra concept words (beyond gallery categories) for image<->word matching.
EXTRA_WORDS = ["engine", "wheel", "flower", "garden", "kitchen", "ocean",
               "building", "animal", "pet", "wildlife", "machine", "food",
               "landscape", "vehicle", "insect", "tree", "road", "sky"]
WORD_FIX = {"automobile": "car"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/gemma4_readout/readout_selected.pt")
    p.add_argument("--heldout", default="data/phase1_heldout.jsonl")
    p.add_argument("--out-dir", default="demo/cached_crossmodal")
    p.add_argument("--per-class", type=int, default=5)
    p.add_argument("--text-per-comm", type=int, default=40)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--topk-comm", type=int, default=3)
    p.add_argument("--topk-word", type=int, default=5)
    p.add_argument("--thumb", type=int, default=160)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    thumb_dir = os.path.join(args.out_dir, "thumbs")
    os.makedirs(thumb_dir, exist_ok=True)

    from transformers import Gemma4ForConditionalGeneration, AutoProcessor
    from datasets import load_dataset
    from srt.config import SRTConfig
    from srt.modules.community import CommunityDiscoveryHead

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cc = ck["config"]
    comm_L, d, dc = cc["community_layer"], cc["d_backbone"], cc["d_community"]

    proc = AutoProcessor.from_pretrained(MID)
    tok = proc.tokenizer
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MID, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    img_tok = model.config.image_token_id

    cfg = SRTConfig(backbone_id=MID)
    community = CommunityDiscoveryHead(cfg.community, d).cuda().eval()
    community.load_state_dict({k[2:]: v for k, v in ck["heads"].items() if k.startswith("0.")})

    @torch.no_grad()
    def text_encoded(text: str) -> torch.Tensor:
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=args.max_seq_len).to("cuda")
        hs = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                   output_hidden_states=True, use_cache=False).hidden_states
        return community(hs[comm_L].float(), attention_mask=enc.attention_mask).encoded[0].float().cpu()

    @torch.no_grad()
    def image_encoded(image) -> torch.Tensor:
        msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": "."}]}]
        enc = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to("cuda")
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        im = (enc["input_ids"] == img_tok).long()
        return community(hs[comm_L].float(), attention_mask=im).encoded[0].float().cpu()

    # ---- text community centroids ----
    by_comm: dict[int, list[str]] = defaultdict(list)
    comm_label: dict[int, str] = {}
    with open(args.heldout) as f:
        for line in f:
            r = json.loads(line)
            if r.get("text") and r.get("community_id") is not None:
                c = int(r["community_id"])
                if len(by_comm[c]) < args.text_per_comm:
                    by_comm[c].append(r["text"])
                    comm_label.setdefault(c, r.get("community_label", str(c)))
    comms = sorted(by_comm)
    labels = [comm_label[c].replace("reddit:", "") for c in comms]
    cent = F.normalize(torch.stack(
        [F.normalize(torch.stack([text_encoded(t) for t in by_comm[c]]).mean(0), dim=0)
         for c in comms]), dim=-1)
    print(f"built {len(comms)} community centroids", flush=True)

    # ---- image gallery ----
    ds10 = load_dataset("uoft-cs/cifar10", split="test")
    names10 = ds10.features["label"].names
    ds100 = load_dataset("uoft-cs/cifar100", split="test")
    names100 = ds100.features["fine_label"].names

    gallery = []  # (pil, category, dataset)
    per = defaultdict(int)
    for row in ds10:
        c = names10[int(row["label"])]
        if per[("c10", c)] < args.per_class:
            gallery.append((row["img"], WORD_FIX.get(c, c), "cifar10")); per[("c10", c)] += 1
    want100 = set(CIFAR100_PICKS)
    for row in ds100:
        c = names100[int(row["fine_label"])]
        if c in want100 and per[("c100", c)] < args.per_class:
            gallery.append((row["img"], c, "cifar100")); per[("c100", c)] += 1
    print(f"gallery: {len(gallery)} images", flush=True)

    # ---- word anchors ----
    cats = sorted({g[1] for g in gallery})
    words = sorted(set(cats) | set(EXTRA_WORDS))
    word_emb = F.normalize(torch.stack(
        [text_encoded("a photo of a " + w) for w in words]), dim=-1)
    print(f"{len(words)} word anchors", flush=True)

    # ---- per-image embeddings + thumbnails ----
    per_img = []
    for i, (pil, cat, dset) in enumerate(gallery):
        emb = F.normalize(image_encoded(pil), dim=0)
        fn = f"img_{i:03d}.png"
        pil.convert("RGB").resize((args.thumb, args.thumb)).save(os.path.join(thumb_dir, fn))
        per_img.append({"emb": emb, "file": f"thumbs/{fn}", "category": cat, "dataset": dset})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(gallery)}", flush=True)

    # ---- aggregate per CATEGORY (the robust, paper-level result) ----
    # Per-modality centering: raw cosine on this backbone is dominated by
    # anisotropy (unrelated vectors already cos~0.2), so we subtract each space's
    # pool mean before comparing (the validated centered-cosine frame).
    img_mean = torch.stack([r["emb"] for r in per_img]).mean(0)
    cent_c = F.normalize(cent - cent.mean(0, keepdim=True), dim=-1)
    word_c = F.normalize(word_emb - word_emb.mean(0, keepdim=True), dim=-1)
    by_cat: dict[str, list] = defaultdict(list)
    for r in per_img:
        by_cat[r["category"]].append(r)
    categories = []
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        emb = torch.stack([r["emb"] for r in rows]).mean(0)
        emb_c = F.normalize(emb - img_mean, dim=0)
        csim = emb_c @ cent_c.T
        cprob = F.softmax(csim * 12.0, dim=0)
        ctop = cprob.topk(args.topk_comm)
        wsim = emb_c @ word_c.T
        wtop = wsim.topk(args.topk_word)
        categories.append({
            "category": cat, "dataset": rows[0]["dataset"],
            "thumbs": [r["file"] for r in rows],
            "top_communities": [[labels[j], round(float(cprob[j]), 3)] for j in ctop.indices.tolist()],
            "top_words": [[words[j], round(float(wsim[j]), 3)] for j in wtop.indices.tolist()],
        })

    out = {"model": MID, "ckpt_step": ck.get("step"),
           "n_communities": len(comms), "communities": labels,
           "n_images": len(per_img), "n_categories": len(categories),
           "categories": categories,
           "note": "Community head trained on TEXT only (discourse SupCon on frozen "
                   "gemma-4-31B); applied here to image tokens with zero image training. "
                   "Assignments are aggregated per category (mean over the category's "
                   "images) to match the paper-level result."}
    with open(os.path.join(args.out_dir, "gallery.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out_dir}/gallery.json + {len(per_img)} thumbs, "
          f"{len(categories)} categories", flush=True)


if __name__ == "__main__":
    main()
