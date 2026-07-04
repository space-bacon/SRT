"""Run the text-trained SRT community read-out on arbitrary image files and
report the nearest concept words + discourse communities for each.

Built for the autostereogram test: feed a colour random-dot autostereogram that
hides a shape, plus a plain visible control of the same shape. If the read-out
(and the gemma-4 vision tower under it) recovers only surface texture, the
stereogram should land on texture/pattern words while the control lands on the
shape word. Recovering the hidden figure would require binocular stereopsis,
which a flat-raster encoder does not perform.

Centred-cosine frame (matches scripts/build_crossmodal_gallery.py): image and
word/community spaces are each mean-centred before comparison, because raw
cosine on this backbone is dominated by anisotropy. A small CIFAR image pool
provides a stable image-modality mean.

Usage (venv transformers>=5):
    python scripts/stereogram_readout.py \
        --ckpt checkpoints/gemma4_readout/readout_selected.pt \
        --heldout data/phase1_heldout.jsonl \
        --images artifacts/nla/gemma4/stereo/control.png \
                 artifacts/nla/gemma4/stereo/stereogram.png \
        --out artifacts/nla/gemma4/stereogram_readout.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F

MID = "google/gemma-4-31B-it"

SHAPE_WORDS = ["heart", "star", "circle", "square", "triangle", "cross",
               "horse", "cat", "dog", "bird", "fish", "face", "tree", "flower"]
SURFACE_WORDS = ["noise", "static", "texture", "pattern", "dots", "mosaic",
                 "confetti", "colorful", "abstract", "wallpaper", "camouflage",
                 "carpet", "tile", "sand", "gravel", "speckles", "television",
                 "sprinkles", "pixels", "grain"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/gemma4_readout/readout_selected.pt")
    p.add_argument("--heldout", default="data/phase1_heldout.jsonl")
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--out", default="artifacts/nla/gemma4/stereogram_readout.json")
    p.add_argument("--text-per-comm", type=int, default=40)
    p.add_argument("--pool-per-class", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--topk-comm", type=int, default=3)
    p.add_argument("--topk-word", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.manual_seed(1)

    from PIL import Image
    from transformers import Gemma4ForConditionalGeneration, AutoProcessor
    from datasets import load_dataset
    from srt.config import SRTConfig
    from srt.modules.community import CommunityDiscoveryHead

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cc = ck["config"]
    comm_L, d = cc["community_layer"], cc["d_backbone"]
    print(f"ckpt step {ck.get('step')} | community@{comm_L} d={d}", flush=True)

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

    # ---- word anchors ----
    words = SHAPE_WORDS + SURFACE_WORDS
    word_emb = F.normalize(torch.stack(
        [text_encoded("a photo of a " + w) for w in words]), dim=-1)
    print(f"{len(words)} word anchors ({len(SHAPE_WORDS)} shape / {len(SURFACE_WORDS)} surface)", flush=True)

    # ---- image-pool mean for centring (stable anisotropy estimate) ----
    ds = load_dataset("uoft-cs/cifar10", split="test")
    per = defaultdict(int)
    pool = []
    for row in ds:
        c = int(row["label"])
        if per[c] < args.pool_per_class:
            pool.append(image_encoded(row["img"])); per[c] += 1
        if all(per[i] >= args.pool_per_class for i in range(10)):
            break
    pool = torch.stack(pool)
    print(f"image pool {pool.shape[0]} for centring", flush=True)

    # ---- test images ----
    test_emb = {}
    for path in args.images:
        img = Image.open(path).convert("RGB")
        test_emb[path] = image_encoded(img)
        print(f"encoded {path}", flush=True)

    img_mean = torch.cat([pool, torch.stack(list(test_emb.values()))]).mean(0)
    cent_c = F.normalize(cent - cent.mean(0, keepdim=True), dim=-1)
    word_c = F.normalize(word_emb - word_emb.mean(0, keepdim=True), dim=-1)

    results = {}
    for path, emb in test_emb.items():
        emb_c = F.normalize(emb - img_mean, dim=0)
        wsim = emb_c @ word_c.T
        wtop = wsim.topk(args.topk_word)
        csim = emb_c @ cent_c.T
        cprob = F.softmax(csim * 12.0, dim=0)
        ctop = cprob.topk(args.topk_comm)
        top_words = [[words[j], round(float(wsim[j]), 3)] for j in wtop.indices.tolist()]
        top_comms = [[labels[j], round(float(cprob[j]), 3)] for j in ctop.indices.tolist()]
        results[os.path.basename(path)] = {"top_words": top_words, "top_communities": top_comms}
        print(f"\n=== {os.path.basename(path)} ===", flush=True)
        print("  words:", ", ".join(f"{w}({s})" for w, s in top_words), flush=True)
        print("  communities:", ", ".join(f"{c}({p})" for c, p in top_comms), flush=True)

    out = {"model": MID, "ckpt_step": ck.get("step"),
           "shape_words": SHAPE_WORDS, "surface_words": SURFACE_WORDS,
           "results": results,
           "note": "Text-trained SRT community read-out applied to image tokens. "
                   "Centred-cosine frame with a CIFAR image-pool mean."}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
