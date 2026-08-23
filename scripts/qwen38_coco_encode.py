"""Paired COCO val2017 encode: base and abliterated, one per GPU, same inputs.

Extreme-capability pass. Produces everything the battery needs in one run:
5,000 image states and 25,014 caption states, under BOTH models, at the read-out
layer. Downstream: Karpathy-protocol retrieval, inventory AUC against the 80
COCO categories, and the abliteration head-space question.

Image batching is verified, not assumed: the first images are encoded both
batched and one-at-a-time and the run aborts unless they agree to 1e-3 cosine.
Batched and sequential VLM encodes differ through padding, and an unverified
speedup that silently changes the states would poison every number downstream.

    python qwen38_coco_encode.py --what images --batch 8
    python qwen38_coco_encode.py --what captions --batch 64
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

BASE = "Qwen/Qwen3.8-27B"
ABL = "JonathanColetti/Qwen3.8-27B-Uncensored"
LAYER = 48
MAX_SEQ = 64


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--what", choices=["images", "captions"], required=True)
    p.add_argument("--img-dir", default="/root/val2017")
    p.add_argument("--ann-dir", default="/root/annotations")
    p.add_argument("--base", default=BASE)
    p.add_argument("--abl", default=ABL)
    p.add_argument("--layer", type=int, default=LAYER)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-dir", default="/root/qwen38_coco")
    return p.parse_args()


def load(name: str, device: str):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16).to(device)
    model.eval()
    return proc, model


@torch.no_grad()
def encode_images(files, img_dir, proc, model, device, layer, batch):
    from PIL import Image
    img_tok = getattr(model.config, "image_token_id", None)
    assert img_tok is not None, "no image_token_id on config"
    out = []
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        msgs = [[{"role": "user", "content": [
            {"type": "image", "image": Image.open(os.path.join(img_dir, f)).convert("RGB")},
            {"type": "text", "text": "Describe this image."}]}] for f in chunk]
        enc = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", padding=True).to(device)
        res = model(**enc, output_hidden_states=True, use_cache=False)
        h = res.hidden_states[layer]
        for b in range(len(chunk)):
            mask = enc["input_ids"][b] == img_tok
            out.append(h[b][mask].float().mean(0).cpu())
        if (i // batch) % 25 == 0:
            print(f"  img {min(i + batch, len(files))}/{len(files)}", flush=True)
    return torch.stack(out)


@torch.no_grad()
def encode_captions(texts, proc, model, device, layer, batch):
    tok = getattr(proc, "tokenizer", proc)
    bos = tok.bos_token_id
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        ids_list = []
        for t in chunk:
            ids = tok(t, truncation=True, max_length=MAX_SEQ, add_special_tokens=True).input_ids
            if bos is not None and ids[0] != bos:
                ids = [bos] + ids[: MAX_SEQ - 1]
            ids_list.append(ids)
        T = max(len(x) for x in ids_list)
        pad = tok.pad_token_id if tok.pad_token_id is not None else bos
        input_ids = torch.full((len(ids_list), T), pad, dtype=torch.long)
        attn = torch.zeros((len(ids_list), T), dtype=torch.long)
        for j, ids in enumerate(ids_list):
            input_ids[j, : len(ids)] = torch.tensor(ids)
            attn[j, : len(ids)] = 1
        res = model(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                    output_hidden_states=True, use_cache=False)
        last = (attn.sum(-1) - 1).to(device)
        rows = torch.arange(len(chunk), device=device)
        out.append(res.hidden_states[layer][rows, last].float().cpu())
        if (i // batch) % 40 == 0:
            print(f"  cap {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    return torch.cat(out)


def verify_batching(files, img_dir, proc, model, device, layer, batch):
    """Batched and one-at-a-time must agree, or the speedup is silently corrupting."""
    probe = files[: batch * 2]
    b = encode_images(probe, img_dir, proc, model, device, layer, batch)
    s = encode_images(probe, img_dir, proc, model, device, layer, 1)
    cos = torch.nn.functional.cosine_similarity(b, s, dim=-1)
    print(f"  batching check: min cos {cos.min():.6f} over {len(probe)} images", flush=True)
    assert cos.min() > 0.999, f"batched != sequential (min cos {cos.min():.6f})"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ann = json.loads((Path(args.ann_dir) / "captions_val2017.json").read_text())
    files = sorted(im["file_name"] for im in ann["images"])
    if args.limit:
        files = files[: args.limit]

    proc_b, model_b = load(args.base, "cuda:0")
    proc_a, model_a = load(args.abl, "cuda:1")

    if args.what == "images":
        verify_batching(files, args.img_dir, proc_b, model_b, "cuda:0", args.layer, args.batch)
        B = encode_images(files, args.img_dir, proc_b, model_b, "cuda:0", args.layer, args.batch)
        A = encode_images(files, args.img_dir, proc_a, model_a, "cuda:1", args.layer, args.batch)
        torch.save({"base": B, "abl": A, "files": files, "layer": args.layer},
                   out_dir / "images.pt")
        print(f"saved images base {tuple(B.shape)} abl {tuple(A.shape)}", flush=True)
    else:
        by_img: dict[int, list[str]] = {}
        for a in ann["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a["caption"].strip())
        id_of = {im["file_name"]: im["id"] for im in ann["images"]}
        texts, owner = [], []
        for f in files:
            for c in by_img.get(id_of[f], []):
                texts.append(c)
                owner.append(f)
        print(f"{len(texts)} captions over {len(files)} images", flush=True)
        B = encode_captions(texts, proc_b, model_b, "cuda:0", args.layer, args.batch)
        A = encode_captions(texts, proc_a, model_a, "cuda:1", args.layer, args.batch)
        torch.save({"base": B, "abl": A, "texts": texts, "owner": owner,
                    "layer": args.layer}, out_dir / "captions.pt")
        print(f"saved captions base {tuple(B.shape)} abl {tuple(A.shape)}", flush=True)


if __name__ == "__main__":
    main()
