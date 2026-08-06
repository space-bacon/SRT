#!/usr/bin/env python
"""Encode the compositional negatives (or any JSON list of texts) to
L47 last-token states, row-aligned, resumable in 5k blocks.

    python encode_texts_bulk.py --texts comp_negs/comp_negatives.json \
        --key negatives --out comp_negs/neg_states.pt
"""
from __future__ import annotations

import argparse
import json
import os

import torch

BACKBONE = "google/gemma-4-31B-it"
LAYER = 47
MAX_SEQ = 64
BATCH = 32
BLOCK = 5000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--texts", required=True, help="json file with a list")
    p.add_argument("--key", default=None, help="key if the json is a dict")
    p.add_argument("--out", required=True)
    return p.parse_args()


@torch.no_grad()
def main():
    from transformers import AutoProcessor, AutoModelForImageTextToText

    args = parse_args()
    with open(args.texts) as f:
        data = json.load(f)
    texts = data[args.key] if args.key else data
    print(f"{len(texts)} texts", flush=True)

    part = args.out + ".part"
    blocks: list[torch.Tensor] = []
    if os.path.exists(part):
        blocks = torch.load(part, map_location="cpu", weights_only=True)
        print(f"resuming: {sum(b.size(0) for b in blocks)} done", flush=True)
    done = sum(b.size(0) for b in blocks)

    proc = AutoProcessor.from_pretrained(BACKBONE)
    tok = getattr(proc, "tokenizer", proc)
    model = AutoModelForImageTextToText.from_pretrained(
        BACKBONE, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    bos = tok.bos_token_id

    buf: list[torch.Tensor] = []
    for i in range(done, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        ids_list = []
        for t in chunk:
            ids = tok(t, truncation=True, max_length=MAX_SEQ,
                      add_special_tokens=True).input_ids
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
        out = model(input_ids=input_ids.cuda(), attention_mask=attn.cuda(),
                    output_hidden_states=True, use_cache=False)
        h = out.hidden_states[LAYER]
        last = attn.sum(-1) - 1
        rows = torch.arange(h.size(0))
        buf.append(h[rows.cuda(), last.cuda()].float().cpu())
        n_buf = sum(b.size(0) for b in buf)
        if n_buf >= BLOCK or i + BATCH >= len(texts):
            blocks.append(torch.cat(buf))
            buf = []
            torch.save(blocks, part)
            total = sum(b.size(0) for b in blocks)
            print(f"  {total}/{len(texts)} (checkpointed)", flush=True)

    X = torch.cat(blocks)
    torch.save({"states": X, "layer": LAYER, "backbone": BACKBONE}, args.out)
    os.remove(part)
    print(f"saved {tuple(X.shape)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
