#!/usr/bin/env python
"""Re-encode SugarCrepe captions with the provenance formatting removed.

Round 16 of the public review: the generated foil ends with a period and the
human COCO caption often does not (1384 of 1384 differing pairs favour the
foil, zero counter-examples). Text states are cached under the raw caption
string, so testing whether a scorer reads that tell needs a re-encode rather
than a harness change.

Emits the same {texts, states} npz as the original caches, keyed by the
NORMALISED caption, so the control script can look states up the same way.

    python sugarcrepe_encode_norm.py --backbone Qwen/Qwen2.5-VL-3B-Instruct \
        --layer 29 --out sc_txt_states_qwen3b_norm.npz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

SPLITS = ["add_obj", "add_att", "replace_obj", "replace_att", "replace_rel",
          "swap_obj", "swap_att"]
MAX_SEQ = 64
BATCH = 32
BLOCK = 2000


def normalise(text: str) -> str:
    """Strip a trailing period and lowercase the first character."""
    t = text.strip()
    while t.endswith("."):
        t = t[:-1].rstrip()
    return (t[:1].lower() + t[1:]) if t else t


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="sugarcrepe")
    p.add_argument("--backbone", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--layer", type=int, default=29)
    p.add_argument("--out", required=True)
    p.add_argument("--raw", action="store_true",
                   help="encode unmodified captions; the re-encode control")
    return p.parse_args()


def collect(data_dir: str, raw: bool) -> list[str]:
    seen: dict[str, None] = {}
    for s in SPLITS:
        with open(os.path.join(data_dir, f"{s}.json")) as f:
            rows = json.load(f)
        for it in (rows.values() if isinstance(rows, dict) else rows):
            for key in ("caption", "negative_caption"):
                t = it[key] if raw else normalise(it[key])
                seen.setdefault(t, None)
    return list(seen)


@torch.no_grad()
def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args = parse_args()
    texts = collect(args.data_dir, args.raw)
    print(f"{len(texts)} unique texts ({'raw' if args.raw else 'normalised'})",
          flush=True)

    part = args.out + ".part.pt"
    blocks: list[torch.Tensor] = []
    if os.path.exists(part):
        blocks = torch.load(part, map_location="cpu", weights_only=True)
        print(f"resuming: {sum(b.size(0) for b in blocks)} done", flush=True)
    done = sum(b.size(0) for b in blocks)

    proc = AutoProcessor.from_pretrained(args.backbone)
    tok = getattr(proc, "tokenizer", proc)
    model = AutoModelForImageTextToText.from_pretrained(
        args.backbone, dtype=torch.bfloat16, device_map="auto")
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
        h = out.hidden_states[args.layer]
        last = attn.sum(-1) - 1
        rows = torch.arange(h.size(0))
        buf.append(h[rows.cuda(), last.cuda()].float().cpu())
        if sum(b.size(0) for b in buf) >= BLOCK or i + BATCH >= len(texts):
            blocks.append(torch.cat(buf))
            buf = []
            torch.save(blocks, part)
            print(f"  {sum(b.size(0) for b in blocks)}/{len(texts)}", flush=True)

    states = torch.cat(blocks).numpy().astype(np.float32)
    np.savez(args.out, texts=np.array(texts), states=states)
    os.remove(part)
    print(f"saved {states.shape} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
