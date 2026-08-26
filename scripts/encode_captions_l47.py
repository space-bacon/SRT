#!/usr/bin/env python
"""Encode every COCO caption to raw gemma L47 states, resumably.

The head that builds the Lab's map was fitted on ONE caption per image
(`gemma4_encode_pairs.py` writes `caption: caps[0]`). The open question is
whether that narrow text supervision is why the map's reader trails a reader
working in a five-caption space. Answering it needs the other four captions
encoded the same way the first one was, so a head can be refit with all five
and nothing else changed.

Convention must match the deployment exactly or the refit head is measuring
noise: BOS-prefixed, LAST token, layer 47 (gemma-4 is BOS-sensitive; a bare
re-encode drops replay 0.9986 -> 0.615).

Writes {out}/chunk_{i:04d}.npy, each (n_images_in_chunk, n_caps, 5376) float16,
plus a matching counts file so ragged caption lists survive the round trip.
Existing chunks are skipped, so the job is safe to kill and relaunch.

    python scripts/encode_captions_l47.py --caps /root/full_caps.json \
        --out /root/caps_l47 --chunk-size 4000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--caps", default="/root/full_caps.json")
    p.add_argument("--out", default="/root/caps_l47")
    p.add_argument("--tower", default="google/gemma-4-31B-it")
    p.add_argument("--layer", type=int, default=47)
    p.add_argument("--max-caps", type=int, default=5)
    p.add_argument("--max-seq", type=int, default=64)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--chunk-size", type=int, default=4000, help="images per chunk file")
    p.add_argument("--limit-chunks", type=int, default=0, help="stop after N chunks (timing)")
    return p.parse_args()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    a = parse()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    caps = json.load(open(a.caps))["captions"]
    n_img = len(caps)
    n_chunks = (n_img + a.chunk_size - 1) // a.chunk_size
    todo = [c for c in range(n_chunks) if not (out / f"chunk_{c:04d}.npy").exists()]
    print(f"{n_img} images, {n_chunks} chunks, {len(todo)} to do", flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(a.tower)
    tower = AutoModelForCausalLM.from_pretrained(
        a.tower, dtype=torch.bfloat16, device_map="cuda").eval()
    bos = getattr(tok, "bos_token_id", None) or 2
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    @torch.no_grad()
    def encode(texts):
        vs = []
        for i in range(0, len(texts), a.batch):
            chunk = texts[i:i + a.batch]
            ids_list = []
            for t in chunk:
                ids = tok(t, truncation=True, max_length=a.max_seq,
                          add_special_tokens=True).input_ids
                if not ids or ids[0] != bos:
                    ids = [bos] + ids[: a.max_seq - 1]
                ids_list.append(ids)
            T = max(len(x) for x in ids_list)
            inp = torch.full((len(ids_list), T), pad, dtype=torch.long)
            att = torch.zeros((len(ids_list), T), dtype=torch.long)
            for j, ids in enumerate(ids_list):
                inp[j, :len(ids)] = torch.tensor(ids)
                att[j, :len(ids)] = 1
            res = tower(input_ids=inp.cuda(), attention_mask=att.cuda(),
                        output_hidden_states=True, use_cache=False)
            last = (att.sum(-1) - 1).cuda()
            rows = torch.arange(len(chunk), device="cuda")
            vs.append(res.hidden_states[a.layer][rows, last].to(torch.float16).cpu())
        return torch.cat(vs).numpy()

    t0, done = time.time(), 0
    for n, c in enumerate(todo):
        lo, hi = c * a.chunk_size, min((c + 1) * a.chunk_size, n_img)
        rows = caps[lo:hi]
        flat, counts = [], []
        for cs in rows:
            cs = cs[: a.max_caps]
            counts.append(len(cs))
            flat.extend(cs)
        v = encode(flat)
        # Ragged lists padded to max_caps; counts say which slots are real.
        buf = np.zeros((len(rows), a.max_caps, v.shape[1]), dtype=np.float16)
        k = 0
        for j, m in enumerate(counts):
            buf[j, :m] = v[k:k + m]
            k += m
        tmp = out / f"chunk_{c:04d}.tmp.npy"
        np.save(tmp, buf)
        tmp.rename(out / f"chunk_{c:04d}.npy")
        np.save(out / f"counts_{c:04d}.npy", np.array(counts, dtype=np.int16))
        done += len(flat)
        el = time.time() - t0
        print(f"chunk {c} ({n + 1}/{len(todo)}) {len(flat)} caps  "
              f"{done / el:.0f} cap/s  elapsed {el / 60:.1f}m", flush=True)
        if a.limit_chunks and n + 1 >= a.limit_chunks:
            print("limit reached", flush=True)
            return
    print("ALL CHUNKS DONE", flush=True)


if __name__ == "__main__":
    main()
