"""Encode a COCO-scale image set into head space, sharded across GPUs.

The 27B tower runs here, once, and what leaves is 2KB per image instead of
10KB of raw hidden state. Projecting on the box rather than after the download
is the difference between shipping a 242MB gallery and a 1.2GB one, and the
projection is the same arithmetic either way.

Resumable by construction: each shard writes numbered chunks and skips any it
already finds, so a dropped instance costs one chunk rather than the run.

    # one process per GPU
    for i in 0 1; do
      CUDA_VISIBLE_DEVICES=$i python scripts/qwen38_gallery_encode.py \
        --img-dir /root/train2017 --head browser_text_head_qwen3_0.6B.pt \
        --out-dir /root/gallery118k --shard $i --shards 2 &
    done
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

BASE = "Qwen/Qwen3.8-27B"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--img-dir", required=True)
    p.add_argument("--head", required=True, help="head .pt with an img side")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default=BASE)
    p.add_argument("--layer", type=int, default=52)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--chunk", type=int, default=2000, help="images per output file")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--save-raw", action="store_true",
                   help="also keep the pre-projection states. Projecting on the "
                        "box saves bandwidth but throws away the only thing a "
                        "new head can be fitted on, which then costs a whole "
                        "re-encode. Keep them when the head might change.")
    p.add_argument("--verify", default=None,
                   help="npz of already-shipped gallery vectors to reproduce before starting")
    p.add_argument("--verify-img-dir", default=None)
    p.add_argument("--verify-only", action="store_true")
    return p.parse_args()


@torch.no_grad()
def encode_batch(files, img_dir, proc, model, device, layer, img_tok):
    from PIL import Image

    msgs = [[{"role": "user", "content": [
        {"type": "image", "image": Image.open(os.path.join(img_dir, f)).convert("RGB")},
        {"type": "text", "text": "Describe this image."}]}] for f in files]
    enc = proc.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt", padding=True).to(device)
    res = model(**enc, output_hidden_states=True, use_cache=False)

    # Mean over image tokens, in float32. The existing gallery was built this
    # way and pooling in bf16 instead drifts: hundreds of vectors are summed
    # per image, and the error accumulates into a different point in space.
    out = []
    for b in range(len(files)):
        mask = enc["input_ids"][b] == img_tok
        out.append(res.hidden_states[layer][b][mask].float().mean(0).cpu().numpy())
    return np.stack(out)


def verify(a, proc, model, device, img_tok, W, b, mu):
    """Encode a few images that are already in the shipped gallery and compare.

    Extending a gallery only works if the new vectors land in the same space as
    the old ones. A pooling or layer mismatch produces vectors that look
    perfectly reasonable on their own and are quietly incomparable, so this is
    checked against known-good output before spending GPU-hours.
    """
    ref = np.load(a.verify, allow_pickle=True)
    ref_files = [str(f) for f in ref["files"]]
    ref_Z = ref["Z_img"].astype(np.float32)
    pick = ref_files[: a.batch]

    V = encode_batch(pick, a.verify_img_dir, proc, model, device, a.layer, img_tok)
    Z = (V - mu) @ W.T + b
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8

    cos = [float(Z[i] @ ref_Z[ref_files.index(f)]) for i, f in enumerate(pick)]
    worst = min(cos)
    print(f"verify: cosine to shipped gallery min {worst:.4f} mean "
          f"{sum(cos) / len(cos):.4f} over {len(cos)} images", flush=True)
    if worst < 0.99:
        raise SystemExit(
            f"verification failed ({worst:.4f}): this encoder does not reproduce "
            "the existing gallery, so extending it would mix two spaces")
    print("verify: same space, safe to extend", flush=True)


def main() -> None:
    a = parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in os.listdir(a.img_dir) if f.endswith(".jpg"))
    if a.limit:
        files = files[: a.limit]
    mine = files[a.shard :: a.shards]
    print(f"shard {a.shard}/{a.shards}: {len(mine)} of {len(files)} images", flush=True)

    head = torch.load(a.head, map_location="cpu", weights_only=True)
    W = head["img"]["weight"].float().numpy()
    b = head["img"]["bias"].float().numpy()
    mu = head["mu_img"].float().numpy()

    from transformers import AutoProcessor, AutoModelForImageTextToText
    device = "cuda"
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16).to(device).eval()
    img_tok = model.config.image_token_id

    if a.verify:
        verify(a, proc, model, device, img_tok, W, b, mu)
        if a.verify_only:
            return

    n_chunks = (len(mine) + a.chunk - 1) // a.chunk
    for c in range(n_chunks):
        path = out / f"shard{a.shard}_chunk{c:04d}.npz"
        if path.exists():
            print(f"  chunk {c}/{n_chunks} present, skipping", flush=True)
            continue
        part = mine[c * a.chunk : (c + 1) * a.chunk]
        vecs, names = [], []
        for i in range(0, len(part), a.batch):
            chunk_files = part[i : i + a.batch]
            try:
                vecs.append(encode_batch(chunk_files, a.img_dir, proc, model,
                                         device, a.layer, img_tok))
                names += chunk_files
            except Exception as e:
                # One unreadable image should not cost the chunk.
                print(f"    skip {chunk_files}: {type(e).__name__} {e}", flush=True)
            if i % (a.batch * 25) == 0:
                print(f"  chunk {c}/{n_chunks} {i}/{len(part)}", flush=True)

        V = np.concatenate(vecs)
        Z = (V - mu) @ W.T + b
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
        out_arrays = {"Z_img": Z.astype(np.float16), "files": np.array(names)}
        if a.save_raw:
            out_arrays["raw"] = V.astype(np.float16)
        np.savez_compressed(path, **out_arrays)
        print(f"  wrote {path} {Z.shape}" + (" +raw" if a.save_raw else ""), flush=True)

    (out / f"shard{a.shard}.done").write_text(json.dumps({
        "model": a.model, "layer": a.layer, "n": len(mine), "head": Path(a.head).name,
    }))
    print(f"shard {a.shard} complete", flush=True)


if __name__ == "__main__":
    main()
