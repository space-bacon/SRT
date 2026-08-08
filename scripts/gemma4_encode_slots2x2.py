"""2x2 quadrant slot encode for v11 (reviewer round-9 request).

Calibration (calibrate_grid_dims.py) established that gemma-4's LM-level
image tokens form a row-major (rows, cols) grid with rows*cols = n_tokens
and cols/rows tracking the image aspect ratio; token count VARIES per
image (640x480 -> 14x19 = 266, 640x426 -> 13x20 = 260, square -> 16x16 =
256). Mirror tests: h-flip reverses grid columns (+0.03..0.05 cos gain),
v-flip reverses rows. There are no special tokens inside the image span.

Slots per image: 4 quadrant means (TL, TR, BL, BR) + global mean = 5,
so gemma4_slot_align.py and the SugarCrepe scorer run unchanged.

    python gemma4_encode_slots2x2.py --calib $SNAP/procrustes/encoded_L47_n5000.pt \
        --img-dir val2017 --eval-out eval_slots_2x2.pt --batch 12
    python gemma4_encode_slots2x2.py --sc-npz sc_img_slots.npz \
        --img-dir val2017 --sc-out sc_img_slots_2x2.npz --batch 12
    python gemma4_encode_slots2x2.py --chunks $SNAP/procrustes/train_pairs/chunk_*.pt \
        --img-dir train2017 --out-dir slot_chunks_2x2 --batch 12
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

BACKBONE = "google/gemma-4-31B-it"
LAYER = 47
N_QUADS = 4


def load_model():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(BACKBONE)
    model = AutoModelForImageTextToText.from_pretrained(
        BACKBONE, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    img_tok = getattr(model.config, "image_token_id", None)
    assert img_tok is not None
    return proc, model, img_tok


def infer_dims(n: int, w: int, h: int) -> tuple[int, int]:
    best = None
    for r in range(1, n + 1):
        if n % r:
            continue
        c = n // r
        err = abs(c / r - w / h)
        if best is None or err < best[0]:
            best = (err, r, c)
    return best[1], best[2]


def encode_batch(proc, model, img_tok, paths: list[str]) -> torch.Tensor:
    """Returns [B, N_QUADS+1, d] float16 slot states."""
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in paths]
    messages = [[{"role": "user", "content": [
        {"type": "image", "image": im},
        {"type": "text", "text": "Describe this image."},
    ]}] for im in imgs]
    enc = proc.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[LAYER]                       # [B, T, d]
    slots = []
    for b, im in enumerate(imgs):
        mask = enc["input_ids"][b] == img_tok
        toks = h[b][mask].float()                      # [n, d], n varies
        w_px, h_px = im.size
        r, c = infer_dims(toks.shape[0], w_px, h_px)
        grid = toks.reshape(r, c, -1)
        r2, c2 = (r + 1) // 2, (c + 1) // 2
        quads = [grid[:r2, :c2].mean((0, 1)), grid[:r2, c2:].mean((0, 1)),
                 grid[r2:, :c2].mean((0, 1)), grid[r2:, c2:].mean((0, 1))]
        slots.append(torch.stack(quads + [toks.mean(0)]))
    return torch.stack(slots).half().cpu()             # [B, 5, d]


def run_chunks(args, proc, model, img_tok) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    for cpath in args.chunks:
        name = Path(cpath).stem
        opath = os.path.join(args.out_dir, f"slot_{name}.pt")
        if os.path.exists(opath):
            print(f"skip {name} (exists)", flush=True)
            continue
        ch = torch.load(cpath, map_location="cpu", weights_only=True)
        files, ref_img = ch["files"], ch["img"].float()
        rows = []
        for i in range(0, len(files), args.batch):
            batch = [os.path.join(args.img_dir, os.path.basename(f))
                     for f in files[i:i + args.batch]]
            rows.append(encode_batch(proc, model, img_tok, batch))
            if (i // args.batch) % 50 == 0:
                print(f"  {name} {i + len(batch)}/{len(files)}", flush=True)
        slots = torch.cat(rows)                        # [n, 5, d]
        cos = torch.nn.functional.cosine_similarity(
            slots[0, N_QUADS].float(), ref_img[0], dim=0).item()
        assert cos > 0.99, f"{name}: slot/global-mean cos {cos:.4f}; " \
                           f"alignment broken"
        torch.save({"slots": slots, "files": files,
                    "layer": LAYER, "n_bands": N_QUADS, "kind": "quad2x2",
                    "backbone": BACKBONE}, opath)
        print(f"wrote {opath} {tuple(slots.shape)} align_cos={cos:.4f}",
              flush=True)
    print("ALL_CHUNKS_DONE", flush=True)


def run_eval(args, proc, model, img_tok) -> None:
    calib = torch.load(args.calib, map_location="cpu", weights_only=True)
    files = calib["files"][-args.n_eval:]
    ref_img = calib["img"].float()[-args.n_eval:]
    rows = []
    for i in range(0, len(files), args.batch):
        batch = [os.path.join(args.img_dir, os.path.basename(f))
                 for f in files[i:i + args.batch]]
        rows.append(encode_batch(proc, model, img_tok, batch))
        if (i // args.batch) % 20 == 0:
            print(f"  eval {i + len(batch)}/{len(files)}", flush=True)
    slots = torch.cat(rows)
    cos = torch.nn.functional.cosine_similarity(
        slots[0, N_QUADS].float(), ref_img[0], dim=0).item()
    assert cos > 0.99, f"eval slot/global-mean cos {cos:.4f}"
    torch.save({"slots": slots, "files": files, "layer": LAYER,
                "n_bands": N_QUADS, "kind": "quad2x2",
                "backbone": BACKBONE}, args.eval_out)
    print(f"wrote {args.eval_out} {tuple(slots.shape)} align_cos={cos:.4f}",
          flush=True)
    print("EVAL_DONE", flush=True)


def run_sc(args, proc, model, img_tok) -> None:
    ref = np.load(args.sc_npz)
    files = [str(f) for f in ref["files"]]
    rows = []
    for i in range(0, len(files), args.batch):
        batch = [os.path.join(args.img_dir, os.path.basename(f))
                 for f in files[i:i + args.batch]]
        rows.append(encode_batch(proc, model, img_tok, batch))
        if (i // args.batch) % 20 == 0:
            print(f"  sc {i + len(batch)}/{len(files)}", flush=True)
    slots = torch.cat(rows).numpy().astype(np.float16)
    np.savez(args.sc_out, files=np.array(files), states=slots)
    print(f"wrote {args.sc_out} {slots.shape}", flush=True)
    print("SC_DONE", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", nargs="*", default=None)
    p.add_argument("--calib", type=Path, default=None)
    p.add_argument("--sc-npz", type=Path, default=None)
    p.add_argument("--img-dir", required=True)
    p.add_argument("--out-dir", default="slot_chunks_2x2")
    p.add_argument("--eval-out", default="eval_slots_2x2.pt")
    p.add_argument("--sc-out", default="sc_img_slots_2x2.npz")
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--batch", type=int, default=12)
    args = p.parse_args()
    proc, model, img_tok = load_model()
    if args.calib is not None:
        run_eval(args, proc, model, img_tok)
    elif args.sc_npz is not None:
        run_sc(args, proc, model, img_tok)
    else:
        assert args.chunks
        run_chunks(args, proc, model, img_tok)


if __name__ == "__main__":
    main()
