"""Slot-pooled image encode for the multi-vector head (post-width-null).

The 0.705 SugarCrepe wall survived weight sweeps, family unions, and a
4x width increase; the slot pilot (slot_pool_pilot.py) showed raw L47
spatial bands separate exactly the splits the trained head is worst at
(swap_obj +6.5 from chance, swap_att +3.2, replace_rel +2.3). This
script re-encodes the training images keeping S=5 slots per image:
4 contiguous bands of the 260-token image sequence + the global mean
(the global mean reproduces the existing pipeline's image state).

Alignment: reads the per-chunk `files` lists from the EXISTING pair
chunks (gemma4_encode_pairs.py output) and encodes in that exact order,
emitting one slot chunk per pair chunk with the same row count. A
hard-assert compares the stored global-mean slot against the chunk's
`img` state on the first row of every chunk (cos > 0.99) so a silent
misalignment cannot survive.

Batched: every image yields the identical 279-token sequence, so
batching needs no padding logic.

    python gemma4_encode_slots.py \
        --chunks $SNAP/procrustes/train_pairs/chunk_*.pt \
        --img-dir train2017 --out-dir slot_chunks --batch 12
    python gemma4_encode_slots.py \
        --calib $SNAP/procrustes/encoded_L47_n5000.pt \
        --img-dir val2017 --eval-out eval_slots.pt --batch 12
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

BACKBONE = "google/gemma-4-31B-it"
LAYER = 47
N_BANDS = 4


def load_model():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(BACKBONE)
    model = AutoModelForImageTextToText.from_pretrained(
        BACKBONE, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    img_tok = getattr(model.config, "image_token_id", None)
    assert img_tok is not None
    return proc, model, img_tok


def encode_batch(proc, model, img_tok, paths: list[str]) -> torch.Tensor:
    """Returns [B, N_BANDS+1, d] float16 slot states."""
    from PIL import Image
    messages = [[{"role": "user", "content": [
        {"type": "image", "image": Image.open(p).convert("RGB")},
        {"type": "text", "text": "Describe this image."},
    ]}] for p in paths]
    enc = proc.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[LAYER]                       # [B, T, d]
    slots = []
    for b in range(len(paths)):
        mask = enc["input_ids"][b] == img_tok
        toks = h[b][mask].float()                      # [260, d]
        bands = [c.mean(0) for c in torch.chunk(toks, N_BANDS, dim=0)]
        slots.append(torch.stack(bands + [toks.mean(0)]))
    return torch.stack(slots).half().cpu()             # [B, 5, d]


def run_chunks(args, proc, model, img_tok) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    for cpath in args.chunks:
        name = Path(cpath).stem                        # chunk_000
        opath = os.path.join(args.out_dir, f"slot_{name}.pt")
        if os.path.exists(opath):
            print(f"skip {name} (exists)", flush=True)
            continue
        ch = torch.load(cpath, map_location="cpu", weights_only=True)
        files, ref_img = ch["files"], ch["img"].float()
        rows = []
        for i in range(0, len(files), args.batch):
            batch = [os.path.join(args.img_dir, f)
                     for f in files[i:i + args.batch]]
            rows.append(encode_batch(proc, model, img_tok, batch))
            if (i // args.batch) % 50 == 0:
                print(f"  {name} {i + len(batch)}/{len(files)}", flush=True)
        slots = torch.cat(rows)                        # [n, 5, d]
        # alignment hard-assert: global-mean slot == chunk img state
        cos = torch.nn.functional.cosine_similarity(
            slots[0, N_BANDS].float(), ref_img[0], dim=0).item()
        assert cos > 0.99, f"{name}: slot/global-mean cos {cos:.4f}; " \
                           f"alignment broken"
        torch.save({"slots": slots, "files": files,
                    "layer": LAYER, "n_bands": N_BANDS,
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
        batch = [os.path.join(args.img_dir, f) for f in files[i:i + args.batch]]
        rows.append(encode_batch(proc, model, img_tok, batch))
        if (i // args.batch) % 20 == 0:
            print(f"  eval {i + len(batch)}/{len(files)}", flush=True)
    slots = torch.cat(rows)
    cos = torch.nn.functional.cosine_similarity(
        slots[0, N_BANDS].float(), ref_img[0], dim=0).item()
    assert cos > 0.99, f"eval slot/global-mean cos {cos:.4f}; alignment broken"
    torch.save({"slots": slots, "files": files, "layer": LAYER,
                "n_bands": N_BANDS, "backbone": BACKBONE}, args.eval_out)
    print(f"wrote {args.eval_out} {tuple(slots.shape)} align_cos={cos:.4f}",
          flush=True)
    print("EVAL_DONE", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunks", nargs="*", default=None,
                   help="pair chunk .pt files; slot chunks are emitted "
                        "row-aligned, one per input chunk")
    p.add_argument("--img-dir", required=True)
    p.add_argument("--out-dir", default="slot_chunks")
    p.add_argument("--calib", default=None,
                   help="encoded_L47_n5000.pt; encodes the eval-tail images")
    p.add_argument("--eval-out", default="eval_slots.pt")
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--batch", type=int, default=12)
    args = p.parse_args()

    proc, model, img_tok = load_model()
    if args.chunks:
        run_chunks(args, proc, model, img_tok)
    if args.calib:
        run_eval(args, proc, model, img_tok)


if __name__ == "__main__":
    main()
