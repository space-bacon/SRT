"""Slot-pooling pilot: does spatial structure in the image tokens carry
compositional signal that global mean pooling destroys?

Encodes the SugarCrepe images (subset of val2017) keeping S=5 slots per
image: 4 contiguous bands of the image-token sequence (raster order ->
horizontal bands) + the global mean. Then scores every SugarCrepe item
with RAW centered cosine two ways: global-mean (the status quo) and
max-over-slots. No training anywhere; this is a pure representation
diagnostic.

If max-over-slots does not beat global mean on the swap/replace splits
here, a trained multi-vector head is unlikely to break the 0.705 wall
and the full 118K slot re-encode is not worth running.
"""
import argparse
import json
import os

import numpy as np
import torch

BACKBONE = "google/gemma-4-31B-it"
LAYER = 47
N_BANDS = 4


def encode_slots(img_dir: str, filenames: list[str], out_path: str) -> None:
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(BACKBONE)
    model = AutoModelForImageTextToText.from_pretrained(
        BACKBONE, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    img_tok = getattr(model.config, "image_token_id", None)
    assert img_tok is not None

    done = {}
    if os.path.exists(out_path):
        d = np.load(out_path, allow_pickle=True)
        done = dict(zip(list(d["files"]), list(d["states"])))
        print(f"resuming: {len(done)} already encoded", flush=True)

    todo = [f for f in filenames if f not in done]
    for i, fname in enumerate(todo):
        img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Describe this image."},
        ]}]
        enc = proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        mask = enc["input_ids"][0] == img_tok
        toks = out.hidden_states[LAYER][0][mask].float().cpu()  # [T, d]
        bands = [b.mean(0) for b in torch.chunk(toks, N_BANDS, dim=0)]
        slots = torch.stack(bands + [toks.mean(0)])  # [5, d]
        done[fname] = slots.numpy().astype(np.float16)
        if (i + 1) % 100 == 0 or i + 1 == len(todo):
            np.savez_compressed(out_path, files=list(done.keys()),
                                states=np.stack(list(done.values())))
            print(f"  {i + 1}/{len(todo)}", flush=True)
    print("encode done", flush=True)


def score(slot_path: str, sc_dir: str, out_json: str) -> None:
    d = np.load(slot_path, allow_pickle=True)
    slot_states = dict(zip(list(d["files"]),
                           [s.astype(np.float32) for s in d["states"]]))
    t = np.load(os.path.join(sc_dir, "txt_states.npz"), allow_pickle=True)
    txt_states = dict(zip(list(t["texts"]),
                          [s.astype(np.float32) for s in t["states"]]))

    # centering pools: per-slot-index mean over images; text pool mean
    all_slots = np.stack(list(slot_states.values()))       # [N, 5, d]
    mu_slots = all_slots.mean(0)                           # [5, d]
    all_txt = np.stack(list(txt_states.values()))
    mu_txt = all_txt.mean(0)

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    splits = ["replace_obj", "replace_att", "replace_rel",
              "swap_obj", "swap_att", "add_obj", "add_att"]
    results = {}
    for s in splits:
        data = json.load(open(os.path.join(sc_dir, f"{s}.json")))
        acc = {"global_mean": 0, "max_slots": 0, "mean_slots": 0}
        margins = {k: [] for k in acc}
        n = 0
        for item in data.values():
            sl = slot_states.get(item["filename"])
            vp = txt_states.get(item["caption"])
            vn = txt_states.get(item["negative_caption"])
            if sl is None or vp is None or vn is None:
                continue
            n += 1
            zp, zn = vp - mu_txt, vn - mu_txt
            csl = sl - mu_slots                            # [5, d]
            # global mean = slot 4 (appended last)
            sp = {"global_mean": (cos(csl[4], zp), cos(csl[4], zn)),
                  "max_slots": (max(cos(c, zp) for c in csl),
                                max(cos(c, zn) for c in csl)),
                  "mean_slots": (float(np.mean([cos(c, zp) for c in csl])),
                                 float(np.mean([cos(c, zn) for c in csl])))}
            for k, (p, q) in sp.items():
                acc[k] += p > q
                margins[k].append(p - q)
        results[s] = {k: round(acc[k] / max(n, 1), 4) for k in acc}
        results[s]["margin"] = {k: round(float(np.mean(margins[k])), 5)
                                for k in margins}
        results[s]["n"] = n
        print(f"{s:14s} n={n:5d}  " +
              "  ".join(f"{k} {results[s][k]:.3f}" for k in acc), flush=True)
    for k in ["global_mean", "max_slots", "mean_slots"]:
        results.setdefault("macro", {})[k] = round(
            sum(results[s][k] for s in splits) / len(splits), 4)
    print("MACRO", results["macro"], flush=True)
    json.dump(results, open(out_json, "w"), indent=1)
    print("wrote", out_json, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", default="val2017")
    p.add_argument("--sc-dir", default="sugarcrepe")
    p.add_argument("--slots", default="sc_img_slots.npz")
    p.add_argument("--out", default="slot_pilot.json")
    p.add_argument("--score-only", action="store_true")
    args = p.parse_args()

    splits = ["replace_obj", "replace_att", "replace_rel",
              "swap_obj", "swap_att", "add_obj", "add_att"]
    files = sorted({item["filename"]
                    for s in splits
                    for item in json.load(
                        open(os.path.join(args.sc_dir, f"{s}.json"))).values()})
    print(f"{len(files)} unique SugarCrepe images", flush=True)
    if not args.score_only:
        encode_slots(args.img_dir, files, args.slots)
    score(args.slots, args.sc_dir, args.out)


if __name__ == "__main__":
    main()
