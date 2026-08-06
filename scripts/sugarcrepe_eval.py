#!/usr/bin/env python
"""SugarCrepe through the SRT-Sunstone head: does frozen-VLM-state
retrieval inherit the text tower's compositionality?

SugarCrepe (RAIVNLab/sugar-crepe): 7,512 image / caption / hard-negative
triples over COCO val2017, where the negative is a compositional
perturbation (swap/replace/add of objects, attributes, relations).
CLIP-class dual encoders sit near chance on the swap splits because
their text encoders are order-insensitive. Our text encoder is a frozen
31B LLM read at layer 47, so the hypothesis is a text-side
compositionality carry-over.

Protocol: score = head-space cosine(image, caption). A triple is
correct when cos(img, positive) > cos(img, negative). Reported per
split for (a) the shipped v1 head, (b) the v3 drift head, (c) raw
mean-centered L47 states (no head), the zero-training baseline.

Usage on a CUDA box (resumable; encodes cache to --work-dir):
    python sugarcrepe_eval.py --work-dir /root/sugarcrepe --img-dir /root/val2017
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

import numpy as np
import torch

BACKBONE = "google/gemma-4-31B-it"
LAYER = 47
MAX_SEQ = 64
BATCH = 16
HEAD_REPO = "RiverRider/srt-sunstone-linear-head"
HEADS = {"v1": "sunstone_linear_head.pt",
         "v3_drift": "sunstone_linear_head_v3_drift.pt"}
SPLITS = ["add_att", "add_obj", "replace_att", "replace_obj",
          "replace_rel", "swap_att", "swap_obj"]
DATA_URL = ("https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/"
            "main/data/{split}.json")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", default="sugarcrepe")
    p.add_argument("--img-dir", default="val2017")
    p.add_argument("--score-only", action="store_true",
                   help="skip encoding, just score from caches")
    return p.parse_args()


def fetch_data(work_dir: str) -> dict[str, dict]:
    data = {}
    for s in SPLITS:
        path = os.path.join(work_dir, f"{s}.json")
        if not os.path.exists(path):
            urllib.request.urlretrieve(DATA_URL.format(split=s), path)
        with open(path) as f:
            data[s] = json.load(f)
    return data


def load_model():
    from transformers import AutoProcessor, AutoModelForImageTextToText

    proc = AutoProcessor.from_pretrained(BACKBONE)
    model = AutoModelForImageTextToText.from_pretrained(
        BACKBONE, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return proc, model


@torch.no_grad()
def encode_images(files: list[str], img_dir: str, cache: str,
                  proc, model) -> dict[str, np.ndarray]:
    """Mean L47 state over image-token positions (Sunstone convention)."""
    from PIL import Image

    done: dict[str, np.ndarray] = {}
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        done = {str(k): v for k, v in zip(z["files"], z["states"])}
    todo = [f for f in files if f not in done]
    print(f"images: {len(done)} cached, {len(todo)} to encode", flush=True)
    img_tok = getattr(model.config, "image_token_id", None)
    for i, fname in enumerate(todo):
        img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Describe this image."},
        ]}]
        enc = proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to("cuda")
        out = model(**enc, output_hidden_states=True, use_cache=False)
        mask = enc["input_ids"][0] == img_tok
        done[fname] = (out.hidden_states[LAYER][0][mask]
                       .float().mean(0).cpu().numpy())
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            np.savez_compressed(cache, files=list(done.keys()),
                                states=np.stack(list(done.values())))
            print(f"  {i + 1}/{len(todo)} (checkpointed)", flush=True)
    return done


@torch.no_grad()
def encode_texts(texts: list[str], cache: str, proc, model) -> dict[str, np.ndarray]:
    """Last-token L47 states, BOS enforced."""
    done: dict[str, np.ndarray] = {}
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        done = {str(k): v for k, v in zip(z["texts"], z["states"])}
    todo = sorted(set(t for t in texts if t not in done))
    print(f"texts: {len(done)} cached, {len(todo)} to encode", flush=True)
    tok = getattr(proc, "tokenizer", proc)
    bos = tok.bos_token_id
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
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
        vecs = h[rows.cuda(), last.cuda()].float().cpu().numpy()
        for t, v in zip(chunk, vecs):
            done[t] = v
        if (i // BATCH) % 20 == 0 or i + BATCH >= len(todo):
            np.savez_compressed(cache, texts=list(done.keys()),
                                states=np.stack(list(done.values())))
            print(f"  {min(i + BATCH, len(todo))}/{len(todo)}", flush=True)
    return done


def score(data, img_states, txt_states) -> None:
    from huggingface_hub import hf_hub_download

    scorers = {}
    for name, fname in HEADS.items():
        d = torch.load(hf_hub_download(HEAD_REPO, fname),
                       map_location="cpu", weights_only=True)
        Wi = d["img"]["weight"].float().numpy()
        bi = d["img"]["bias"].float().numpy()
        mi = d["mu_img"].float().numpy()
        Wt = d["txt"]["weight"].float().numpy()
        bt = d["txt"]["bias"].float().numpy()
        mt = d["mu_txt"].float().numpy()

        def make(Wi=Wi, bi=bi, mi=mi, Wt=Wt, bt=bt, mt=mt):
            def f(vi, vt):
                zi = (vi - mi) @ Wi.T + bi
                zt = (vt - mt) @ Wt.T + bt
                return float(zi @ zt / (np.linalg.norm(zi) * np.linalg.norm(zt) + 1e-8))
            return f
        scorers[name] = make()

    # raw centered baseline: center by the pool means of this benchmark
    all_img = np.stack(list(img_states.values()))
    all_txt = np.stack(list(txt_states.values()))
    mu_i_raw, mu_t_raw = all_img.mean(0), all_txt.mean(0)

    def raw_scorer(vi, vt):
        a, b = vi - mu_i_raw, vt - mu_t_raw
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    scorers["raw_centered"] = raw_scorer

    results: dict = {}
    for sname, split in data.items():
        accs = {k: 0 for k in scorers}
        n = 0
        for item in split.values():
            vi = img_states.get(item["filename"])
            vp = txt_states.get(item["caption"])
            vn = txt_states.get(item["negative_caption"])
            if vi is None or vp is None or vn is None:
                continue
            n += 1
            for k, f in scorers.items():
                if f(vi, vp) > f(vi, vn):
                    accs[k] += 1
        results[sname] = {k: round(v / max(n, 1), 4) for k, v in accs.items()}
        results[sname]["n"] = n
        print(f"{sname:14s} n={n:5d}  " +
              "  ".join(f"{k} {results[sname][k]:.3f}" for k in scorers),
              flush=True)
    # macro average
    avg = {k: round(float(np.mean([results[s][k] for s in SPLITS])), 4)
           for k in scorers}
    results["macro_avg"] = avg
    print("macro_avg      " + "  ".join(f"{k} {v:.3f}" for k, v in avg.items()),
          flush=True)
    print(json.dumps(results, indent=2))


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    data = fetch_data(args.work_dir)
    files = sorted({it["filename"] for s in data.values() for it in s.values()})
    texts = [t for s in data.values() for it in s.values()
             for t in (it["caption"], it["negative_caption"])]
    print(f"{sum(len(s) for s in data.values())} triples, "
          f"{len(files)} unique images, {len(set(texts))} unique texts",
          flush=True)

    img_cache = os.path.join(args.work_dir, "img_states.npz")
    txt_cache = os.path.join(args.work_dir, "txt_states.npz")
    if not args.score_only:
        proc, model = load_model()
        img_states = encode_images(files, args.img_dir, img_cache, proc, model)
        txt_states = encode_texts(texts, txt_cache, proc, model)
    else:
        z = np.load(img_cache, allow_pickle=True)
        img_states = {str(k): v for k, v in zip(z["files"], z["states"])}
        z = np.load(txt_cache, allow_pickle=True)
        txt_states = {str(k): v for k, v in zip(z["texts"], z["states"])}
    score(data, img_states, txt_states)


if __name__ == "__main__":
    main()
