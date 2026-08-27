"""Does the object/arrangement spectrum shift with depth?

Asked by a reader of the SugarCrepe work: are earlier layers richer on WHAT and
later layers richer on relational structure? If that gradient exists you are
looking at the projection hierarchy directly.

Two design points decide whether the answer means anything.

The shipped head is fitted on layer 47 states, so pushing layer 12 through it
measures the head being out of distribution rather than the layer being poor.
Every number here is therefore HEAD FREE: pool-mean centred cosine on raw
states, which is the same zero-training arm already reported at L47 in
artifacts/nla/q4/sugarcrepe_diagnostic_20260806.json. That file is the control.
L47 in this sweep must reproduce it.

One forward pass already computes every layer, so encoding once and keeping all
requested layers costs one pass instead of nine.

Both metrics are reported, because they answer different questions:
  acc  cos(image, positive) > cos(image, negative), the standard SugarCrepe
       accuracy, comparable to published per-split numbers.
  sep  mean cos(positive caption, negative caption). LOWER means the two
       captions are better separated. This is the metric the L47 diagnostic
       used to show raw states separate word-order swaps (0.396) better than
       object replacements (0.582).

    python sugarcrepe_layer_sweep.py --layers 4,12,20,28,36,44,47,54,60
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

import numpy as np
import torch

BACKBONE = "google/gemma-4-31B-it"
MAX_SEQ = 64
BATCH = 16
SPLITS = ["add_att", "add_obj", "replace_att", "replace_obj",
          "replace_rel", "swap_att", "swap_obj"]
DATA_URL = ("https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/"
            "main/data/{split}.json")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", default="/root/sugarcrepe")
    p.add_argument("--img-dir", default="/root/val2017")
    p.add_argument("--backbone", default=BACKBONE)
    p.add_argument("--layers", default="4,12,20,28,36,44,47,54,60")
    p.add_argument("--limit", type=int, default=0,
                   help="cap triples per split, for a smoke run")
    p.add_argument("--out", default="/root/sugarcrepe_layer_sweep.json")
    return p.parse_args()


def fetch_data(work_dir: str, limit: int) -> dict[str, dict]:
    os.makedirs(work_dir, exist_ok=True)
    data = {}
    for s in SPLITS:
        path = os.path.join(work_dir, f"{s}.json")
        if not os.path.exists(path):
            urllib.request.urlretrieve(DATA_URL.format(split=s), path)
        with open(path) as f:
            d = json.load(f)
        if limit:
            d = dict(list(d.items())[:limit])
        data[s] = d
    return data


@torch.no_grad()
def encode(data, img_dir, layers, proc, model, cache):
    """One pass over images and captions, keeping every requested layer."""
    from PIL import Image

    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print(f"resuming from {cache}", flush=True)
        return (z["img_files"].tolist(), z["img"],
                z["txts"].tolist(), z["txt"])

    files, texts = [], []
    for split in data.values():
        for it in split.values():
            files.append(it["filename"])
            texts.append(it["caption"])
            texts.append(it["negative_caption"])
    files = sorted(set(files))
    texts = sorted(set(texts))
    print(f"{len(files)} images, {len(texts)} captions, layers {layers}", flush=True)

    img_tok = getattr(model.config, "image_token_id", None)
    img = np.zeros((len(files), len(layers), model.config.text_config.hidden_size
                    if hasattr(model.config, "text_config") else 5376), np.float16)
    for i, fname in enumerate(files):
        im = Image.open(os.path.join(img_dir, fname)).convert("RGB")
        msg = [{"role": "user", "content": [
            {"type": "image", "image": im},
            {"type": "text", "text": "Describe this image."}]}]
        enc = proc.apply_chat_template(msg, add_generation_prompt=True,
                                       tokenize=True, return_dict=True,
                                       return_tensors="pt").to("cuda")
        out = model(**enc, output_hidden_states=True, use_cache=False)
        mask = enc["input_ids"][0] == img_tok
        for li, L in enumerate(layers):
            img[i, li] = out.hidden_states[L][0][mask].float().mean(0).cpu().numpy()
        if (i + 1) % 200 == 0:
            print(f"  img {i + 1}/{len(files)}", flush=True)

    tok = getattr(proc, "tokenizer", proc)
    bos = tok.bos_token_id
    txt = np.zeros((len(texts), len(layers), img.shape[2]), np.float16)
    for i in range(0, len(texts), BATCH):
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
        last = (attn.sum(-1) - 1).cuda()
        rows = torch.arange(len(chunk)).cuda()
        for li, L in enumerate(layers):
            txt[i:i + len(chunk), li] = (out.hidden_states[L][rows, last]
                                         .float().cpu().numpy().astype(np.float16))
        if (i // BATCH) % 50 == 0:
            print(f"  txt {i}/{len(texts)}", flush=True)

    np.savez_compressed(cache, img_files=np.array(files), img=img,
                        txts=np.array(texts), txt=txt)
    return files, img, texts, txt


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main() -> None:
    a = parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    data = fetch_data(a.work_dir, a.limit)

    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(a.backbone)
    model = AutoModelForImageTextToText.from_pretrained(
        a.backbone, dtype=torch.bfloat16, device_map="auto").eval()

    cache = os.path.join(a.work_dir, f"states_{'_'.join(map(str, layers))}.npz")
    files, img, texts, txt = encode(data, a.img_dir, layers, proc, model, cache)
    fi = {f: i for i, f in enumerate(files)}
    ti = {t: i for i, t in enumerate(texts)}

    results: dict = {}
    for li, L in enumerate(layers):
        # Centre by this layer's own pool mean, per modality. Anisotropy differs
        # by layer and by modality, so a shared centre would confound the sweep.
        I = img[:, li].astype(np.float32)
        T = txt[:, li].astype(np.float32)
        mi, mt = I.mean(0), T.mean(0)
        Ic, Tc = I - mi, T - mt
        per = {}
        for sname, split in data.items():
            hit = n = 0
            seps = []
            for it in split.values():
                vi = Ic[fi[it["filename"]]]
                vp, vn = Tc[ti[it["caption"]]], Tc[ti[it["negative_caption"]]]
                n += 1
                if cos(vi, vp) > cos(vi, vn):
                    hit += 1
                seps.append(cos(vp, vn))
            per[sname] = {"acc": round(hit / max(n, 1), 4),
                          "sep": round(float(np.mean(seps)), 4), "n": n}
        per["macro_acc"] = round(
            float(np.mean([per[s]["acc"] for s in SPLITS])), 4)
        results[f"L{L}"] = per
        print(f"L{L}: macro_acc {per['macro_acc']:.4f}  "
              f"replace_obj {per['replace_obj']['acc']:.3f}  "
              f"swap_obj {per['swap_obj']['acc']:.3f}", flush=True)

    out = {"question": "does the object/arrangement spectrum shift with depth",
           "arm": "head-free, pool-mean centred cosine on raw states",
           "control": "L47 must reproduce sugarcrepe_diagnostic_20260806.json "
                      "sep: swap_obj 0.396, replace_obj 0.582, "
                      "swap_att 0.498, replace_rel 0.485",
           "sep_note": "sep = mean cos(positive, negative caption), lower is "
                       "better separated",
           "backbone": a.backbone, "layers": layers, "results": results}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
