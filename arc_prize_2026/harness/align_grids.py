#!/usr/bin/env python
"""Zero-training test: does Qwen3.8-27B place a grid shown as text and the same grid shown as an image in one space?

Pooled hidden states over the grid tokens (text view) or the image tokens (image view) are centred per modality with the
mean of the fit tasks, then image-to-text retrieval is scored on held-out tasks. Printed beside every result: raw
anisotropy (mean off-diagonal cosine) per modality, the uncentred retrieval, and a permuted-pair floor.
One JSON line per layer goes to <out>.jsonl; the summary is written through a temp file and renamed.
"""
import argparse, json, os, random, time
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

PALETTE = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]


def render(g):
    h, w = len(g), len(g[0])
    cell = max(14, min(32, 448 // max(h, w)))
    im = Image.new("RGB", (w * cell + 1, h * cell + 1), "#555555")
    d = ImageDraw.Draw(im)
    for r in range(h):
        for c in range(w):
            d.rectangle([c * cell + 1, r * cell + 1, (c + 1) * cell - 1, (r + 1) * cell - 1], fill=PALETTE[g[r][c]])
    return im


def pick_grids(path, n, seed, max_side=20, per_task=3):
    data = json.load(open(path))
    rng = random.Random(seed)
    tids = sorted(data)
    rng.shuffle(tids)
    out = []
    for tid in tids:
        seen, cand = set(), []
        for ex in data[tid]["train"]:
            for g in (ex["input"], ex["output"]):
                h, w = len(g), len(g[0])
                key = tuple(map(tuple, g))
                if 4 <= h <= max_side and 4 <= w <= max_side and key not in seen and len({c for r in g for c in r}) >= 2:
                    seen.add(key)
                    cand.append(g)
        rng.shuffle(cand)
        for g in cand[:per_task]:
            out.append((tid, g))
        if len(out) >= n:
            break
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default="8,16,24,32,40,48,56,64")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    items = pick_grids(a.challenges, a.n, a.seed)
    tasks = sorted({t for t, _ in items})
    random.Random(a.seed + 1).shuffle(tasks)
    fit_tasks = set(tasks[: int(0.6 * len(tasks))])
    is_fit = np.array([t in fit_tasks for t, _ in items])
    print(f"{len(items)} grids from {len(tasks)} tasks; fit {is_fit.sum()} / eval {(~is_fit).sum()}", flush=True)

    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
    img_id = tok.convert_tokens_to_ids("<|image_pad|>")

    def pooled(out, mask):
        hs = out.hidden_states
        top = len(hs) - 1
        return np.stack([hs[min(l, top)][0][mask].float().mean(0).cpu().numpy() for l in layers])

    def text_view(rows):
        msg = [{"role": "user", "content": [{"type": "text", "text": "Grid:\n" + rows + "\n"}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        s = text.index(rows)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")
        offs = enc["offset_mapping"][0].numpy()
        mask = torch.tensor([(o[0] >= s and o[1] <= s + len(rows) and o[1] > o[0]) for o in offs])
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"].cuda(), output_hidden_states=True, use_cache=False)
        return pooled(out, mask.cuda())

    def image_view(g):
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Grid image."}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        inp = proc(text=[text], images=[render(g)], return_tensors="pt").to("cuda")
        mask = inp["input_ids"][0] == img_id
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True, use_cache=False)
        return pooled(out, mask)

    XT, XI, XT2 = [], [], []
    log = open(a.out + ".jsonl", "a")
    t0 = time.time()
    for i, (tid, g) in enumerate(items):
        rows = "\n".join("".join(str(c) for c in r) for r in g)
        rows2 = "\n".join(" ".join(str(c) for c in r) for r in g)
        XT.append(text_view(rows)); XI.append(image_view(g)); XT2.append(text_view(rows2))
        if i % 100 == 0:
            log.write(json.dumps({"step": "encode", "i": i, "seconds": round(time.time() - t0, 1)}) + "\n"); log.flush()
    XT, XI, XT2 = np.stack(XT), np.stack(XI), np.stack(XT2)
    np.savez_compressed(a.out + ".states.npz", XT=XT.astype(np.float16), XI=XI.astype(np.float16), XT2=XT2.astype(np.float16),
                        is_fit=is_fit, layers=np.array(layers))

    def unit(z):
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)

    def retrieval(A, B):
        S = unit(A) @ unit(B).T
        rank = (S > np.diag(S)[:, None]).sum(1) + 1
        return {"r1": float((rank == 1).mean()), "r5": float((rank <= 5).mean()), "r10": float((rank <= 10).mean()), "median_rank": float(np.median(rank))}

    def anis(Z):
        U = unit(Z)
        S = U @ U.T
        n = len(Z)
        return float((S.sum() - n) / (n * (n - 1)))

    res = {"n_grids": len(items), "n_fit": int(is_fit.sum()), "n_eval": int((~is_fit).sum()), "chance_r1": float(1 / (~is_fit).sum()), "layers": {}}
    rng = np.random.default_rng(0)
    ev = ~is_fit
    for k, l in enumerate(layers):
        T, I, T2 = XT[:, k], XI[:, k], XT2[:, k]
        mT, mI, mT2 = T[is_fit].mean(0), I[is_fit].mean(0), T2[is_fit].mean(0)
        row = {
            "anisotropy_raw": {"text": anis(T[ev]), "image": anis(I[ev])},
            "anisotropy_centred": {"text": anis(T[ev] - mT), "image": anis(I[ev] - mI)},
            "image_to_text_raw": retrieval(I[ev], T[ev]),
            "image_to_text_centred": retrieval(I[ev] - mI, T[ev] - mT),
            "text_to_image_centred": retrieval(T[ev] - mT, I[ev] - mI),
            "text_format_control_centred": retrieval(T2[ev] - mT2, T[ev] - mT),
        }
        perm = [retrieval((I[ev] - mI)[rng.permutation(ev.sum())], T[ev] - mT)["r1"] for _ in range(100)]
        row["permuted_floor_r1"] = {"mean": float(np.mean(perm)), "max": float(np.max(perm))}
        res["layers"][str(l)] = row
        log.write(json.dumps({"step": "layer", "layer": l, **row}) + "\n"); log.flush()
    tmp = a.out + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, a.out)
    print(json.dumps({l: {"i2t_centred_r1": r["image_to_text_centred"]["r1"], "i2t_raw_r1": r["image_to_text_raw"]["r1"],
                           "anis_raw": r["anisotropy_raw"], "ctrl_r1": r["text_format_control_centred"]["r1"]} for l, r in res["layers"].items()}, indent=1))


if __name__ == "__main__":
    main()
