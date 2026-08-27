"""Is the L60 collapse lost information, or a readout that lost reach?

The depth sweep scored raw states with a centred cosine. A cosine only finds
structure in the basis it is handed, so "the forward pass discarded it" and "the
forward pass moved it somewhere a cosine cannot follow" produce the same reading.

A first attempt to separate them fitted a ridge map on 936 images and failed its
own control, losing to the un-fitted arm at L47. That was a sample-size failure,
not an answer. This runs the real recipe instead: the same contrastive head the
Lab ships, two 1024-d towers trained with InfoNCE, fitted independently at each
depth on 4,000 images and scored on 1,000 held out.

The control is L47. A fitted head there must clearly beat the head-free cosine,
because that is the layer the shipped head uses. If it does not, the instrument
is invalid again and no other layer's number means anything.

    python scripts/l60_head_refit.py --layers 28,47,54,60
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

BACKBONE = "google/gemma-4-31B-it"
BATCH = 16
MAX_SEQ = 64


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", default="/root/val2017")
    p.add_argument("--caps", default="/root/annotations/captions_val2017.json")
    p.add_argument("--layers", default="28,47,54,60")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--holdout", type=int, default=1000)
    p.add_argument("--proj-dim", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--cache", default="/root/l60_states.npz")
    p.add_argument("--out", default="/root/l60_head_refit.json")
    return p.parse_args()


@torch.no_grad()
def encode(files, caps, img_dir, layers, proc, model, cache):
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print(f"resuming from {cache}", flush=True)
        return z["img"], z["txt"]

    from PIL import Image
    d = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else 5376
    img_tok = getattr(model.config, "image_token_id", None)
    img = np.zeros((len(files), len(layers), d), np.float16)
    for i, fname in enumerate(files):
        im = Image.open(os.path.join(img_dir, fname)).convert("RGB")
        msg = [{"role": "user", "content": [
            {"type": "image", "image": im},
            {"type": "text", "text": "Describe this image."}]}]
        enc = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to("cuda")
        out = model(**enc, output_hidden_states=True, use_cache=False)
        mask = enc["input_ids"][0] == img_tok
        for li, L in enumerate(layers):
            img[i, li] = out.hidden_states[L][0][mask].float().mean(0).cpu().numpy()
        if (i + 1) % 250 == 0:
            print(f"  img {i + 1}/{len(files)}", flush=True)

    tok = getattr(proc, "tokenizer", proc)
    bos = tok.bos_token_id
    txt = np.zeros((len(caps), len(layers), d), np.float16)
    for i in range(0, len(caps), BATCH):
        chunk = caps[i:i + BATCH]
        ids_list = []
        for t in chunk:
            ids = tok(t, truncation=True, max_length=MAX_SEQ, add_special_tokens=True).input_ids
            if bos is not None and ids[0] != bos:
                ids = [bos] + ids[: MAX_SEQ - 1]
            ids_list.append(ids)
        T = max(len(x) for x in ids_list)
        pad = tok.pad_token_id if tok.pad_token_id is not None else bos
        input_ids = torch.full((len(chunk), T), pad, dtype=torch.long)
        attn = torch.zeros((len(chunk), T), dtype=torch.long)
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
            print(f"  txt {i}/{len(caps)}", flush=True)

    np.savez_compressed(cache, img=img, txt=txt)
    return img, txt


def ranks(A, B):
    A = F.normalize(A, dim=1)
    B = F.normalize(B, dim=1)
    S = A @ B.T
    d = torch.arange(len(A), device=A.device)
    return (S > S[d, d][:, None]).sum(1) + 1


def report(r):
    r = r.float()
    return {"r@1": float((r == 1).float().mean()),
            "r@10": float((r <= 10).float().mean()),
            "median_rank": float(r.median())}


def fit_head(I_tr, T_tr, I_te, T_te, dim, epochs, lr, tau):
    dev = "cuda"
    mu_i, mu_t = I_tr.mean(0, keepdim=True), T_tr.mean(0, keepdim=True)
    Wi = torch.nn.Linear(I_tr.shape[1], dim).to(dev)
    Wt = torch.nn.Linear(T_tr.shape[1], dim).to(dev)
    opt = torch.optim.Adam(list(Wi.parameters()) + list(Wt.parameters()), lr=lr)
    n = len(I_tr)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            if len(idx) < 8:
                continue
            a = F.normalize(Wi(I_tr[idx] - mu_i), dim=1)
            b = F.normalize(Wt(T_tr[idx] - mu_t), dim=1)
            logits = a @ b.T / tau
            lab = torch.arange(len(idx), device=dev)
            loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        a = Wi(I_te - mu_i)
        b = Wt(T_te - mu_t)
    return a, b


def main() -> None:
    a = parse_args()
    layers = [int(x) for x in a.layers.split(",")]

    ann = json.load(open(a.caps))
    first = {}
    for c in sorted(ann["annotations"], key=lambda x: x["id"]):
        first.setdefault(c["image_id"], c["caption"].strip())
    id2file = {im["id"]: im["file_name"] for im in ann["images"]}
    ids = sorted(i for i in first if i in id2file)[:a.n]
    files = [id2file[i] for i in ids]
    caps = [first[i] for i in ids]
    print(f"{len(files)} images with a caption, layers {layers}", flush=True)

    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(BACKBONE)
    model = AutoModelForImageTextToText.from_pretrained(
        BACKBONE, dtype=torch.bfloat16, device_map="auto").eval()
    img, txt = encode(files, caps, a.img_dir, layers, proc, model, a.cache)
    del model
    torch.cuda.empty_cache()

    rng = np.random.default_rng(0)
    order = rng.permutation(len(files))
    te, tr = order[:a.holdout], order[a.holdout:]

    results = {}
    for li, L in enumerate(layers):
        I = torch.tensor(img[:, li], dtype=torch.float32, device="cuda")
        T = torch.tensor(txt[:, li], dtype=torch.float32, device="cuda")
        I_te_c = I[te] - I[te].mean(0, keepdim=True)
        T_te_c = T[te] - T[te].mean(0, keepdim=True)
        free = report(ranks(I_te_c, T_te_c))

        a_te, b_te = fit_head(I[tr], T[tr], I[te], T[te],
                              a.proj_dim, a.epochs, a.lr, a.tau)
        fitted = report(ranks(a_te, b_te))
        results[f"L{L}"] = {"head_free": free, "fitted_head": fitted}
        print(f"L{L:<3d} head-free r@1 {free['r@1']:.3f} med {free['median_rank']:.0f}   "
              f"fitted r@1 {fitted['r@1']:.3f} med {fitted['median_rank']:.0f}", flush=True)

    out = {
        "question": "is the L60 collapse lost information or a readout that lost reach",
        "method": "shipped head recipe: two 1024-d towers, InfoNCE, fitted per layer",
        "control": "a fitted head at L47 must clearly beat head-free, or the instrument is invalid",
        "n_images": len(files), "n_train": len(tr), "n_holdout": len(te),
        "backbone": BACKBONE, "results": results,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
