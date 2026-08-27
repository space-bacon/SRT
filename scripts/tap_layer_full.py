"""Fit the head at full scale at L20 and L47, and settle the tap layer.

L20 beats the shipped L47 tap on two image distributions, and the gap widens
with data: +0.043 r@1 at 28,000 training images and still climbing. What is
missing is a head fitted at the scale the shipped one was, which is the
difference between evidence and a deployable candidate.

Encoding 118,287 images takes hours, so this checkpoints. A partial cache is
written every CHUNK images and resumed on restart, because losing seven hours
to one crash is the avoidable failure in a job this long.

    python scripts/tap_layer_full.py --layers 20,47
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BACKBONE = "google/gemma-4-31B-it"
BATCH = 16
MAX_SEQ = 64
CHUNK = 10000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", default="/root/train2017")
    p.add_argument("--caps", default="/root/annotations/captions_train2017.json")
    p.add_argument("--layers", default="20,47")
    p.add_argument("--n", type=int, default=118287)
    p.add_argument("--holdout", type=int, default=5000)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--cache", default="/root/full_states_L20_L47.npz")
    p.add_argument("--out", default="/root/tap_layer_full.json")
    return p.parse_args()


def pairs(caps_path: str, n: int):
    ann = json.load(open(caps_path))
    first: dict[int, str] = {}
    for c in sorted(ann["annotations"], key=lambda x: x["id"]):
        first.setdefault(c["image_id"], c["caption"].strip())
    id2file = {im["id"]: im["file_name"] for im in ann["images"]}
    ids = sorted(i for i in first if i in id2file)[:n]
    return [id2file[i] for i in ids], [first[i] for i in ids]


@torch.no_grad()
def encode(files, caps, img_dir, layers, proc, model, cache):
    from PIL import Image
    part = cache.replace(".npz", ".partial.npz")
    d = 5376
    img = np.zeros((len(files), len(layers), d), np.float16)
    txt = np.zeros((len(caps), len(layers), d), np.float16)
    done_img = done_txt = 0
    if os.path.exists(part):
        z = np.load(part, allow_pickle=False)
        # Older partials carry a single `done` for an interleaved image+caption loop.
        done_img = int(z["done_img"]) if "done_img" in z.files else int(z["done"])
        done_txt = int(z["done_txt"]) if "done_txt" in z.files else int(z["done"])
        img[:done_img] = z["img"][:done_img]
        txt[:done_txt] = z["txt"][:done_txt]
        print(f"resuming: {done_img} images, {done_txt} captions", flush=True)

    img_tok = getattr(model.config, "image_token_id", None)
    tok = getattr(proc, "tokenizer", proc)
    bos = tok.bos_token_id

    def save(ni, nt):
        np.savez(part, img=img, txt=txt, done_img=ni, done_txt=nt)
        print(f"  checkpointed {ni} images, {nt} captions", flush=True)

    t0 = time.time()
    for i in range(done_img, len(files)):
        im = Image.open(os.path.join(img_dir, files[i])).convert("RGB")
        msg = [{"role": "user", "content": [
            {"type": "image", "image": im},
            {"type": "text", "text": "Describe this image."}]}]
        enc = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to("cuda")
        out = model(**enc, output_hidden_states=True, use_cache=False)
        mask = enc["input_ids"][0] == img_tok
        for li, L in enumerate(layers):
            img[i, li] = out.hidden_states[L][0][mask].float().mean(0).cpu().numpy()

        n = i + 1 - done_img
        if n % 500 == 0:
            r = n / (time.time() - t0)
            eta = (len(files) - i - 1) / r / 3600
            print(f"  img {i + 1}/{len(files)}  {r:.2f}/s  eta {eta:.2f}h", flush=True)
        if (i + 1) % CHUNK == 0:
            save(i + 1, done_txt)
    save(len(files), done_txt)

    # A caption is ~20 tokens, so a per-caption forward is almost all weight-load
    # overhead on a 31B backbone. Batching here is what makes the run tractable.
    t0 = time.time()
    for i in range(done_txt, len(caps), BATCH):
        chunk = caps[i:i + BATCH]
        ids_list = []
        for t in chunk:
            ids = tok(t, truncation=True, max_length=MAX_SEQ,
                      add_special_tokens=True).input_ids
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

        n = i + len(chunk) - done_txt
        if (i // BATCH) % 50 == 0 and n:
            r = n / (time.time() - t0)
            print(f"  txt {i + len(chunk)}/{len(caps)}  {r:.1f}/s  "
                  f"eta {(len(caps) - i) / r / 3600:.2f}h", flush=True)
        if (i + len(chunk)) % CHUNK < BATCH:
            save(len(files), i + len(chunk))

    np.savez_compressed(cache, img=img, txt=txt)
    if os.path.exists(part):
        os.remove(part)
    return img, txt


def fit_and_score(I_tr, T_tr, I_te, T_te, dim, epochs, lr, tau):
    mu_i, mu_t = I_tr.mean(0, keepdim=True), T_tr.mean(0, keepdim=True)
    Wi = torch.nn.Linear(I_tr.shape[1], dim).cuda()
    Wt = torch.nn.Linear(T_tr.shape[1], dim).cuda()
    opt = torch.optim.Adam(list(Wi.parameters()) + list(Wt.parameters()), lr=lr)
    n = len(I_tr)
    for ep in range(epochs):
        perm = torch.randperm(n, device="cuda")
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            if len(idx) < 8:
                continue
            a = F.normalize(Wi(I_tr[idx] - mu_i), dim=1)
            b = F.normalize(Wt(T_tr[idx] - mu_t), dim=1)
            lg = a @ b.T / tau
            lab = torch.arange(len(idx), device="cuda")
            loss = 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 10 == 0:
            print(f"    epoch {ep} loss {loss.item():.4f}", flush=True)
    with torch.no_grad():
        a = F.normalize(Wi(I_te - mu_i), dim=1)
        b = F.normalize(Wt(T_te - mu_t), dim=1)
        S = a @ b.T
        dg = torch.arange(len(a), device="cuda")
        rank = (S > S[dg, dg][:, None]).sum(1) + 1
    return ({"r@1": float((rank == 1).float().mean()),
             "r@10": float((rank <= 10).float().mean()),
             "median_rank": float(rank.float().median())},
            {"img": Wi.state_dict(), "txt": Wt.state_dict(),
             "mu_img": mu_i.cpu(), "mu_txt": mu_t.cpu()})


def main() -> None:
    a = parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    files, caps = pairs(a.caps, a.n)
    print(f"{len(files)} image-caption pairs, layers {layers}", flush=True)

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
    print(f"\nfitting on {len(tr)}, scoring {len(te)}", flush=True)

    results = {}
    for li, L in enumerate(layers):
        I = torch.tensor(img[:, li], dtype=torch.float32, device="cuda")
        T = torch.tensor(txt[:, li], dtype=torch.float32, device="cuda")
        print(f"  L{L}", flush=True)
        score, head = fit_and_score(I[tr], T[tr], I[te], T[te],
                                    a.dim, a.epochs, a.lr, a.tau)
        results[f"L{L}"] = score
        torch.save(head, f"/root/head_full_L{L}.pt")
        print(f"  L{L} r@1 {score['r@1']:.4f} median {score['median_rank']:.0f}", flush=True)

    Path(a.out).write_text(json.dumps({
        "question": "does L20 beat the shipped L47 tap at full scale",
        "n_pairs": len(files), "n_train": len(tr), "n_holdout": len(te),
        "results": results}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
