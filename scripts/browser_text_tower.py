"""Browser rung: can a 0.6B text tower carry the cross-modal bridge?

The browser plan has always had one blocker: no sub-3B rung exists, and whether
the linear cross-modal correspondence survives that small is unknown. This
answers it with the artifact we want to ship.

Architecture, justified by the program's own cross-vendor result (gemma images
searched by qwen text cost five R@1 points):

  images  ->  encoded OFFLINE through a large VLM, shipped as a gallery
  text    ->  encoded IN-BROWSER through Qwen3-0.6B (candle-wasm, ~420MB q4)
  bridge  ->  one linear map, ~4MB

Everything after the one-time model download is a matmul on the visitor's own
silicon. Zero server.

Read-out layer is selected on held-out retrieval among batching-stable layers,
the same discipline that cost 17.5 R@1 points when skipped on Qwen3.8.

    python browser_text_tower.py --image-states /root/qwen38_coco_ls/images.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TEXT_MODEL = "Qwen/Qwen3-0.6B"
N_EVAL_IMAGES = 1000
PROJ_DIM = 1024
TAU = 0.05
MAX_SEQ = 64


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-states", type=Path, required=True,
                   help="paired COCO image states [n, n_layers, d] from the big VLM")
    p.add_argument("--image-layer-index", type=int, default=5,
                   help="index into the image file's layer list (L60 for Qwen3.8)")
    p.add_argument("--ann-dir", type=Path, default=Path("/root/annotations"))
    p.add_argument("--text-model", default=TEXT_MODEL)
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="candidate text read-out layers; default = evenly spaced")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--select-epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--fit-batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("/root/browser_text_tower.json"))
    p.add_argument("--save-head", type=Path, default=Path("/root/browser_text_head.pt"))
    return p.parse_args()


@torch.no_grad()
def encode(texts, tok, model, device, layers, batch):
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        ids_list = []
        for t in chunk:
            ids = tok(t, truncation=True, max_length=MAX_SEQ, add_special_tokens=True).input_ids
            bos = tok.bos_token_id
            if bos is not None and ids[0] != bos:
                ids = [bos] + ids[: MAX_SEQ - 1]
            ids_list.append(ids)
        T = max(len(x) for x in ids_list)
        pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
        input_ids = torch.full((len(ids_list), T), pad, dtype=torch.long)
        attn = torch.zeros((len(ids_list), T), dtype=torch.long)
        for j, ids in enumerate(ids_list):
            input_ids[j, : len(ids)] = torch.tensor(ids)
            attn[j, : len(ids)] = 1
        res = model(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                    output_hidden_states=True, use_cache=False)
        last = (attn.sum(-1) - 1).to(device)
        rows = torch.arange(len(chunk), device=device)
        out.append(torch.stack([res.hidden_states[l][rows, last].float().cpu()
                                for l in layers], dim=1))
        if (i // batch) % 40 == 0:
            print(f"  {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    return torch.cat(out)


class Head(torch.nn.Module):
    def __init__(self, d_img, d_txt, proj):
        super().__init__()
        self.img = torch.nn.Linear(d_img, proj)
        self.txt = torch.nn.Linear(d_txt, proj)


def fit(Xi, Xt, args, epochs, shuffle=False, device="cuda:0"):
    g = torch.Generator().manual_seed(args.seed)
    mu_i, mu_t = Xi.mean(0), Xt.mean(0)
    Zi, Zt = (Xi - mu_i).to(device), (Xt - mu_t).to(device)
    if shuffle:
        Zt = Zt[torch.randperm(len(Zt), generator=g)]
    head = Head(Xi.shape[1], Xt.shape[1], PROJ_DIM).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    n = len(Zi)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, args.fit_batch):
            idx = perm[i:i + args.fit_batch].to(device)
            if len(idx) < 8:
                continue
            a = F.normalize(head.img(Zi[idx]), dim=-1)
            b = F.normalize(head.txt(Zt[idx]), dim=-1)
            logits = a @ b.T / TAU
            tgt = torch.arange(len(idx), device=device)
            loss = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))
            opt.zero_grad(); loss.backward(); opt.step()
    return head, mu_i, mu_t


@torch.no_grad()
def retrieval(head, mu_i, mu_t, img, cap, owner_idx, device="cuda:0"):
    zi = F.normalize(head.img((img - mu_i).to(device)), dim=-1)
    zt = F.normalize(head.txt((cap - mu_t).to(device)), dim=-1)
    S = zi @ zt.T
    owner = torch.as_tensor(owner_idx, device=device)
    out = {}
    rank = S.argsort(dim=1, descending=True)
    hit = owner[rank] == torch.arange(len(zi), device=device)[:, None]
    for k in (1, 5, 10):
        out[f"i2t_r@{k}"] = round(hit[:, :k].any(1).float().mean().item(), 4)
    rank_t = S.T.argsort(dim=1, descending=True)
    correct = owner[:, None] == rank_t
    for k in (1, 5, 10):
        out[f"t2i_r@{k}"] = round(correct[:, :k].any(1).float().mean().item(), 4)
    return out


def main() -> None:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    args = parse_args()
    dev = "cuda:0"

    I = torch.load(args.image_states, map_location="cpu", weights_only=True)
    files = I["files"]
    img = I["base"][:, args.image_layer_index] if I["base"].dim() == 3 else I["base"]
    print(f"image states {tuple(img.shape)} from layer "
          f"{I.get('layers', ['?'])[args.image_layer_index]}", flush=True)

    ann = json.loads((args.ann_dir / "captions_val2017.json").read_text())
    by_img: dict[int, list[str]] = {}
    for a in ann["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a["caption"].strip())
    id_of = {im["file_name"]: im["id"] for im in ann["images"]}
    f_idx = {f: i for i, f in enumerate(files)}
    texts, owner = [], []
    for f in files:
        for c in by_img.get(id_of[f], []):
            texts.append(c)
            owner.append(f_idx[f])
    owner = np.array(owner)
    print(f"{len(texts)} captions over {len(files)} images", flush=True)

    tok = AutoTokenizer.from_pretrained(args.text_model)
    model = AutoModelForCausalLM.from_pretrained(args.text_model, dtype=torch.bfloat16).to(dev)
    model.eval()
    n_hidden = model.config.num_hidden_layers + 1
    layers = args.layers or list(range(4, n_hidden, max(1, n_hidden // 8)))
    print(f"{args.text_model}: {n_hidden - 1} layers, d={model.config.hidden_size}, "
          f"candidates {layers}", flush=True)

    cap = encode(texts, tok, model, dev, layers, args.batch)
    del model
    torch.cuda.empty_cache()

    n_img = len(files)
    n_train = n_img - N_EVAL_IMAGES
    tr_imgs, ev_img = np.arange(n_train), np.arange(n_train, n_img)
    first = {}
    for j in np.where(owner < n_train)[0]:
        first.setdefault(int(owner[j]), int(j))
    tr_pairs = np.array([first[i] for i in tr_imgs if i in first])
    tr_imgs = np.array([i for i in tr_imgs if i in first])
    ev_cap = np.where(owner >= n_train)[0]

    print(f"\nlayer selection ({len(tr_imgs)} train pairs, {len(ev_img)} eval images)", flush=True)
    scan = {}
    for k, l in enumerate(layers):
        h, mi, mt = fit(img[tr_imgs], cap[tr_pairs, k], args, args.select_epochs, device=dev)
        r = retrieval(h, mi, mt, img[ev_img], cap[ev_cap, k], owner[ev_cap] - n_train, device=dev)
        scan[l] = r
        print(f"  L{l:<3d} i2t R@1 {r['i2t_r@1']:.4f}  R@5 {r['i2t_r@5']:.4f}  "
              f"t2i R@1 {r['t2i_r@1']:.4f}", flush=True)
    best = max(scan, key=lambda l: scan[l]["i2t_r@1"])
    bi = layers.index(best)
    print(f"selected text read-out layer L{best}\n", flush=True)

    head, mi, mt = fit(img[tr_imgs], cap[tr_pairs, bi], args, args.epochs, device=dev)
    real = retrieval(head, mi, mt, img[ev_img], cap[ev_cap, bi], owner[ev_cap] - n_train, device=dev)
    sh, smi, smt = fit(img[tr_imgs], cap[tr_pairs, bi], args, args.epochs,
                       shuffle=True, device=dev)
    shuf = retrieval(sh, smi, smt, img[ev_img], cap[ev_cap, bi],
                     owner[ev_cap] - n_train, device=dev)
    print(f"bridge      {real}", flush=True)
    print(f"shuffled    {shuf}", flush=True)

    d_txt = cap.shape[2]
    payload = {
        "text_tower": args.text_model, "text_layer": best, "d_txt": d_txt,
        "image_states": str(args.image_states),
        "image_layer": I.get("layers", [None] * 8)[args.image_layer_index],
        "proj_dim": PROJ_DIM, "n_train": len(tr_imgs), "n_eval_images": len(ev_img),
        "n_eval_captions": len(ev_cap), "layer_scan": scan,
        "bridge": real, "shuffled_control": shuf,
        "chance_i2t_r@1": round(1.0 / len(ev_cap), 6),
        "browser_payload_mb": {
            "text_model_q4": round(1.50 * 0.28 * 1000),
            "text_head": round(d_txt * PROJ_DIM * 2 / 1e6, 1),
            "gallery_1k_images_fp16": round(1000 * PROJ_DIM * 2 / 1e6, 1),
        },
    }
    args.out.write_text(json.dumps(payload, indent=2))
    torch.save({"txt": head.txt.state_dict(), "img": head.img.state_dict(),
                "mu_img": mi, "mu_txt": mt,
                "meta": {k: payload[k] for k in ("text_tower", "text_layer", "proj_dim")}},
               args.save_head)
    print(f"\nwrote {args.out} and {args.save_head}", flush=True)
    print(f"browser payload: {payload['browser_payload_mb']}", flush=True)


if __name__ == "__main__":
    main()
