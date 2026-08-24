"""Refit the browser bridge head on 118K pairs instead of 4,000.

The shipped head was fitted on 4,000 image-caption pairs, and it shows: on the
deployed 123K gallery it retrieves at t2i R@1 0.0148. This refits the same
architecture on the whole of COCO train2017 and evaluates on the whole of
val2017, which is a cleaner split than the original had available: the old head
trained and tested inside val2017, this one never sees a val image.

Everything else is deliberately unchanged from browser_text_tower.py, including
one caption per training image, so a difference in the result is a difference in
data rather than in recipe.

    python scripts/browser_head_118k.py \
        --raw-dir /root/raw118k --val-states /root/qwen38_coco_ls/images.pt \
        --ann /root/annotations --out /root/browser_head_118k.pt
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJ_DIM = 1024
TAU = 0.05
MAX_SEQ = 64


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", required=True, help="chunks with raw image states")
    p.add_argument("--val-states", required=True, help="images.pt holding val2017")
    p.add_argument("--val-layer-index", type=int, default=5, help="slot of layer 52")
    p.add_argument("--ann", required=True, help="COCO annotations dir")
    p.add_argument("--text-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--fit-batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--select-epochs", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_captions(ann_dir: Path, name: str) -> dict[str, str]:
    """First caption per image, matching the original recipe."""
    with open(ann_dir / name) as f:
        d = json.load(f)
    id2file = {im["id"]: im["file_name"] for im in d["images"]}
    first: dict[str, str] = {}
    for a in sorted(d["annotations"], key=lambda x: x["id"]):
        fn = id2file.get(a["image_id"])
        if fn is not None:
            first.setdefault(fn, a["caption"])
    return first


def load_all_captions(ann_dir: Path, name: str) -> list[tuple[str, str]]:
    with open(ann_dir / name) as f:
        d = json.load(f)
    id2file = {im["id"]: im["file_name"] for im in d["images"]}
    out = []
    for a in d["annotations"]:
        fn = id2file.get(a["image_id"])
        if fn is not None:
            out.append((fn, a["caption"]))
    return out


@torch.no_grad()
def encode(texts, tok, model, device, layers, batch):
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        ids_list = []
        for t in chunk:
            ids = tok(t, truncation=True, max_length=MAX_SEQ,
                      add_special_tokens=True).input_ids
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
        if (i // batch) % 100 == 0:
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
    for ep in range(epochs):
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
        if not shuffle:
            print(f"    epoch {ep + 1}/{epochs} loss {loss.item():.4f}", flush=True)
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
    a = parse_args()
    ann = Path(a.ann)
    dev = a.device

    files, raws = [], []
    for f in sorted(glob.glob(f"{a.raw_dir}/shard*_chunk*.npz")):
        z = np.load(f, allow_pickle=True)
        if "raw" not in z.files:
            raise SystemExit(f"{f} has no raw states; re-encode with --save-raw")
        raws.append(z["raw"].astype(np.float32))
        files += [str(x) for x in z["files"]]
    img_train = torch.from_numpy(np.concatenate(raws))
    del raws
    print(f"train images {tuple(img_train.shape)}", flush=True)

    val = torch.load(a.val_states, map_location="cpu", weights_only=False)
    img_val = val["base"][:, a.val_layer_index, :].float()
    val_files = [str(x) for x in val["files"]]
    print(f"val images {tuple(img_val.shape)}", flush=True)

    train_first = load_captions(ann, "captions_train2017.json")
    keep = [i for i, f in enumerate(files) if f in train_first]
    img_train = img_train[keep]
    train_texts = [train_first[files[i]] for i in keep]
    print(f"{len(train_texts)} training pairs (one caption per image)", flush=True)

    val_pairs = load_all_captions(ann, "captions_val2017.json")
    vrow = {f: i for i, f in enumerate(val_files)}
    val_pairs = [(f, t) for f, t in val_pairs if f in vrow]
    val_texts = [t for _, t in val_pairs]
    val_owner = np.array([vrow[f] for f, _ in val_pairs])
    print(f"{len(val_texts)} eval captions over {len(val_files)} val images", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.text_model)
    model = AutoModelForCausalLM.from_pretrained(
        a.text_model, dtype=torch.bfloat16).to(dev).eval()
    n_hidden = model.config.num_hidden_layers + 1
    layers = a.layers or list(range(4, n_hidden, max(1, n_hidden // 8)))
    print(f"candidate text layers {layers}", flush=True)

    print("encoding training captions", flush=True)
    cap_train = encode(train_texts, tok, model, dev, layers, a.batch)
    print("encoding eval captions", flush=True)
    cap_val = encode(val_texts, tok, model, dev, layers, a.batch)
    del model
    torch.cuda.empty_cache()

    scan = {}
    for k, l in enumerate(layers):
        h, mi, mt = fit(img_train, cap_train[:, k], a, a.select_epochs, device=dev)
        r = retrieval(h, mi, mt, img_val, cap_val[:, k], val_owner, device=dev)
        scan[l] = r
        print(f"  L{l:<3d} i2t R@1 {r['i2t_r@1']:.4f}  t2i R@1 {r['t2i_r@1']:.4f}", flush=True)
    best = max(scan, key=lambda l: scan[l]["i2t_r@1"])
    bi = layers.index(best)
    print(f"selected text layer L{best}", flush=True)

    head, mi, mt = fit(img_train, cap_train[:, bi], a, a.epochs, device=dev)
    real = retrieval(head, mi, mt, img_val, cap_val[:, bi], val_owner, device=dev)
    sh, smi, smt = fit(img_train, cap_train[:, bi], a, a.epochs, shuffle=True, device=dev)
    shuf = retrieval(sh, smi, smt, img_val, cap_val[:, bi], val_owner, device=dev)
    print(f"bridge   {real}", flush=True)
    print(f"shuffled {shuf}", flush=True)

    torch.save({
        "img": {k: v.cpu() for k, v in head.img.state_dict().items()},
        "txt": {k: v.cpu() for k, v in head.txt.state_dict().items()},
        "mu_img": mi.cpu(), "mu_txt": mt.cpu(),
        "meta": {
            "text_tower": a.text_model, "text_layer": best, "proj_dim": PROJ_DIM,
            "image_layer": 52, "n_train": len(train_texts),
            "n_eval_images": len(val_files), "n_eval_captions": len(val_texts),
            "split": "train on COCO train2017, evaluate on all of val2017",
            "layer_scan": scan, "bridge": real, "shuffled_control": shuf,
        },
    }, a.out)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
