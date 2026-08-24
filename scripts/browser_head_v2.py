"""Improve the browser bridge without growing the download.

The text tower is fixed at Qwen3-0.6B because the payload is fixed. Everything
here is a lever that costs nothing, or nearly nothing, in bytes a visitor has
to fetch:

  captions   COCO ships five per image and the first fit used one, discarding
             80% of the supervision already on disk. Sampling a different one
             each epoch is 5x augmentation; putting all five in a batch instead
             would make them false negatives fighting each other.
  layers     the residual stream nests, so L28 = L25 + delta. Handing the head
             several layers lets it use the delta, which a single-layer read-out
             cannot see. The forward already computes them, so this costs only
             the head's input width: about 4MB on a 380MB payload.
  image head runs offline. The browser only receives the projected gallery,
             whose size is fixed by the index format regardless of how the
             projection was computed, so depth on this side is free.
  pooling    last-token was inherited, not chosen.

Encoding is the only slow part and it is cached, so each arm after the first is
a fit and an evaluation.

    python scripts/browser_head_v2.py --raw-dir /root/raw118k \\
        --val-states /root/qwen38_coco_ls/images.pt --ann /root/annotations \\
        --cache /root/capcache --arm baseline --out /root/v2_baseline.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJ_DIM = 1024
TAU = 0.05
MAX_SEQ = 64
LAYERS = [22, 25, 28]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--val-states", required=True)
    p.add_argument("--val-layer-index", type=int, default=5)
    p.add_argument("--ann", required=True)
    p.add_argument("--cache", required=True, help="dir for encoded caption states")
    p.add_argument("--text-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--arm", default="baseline")
    p.add_argument("--captions-per-image", type=int, default=1, choices=[1, 5])
    p.add_argument("--text-layers", type=int, nargs="+", default=[28])
    p.add_argument("--pooling", choices=["last", "mean"], default="last")
    p.add_argument("--img-head", choices=["linear", "mlp"], default="linear")
    p.add_argument("--img-hidden", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--fit-batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save-head", default=None)
    p.add_argument("--out", required=True)
    return p.parse_args()


# ---------------------------------------------------------------- encoding

@torch.no_grad()
def encode(texts, tok, model, device, layers, batch):
    """Both poolings from one forward, since the forward is the expensive part."""
    last_out, mean_out = [], []
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
        a = attn.to(device).unsqueeze(-1)
        idx = (attn.sum(-1) - 1).to(device)
        rows = torch.arange(len(chunk), device=device)
        ls, ms = [], []
        for l in layers:
            h = res.hidden_states[l].float()
            ls.append(h[rows, idx])
            ms.append((h * a).sum(1) / a.sum(1).clamp(min=1))
        last_out.append(torch.stack(ls, 1).half().cpu())
        mean_out.append(torch.stack(ms, 1).half().cpu())
        if (i // batch) % 200 == 0:
            print(f"  {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    return torch.cat(last_out), torch.cat(mean_out)


def build_cache(a, train_texts, val_texts):
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    tag = a.text_model.replace("/", "_")
    paths = {k: cache / f"{tag}_{k}.npy" for k in
             ("train_last", "train_mean", "val_last", "val_mean")}
    if all(p.exists() for p in paths.values()):
        print("caption cache hit", flush=True)
        return {k: torch.from_numpy(np.load(v, mmap_mode="r")) for k, v in paths.items()}

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.text_model)
    model = AutoModelForCausalLM.from_pretrained(
        a.text_model, dtype=torch.bfloat16).to(a.device).eval()
    print(f"encoding {len(train_texts)} train captions at {LAYERS}", flush=True)
    tl, tm = encode(train_texts, tok, model, a.device, LAYERS, a.batch)
    print(f"encoding {len(val_texts)} val captions", flush=True)
    vl, vm = encode(val_texts, tok, model, a.device, LAYERS, a.batch)
    del model
    torch.cuda.empty_cache()
    for k, v in (("train_last", tl), ("train_mean", tm), ("val_last", vl), ("val_mean", vm)):
        np.save(paths[k], v.numpy())
    return {"train_last": tl, "train_mean": tm, "val_last": vl, "val_mean": vm}


# ---------------------------------------------------------------- heads

class Bridge(nn.Module):
    def __init__(self, d_img, d_txt, proj, img_kind, img_hidden):
        super().__init__()
        # Depth on the image side never reaches a browser: only the projected
        # gallery ships, and its size is set by the index format.
        self.img = (nn.Linear(d_img, proj) if img_kind == "linear"
                    else nn.Sequential(nn.Linear(d_img, img_hidden), nn.GELU(),
                                       nn.Linear(img_hidden, proj)))
        self.txt = nn.Linear(d_txt, proj)


def fit(Xi, Xt_epochs, a, device):
    """Xt_epochs(epoch, generator) yields this epoch's text rows, aligned to Xi."""
    g = torch.Generator().manual_seed(a.seed)
    mu_i = Xi.mean(0)
    Zi = (Xi - mu_i).to(device)
    probe = Xt_epochs(0, g)
    mu_t = probe.mean(0)
    head = Bridge(Xi.shape[1], probe.shape[1], PROJ_DIM, a.img_head, a.img_hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    n = len(Zi)
    for ep in range(a.epochs):
        Zt = (Xt_epochs(ep, g) - mu_t).to(device)
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, a.fit_batch):
            idx = perm[i:i + a.fit_batch].to(device)
            if len(idx) < 8:
                continue
            x = F.normalize(head.img(Zi[idx]), dim=-1)
            y = F.normalize(head.txt(Zt[idx]), dim=-1)
            logits = x @ y.T / TAU
            tgt = torch.arange(len(idx), device=device)
            loss = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))
            opt.zero_grad(); loss.backward(); opt.step()
        del Zt
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"    epoch {ep + 1}/{a.epochs} loss {loss.item():.4f}", flush=True)
    return head, mu_i, mu_t


@torch.no_grad()
def retrieval(head, mu_i, mu_t, img, cap, owner_idx, device):
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


# ---------------------------------------------------------------- data

def load_caption_groups(ann_dir: Path, name: str):
    with open(ann_dir / name) as f:
        d = json.load(f)
    id2file = {im["id"]: im["file_name"] for im in d["images"]}
    by_file: dict[str, list[str]] = {}
    for ann in sorted(d["annotations"], key=lambda x: x["id"]):
        fn = id2file.get(ann["image_id"])
        if fn is not None:
            by_file.setdefault(fn, []).append(ann["caption"])
    return by_file


def main() -> None:
    a = parse_args()
    ann = Path(a.ann)
    dev = a.device

    files, raws = [], []
    for f in sorted(glob.glob(f"{a.raw_dir}/shard*_chunk*.npz")):
        z = np.load(f, allow_pickle=True)
        raws.append(z["raw"].astype(np.float32))
        files += [str(x) for x in z["files"]]
    img_train = torch.from_numpy(np.concatenate(raws))
    del raws

    val = torch.load(a.val_states, map_location="cpu", weights_only=False)
    img_val = val["base"][:, a.val_layer_index, :].float()
    val_files = [str(x) for x in val["files"]]

    train_groups = load_caption_groups(ann, "captions_train2017.json")
    keep = [i for i, f in enumerate(files) if f in train_groups]
    img_train = img_train[keep]
    kept_files = [files[i] for i in keep]

    # Flat caption list plus the row spans belonging to each image, so an epoch
    # can pick one caption per image and never place two of the same image in a
    # batch.
    train_texts, groups = [], []
    for f in kept_files:
        caps = train_groups[f][:5]
        groups.append(list(range(len(train_texts), len(train_texts) + len(caps))))
        train_texts += caps

    val_groups = load_caption_groups(ann, "captions_val2017.json")
    vrow = {f: i for i, f in enumerate(val_files)}
    val_texts, val_owner = [], []
    for f, caps in val_groups.items():
        if f in vrow:
            for c in caps:
                val_texts.append(c)
                val_owner.append(vrow[f])
    val_owner = np.array(val_owner)
    print(f"{len(kept_files)} images, {len(train_texts)} train captions, "
          f"{len(val_texts)} val captions", flush=True)

    cache = build_cache(a, train_texts, val_texts)
    tkey, vkey = f"train_{a.pooling}", f"val_{a.pooling}"
    li = [LAYERS.index(l) for l in a.text_layers]

    def slab(t, rows=None):
        x = t[rows] if rows is not None else t
        return x[:, li].reshape(len(x), -1).float()

    first = torch.tensor([g[0] for g in groups])

    def epoch_rows(ep, g):
        if a.captions_per_image == 1:
            return slab(cache[tkey], first)
        pick = torch.tensor([grp[torch.randint(len(grp), (1,), generator=g).item()]
                             for grp in groups])
        return slab(cache[tkey], pick)

    head, mi, mt = fit(img_train, epoch_rows, a, dev)
    cap_val = slab(cache[vkey])
    real = retrieval(head, mi, mt, img_val, cap_val, val_owner, dev)
    print(f"{a.arm}: {real}", flush=True)

    n_txt_params = sum(p.numel() for p in head.txt.parameters())
    result = {
        "arm": a.arm,
        "config": {
            "captions_per_image": a.captions_per_image,
            "text_layers": a.text_layers,
            "pooling": a.pooling,
            "img_head": a.img_head,
            "epochs": a.epochs,
            "text_model": a.text_model,
        },
        "val_only": real,
        "shipped_text_head_mb": round(n_txt_params * 2 / 1e6, 2),
        "n_images": len(kept_files),
        "n_train_captions": len(train_texts),
    }
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {a.out}", flush=True)

    if a.save_head:
        torch.save({
            "img_state": head.img.state_dict(),
            "txt": {k: v.cpu() for k, v in head.txt.state_dict().items()},
            "mu_img": mi.cpu(), "mu_txt": mt.cpu(),
            "meta": {**result["config"], "proj_dim": PROJ_DIM, "image_layer": 52,
                     "val_only": real},
        }, a.save_head)
        print(f"wrote {a.save_head}", flush=True)


if __name__ == "__main__":
    main()
