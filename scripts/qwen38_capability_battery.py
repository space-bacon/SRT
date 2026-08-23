"""Capability battery for a locally-deployable Qwen3.8-27B instrument pack.

Answers the questions a release actually has to answer, not benchmark proxies:

  1. RETRIEVAL   Does a linear head on Qwen3.8-27B beat our banked gemma-4-31B
                 number (Karpathy-protocol i2t R@1 0.416)? This is the axis we
                 compete on: capability per trainable MB on a frozen backbone.
  2. TRANSFER    Does a head fit on the BASE model read the ABLITERATED one
                 unchanged? If yes, one artifact ships for both, which is the
                 product claim. Tonight's paired measurement says the edit is
                 ~18x smaller geometrically than a quantizer swap, so it should.
  3. RECAL       If transfer costs something, does a mean-only recalibration
                 (the 42KB fix) recover it, as it does across runtimes?
  4. INVENTORY   Does the head recover what is IN the scene (80 COCO
                 categories, AUC + per-image R-precision)? Measured against
                 annotations, so no caption prior can reach it.

Controls at every rung: a shuffled-pairs fit (can the procedure manufacture
alignment from noise) and, for retrieval, the random-chance floor.

    python qwen38_capability_battery.py --coco-dir /root/qwen38_coco
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

N_EVAL_IMAGES = 1000       # held out; matches the published procrustes protocol
PROJ_DIM = 1024
TAU = 0.05


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco-dir", type=Path, default=Path("/root/qwen38_coco"))
    p.add_argument("--ann-dir", type=Path, default=Path("/root/annotations"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("/root/qwen38_capability_battery.json"))
    p.add_argument("--save-head", type=Path, default=Path("/root/qwen38_head.pt"))
    p.add_argument("--select-epochs", type=int, default=25,
                   help="shorter fits for the layer-selection pass")
    return p.parse_args()


class Head(torch.nn.Module):
    def __init__(self, d_img: int, d_txt: int, proj: int):
        super().__init__()
        self.img = torch.nn.Linear(d_img, proj)
        self.txt = torch.nn.Linear(d_txt, proj)


def fit(Xi, Xt, d_img, d_txt, args, shuffle=False, device="cuda:0"):
    """Symmetric InfoNCE on centred states; one caption per image per step."""
    g = torch.Generator().manual_seed(args.seed)
    mu_i, mu_t = Xi.mean(0), Xt.mean(0)
    Zi, Zt = (Xi - mu_i).to(device), (Xt - mu_t).to(device)
    if shuffle:
        Zt = Zt[torch.randperm(len(Zt), generator=g)]
    head = Head(d_img, d_txt, PROJ_DIM).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    n = len(Zi)
    for ep in range(args.epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch].to(device)
            if len(idx) < 8:
                continue
            a = F.normalize(head.img(Zi[idx]), dim=-1)
            b = F.normalize(head.txt(Zt[idx]), dim=-1)
            logits = a @ b.T / TAU
            tgt = torch.arange(len(idx), device=device)
            loss = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head, mu_i, mu_t


@torch.no_grad()
def retrieval(head, mu_i, mu_t, img, cap, owner_idx, device="cuda:0"):
    """i2t and t2i over the full eval pool. A hit is any of the image's captions."""
    zi = F.normalize(head.img((img - mu_i).to(device)), dim=-1)
    zt = F.normalize(head.txt((cap - mu_t).to(device)), dim=-1)
    S = zi @ zt.T                                   # [n_img, n_cap]
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


@torch.no_grad()
def inventory(head, mu_i, mu_t, img, cat_states, labels, device="cuda:0"):
    """Per-category detection AUC and per-image R-precision vs COCO annotations."""
    zi = F.normalize(head.img((img - mu_i).to(device)), dim=-1)
    zc = F.normalize(head.txt((cat_states - mu_t).to(device)), dim=-1)
    S = (zi @ zc.T).cpu().numpy()                   # [n_img, 80]
    aucs = []
    for c in range(S.shape[1]):
        y = labels[:, c]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        s = S[:, c]
        order = np.argsort(-s)
        yy = y[order]
        pos, neg = yy.sum(), (1 - yy).sum()
        aucs.append(float((np.cumsum(1 - yy)[yy == 1].sum()) / (pos * neg)))
    rp = []
    for i in range(len(S)):
        r = int(labels[i].sum())
        if r == 0:
            continue
        top = np.argsort(-S[i])[:r]
        rp.append(labels[i][top].sum() / r)
    chance = labels.mean()
    return {"mean_category_auc": round(float(np.mean(aucs)), 4),
            "n_categories": len(aucs),
            "per_image_r_precision": round(float(np.mean(rp)), 4),
            "chance_r_precision": round(float(chance), 4)}


def main() -> None:
    args = parse_args()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    I = torch.load(args.coco_dir / "images.pt", map_location="cpu", weights_only=True)
    C = torch.load(args.coco_dir / "captions.pt", map_location="cpu", weights_only=True)
    files = I["files"]
    f_idx = {f: i for i, f in enumerate(files)}
    owner_all = np.array([f_idx[o] for o in C["owner"]])

    n_img = len(files)
    n_train = n_img - N_EVAL_IMAGES
    train_img = np.arange(n_train)
    eval_img = np.arange(n_train, n_img)
    tr_cap = np.where(owner_all < n_train)[0]
    ev_cap = np.where(owner_all >= n_train)[0]
    # one caption per training image keeps the InfoNCE batch free of duplicates
    first_cap = {}
    for j in tr_cap:
        first_cap.setdefault(int(owner_all[j]), int(j))
    tr_pairs = np.array([first_cap[i] for i in train_img if i in first_cap])
    tr_imgs = np.array([i for i in train_img if i in first_cap])
    print(f"{n_img} images, {len(C['texts'])} captions | train {len(tr_imgs)} "
          f"eval {len(eval_img)} images / {len(ev_cap)} captions", flush=True)

    # Read-out layer selection. L48 was the abliteration sweep's peak-rho layer,
    # which maximises the refusal-edit signal, not read-out quality; those are
    # different objectives. Select on held-out retrieval instead, and only among
    # layers whose batched encode matched the sequential one.
    layer_scan = {}
    if I["base"].dim() == 3:
        layers = I.get("layers", list(range(I["base"].shape[1])))
        stable = I.get("batching_stable_layers", layers)
        print(f"\nlayer selection over batching-stable layers {stable}", flush=True)
        sel = argparse.Namespace(**{**vars(args), "epochs": args.select_epochs})
        for k, l in enumerate(layers):
            if l not in stable:
                print(f"  L{l:<3d} skipped (batching-unstable)", flush=True)
                continue
            Xi, Xt = I["base"][tr_imgs, k], C["base"][tr_pairs, k]
            h, mi, mt = fit(Xi, Xt, Xi.shape[1], Xt.shape[1], sel, device=dev)
            r = retrieval(h, mi, mt, I["base"][eval_img, k], C["base"][ev_cap, k],
                          owner_all[ev_cap] - n_train, device=dev)
            layer_scan[l] = r
            print(f"  L{l:<3d} i2t R@1 {r['i2t_r@1']:.4f}  R@5 {r['i2t_r@5']:.4f}  "
                  f"t2i R@1 {r['t2i_r@1']:.4f}", flush=True)
        best = max(layer_scan, key=lambda l: layer_scan[l]["i2t_r@1"])
        bi = layers.index(best)
        print(f"selected read-out layer L{best}\n", flush=True)
        I = {"base": I["base"][:, bi], "abl": I["abl"][:, bi], "layer": best}
        C = {"base": C["base"][:, bi], "abl": C["abl"][:, bi],
             "texts": C["texts"], "owner": C["owner"]}

    # 80 COCO category prompts, encoded by reusing caption states is not possible,
    # so inventory runs only if a category-state file was produced separately.
    cat_path = args.coco_dir / "categories.pt"
    inv_ready = cat_path.exists()

    results: dict = {}
    heads: dict = {}
    for arm in ("base", "abl"):
        Xi, Xt = I[arm][tr_imgs], C[arm][tr_pairs]
        head, mu_i, mu_t = fit(Xi, Xt, I[arm].shape[1], C[arm].shape[1], args, device=dev)
        heads[arm] = (head, mu_i, mu_t)
        r = retrieval(head, mu_i, mu_t, I[arm][eval_img], C[arm][ev_cap],
                      owner_all[ev_cap] - n_train, device=dev)
        results[f"{arm}_head_on_{arm}"] = r
        print(f"{arm:4s} head on {arm:4s}: {r}", flush=True)

    # Transfer: the product claim. One head, both model variants.
    for hsrc, tgt in (("base", "abl"), ("abl", "base")):
        head, mu_i, mu_t = heads[hsrc]
        r = retrieval(head, mu_i, mu_t, I[tgt][eval_img], C[tgt][ev_cap],
                      owner_all[ev_cap] - n_train, device=dev)
        results[f"{hsrc}_head_on_{tgt}"] = r
        print(f"{hsrc:4s} head on {tgt:4s}: {r}", flush=True)
        # 42KB fix: swap in the target's own means, weights untouched.
        mi = I[tgt][tr_imgs].mean(0)
        mt = C[tgt][tr_pairs].mean(0)
        r2 = retrieval(head, mi, mt, I[tgt][eval_img], C[tgt][ev_cap],
                       owner_all[ev_cap] - n_train, device=dev)
        results[f"{hsrc}_head_on_{tgt}_recal"] = r2
        print(f"{hsrc:4s} head on {tgt:4s} + recal: {r2}", flush=True)

    sh, smi, smt = fit(I["base"][tr_imgs], C["base"][tr_pairs],
                       I["base"].shape[1], C["base"].shape[1], args,
                       shuffle=True, device=dev)
    results["shuffled_control"] = retrieval(sh, smi, smt, I["base"][eval_img],
                                            C["base"][ev_cap],
                                            owner_all[ev_cap] - n_train, device=dev)
    print("shuffled control:", results["shuffled_control"], flush=True)
    results["chance_i2t_r@1"] = round(1.0 / len(ev_cap), 6)

    if inv_ready:
        K = torch.load(cat_path, map_location="cpu", weights_only=True)
        for arm in ("base", "abl"):
            head, mu_i, mu_t = heads[arm]
            results[f"inventory_{arm}"] = inventory(
                head, mu_i, mu_t, I[arm][eval_img], K[arm],
                K["labels"][eval_img], device=dev)
            print(f"inventory {arm}: {results[f'inventory_{arm}']}", flush=True)

    payload = {"backbone_base": "Qwen/Qwen3.8-27B",
               "backbone_abl": "JonathanColetti/Qwen3.8-27B-Uncensored",
               "layer": I["layer"], "proj_dim": PROJ_DIM, "tau": TAU,
               "n_train_images": len(tr_imgs), "n_eval_images": len(eval_img),
               "n_eval_captions": len(ev_cap),
               "reference_gemma4_karpathy_i2t_r@1": 0.416,
               "reference_gemma4_matched_n3500_i2t_r@1": 0.334,
               "layer_scan": layer_scan,
               "results": results}
    args.out.write_text(json.dumps(payload, indent=2))
    head, mu_i, mu_t = heads["base"]
    torch.save({"img": head.img.state_dict(), "txt": head.txt.state_dict(),
                "mu_img": mu_i, "mu_txt": mu_t,
                "meta": {k: payload[k] for k in ("layer", "proj_dim", "tau")}},
               args.save_head)
    print(f"\nwrote {args.out} and {args.save_head}", flush=True)


if __name__ == "__main__":
    main()
