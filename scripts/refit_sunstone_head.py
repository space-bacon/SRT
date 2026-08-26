#!/usr/bin/env python
"""Refit the sunstone head, varying only how many captions it is shown.

The shipped head was fitted on ONE caption per image (`gemma4_encode_pairs.py`
writes `caption: caps[0]`). Two things follow from that and this script exists
to separate them:

  1. Any evaluation that scores an image's FIRST caption against a train2017
     gallery is scoring a pair the head was trained to align. That is why the
     shipped eval's "human ceiling" read median 8 while an unseen caption of the
     same photograph read 45.
  2. A head only ever asked to match one description may discard the rest of the
     scene, which would explain why the map's reader trails a reader working in
     a five-caption space even though both spaces align unseen captions about
     equally well.

So: same backbone states, same images, same split, same objective. The only
thing that moves is `--mode`.

    cap0   first caption only, reproducing the shipped recipe
    all5   one of the image's five captions resampled every epoch

This also holds the 5,000 evaluation images out of HEAD training, not just
reader training, so the resulting gold-caption arm is a real human reference
rather than a memorised pair.

    python scripts/refit_sunstone_head.py --mode all5 --out /root/head_all5.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--img", default="/root/full_img_vecs.npy")
    p.add_argument("--caps-dir", default="/root/caps_l47")
    p.add_argument("--mode", choices=["cap0", "all5"], required=True)
    p.add_argument("--holdout", type=int, default=5000)
    p.add_argument("--proj-dim", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--gallery-out", default=None,
                   help="also write the projected, L2-normalised image gallery")
    return p.parse_args()


def load_caps(d: Path, n_img: int, max_caps: int = 5):
    chunks = sorted(d.glob("chunk_*.npy"))
    if not chunks:
        raise SystemExit(f"no chunks in {d}")
    parts, counts = [], []
    for c in chunks:
        parts.append(np.load(c))
        cf = c.parent / c.name.replace("chunk_", "counts_")
        counts.append(np.load(cf) if cf.exists()
                      else np.full(len(parts[-1]), max_caps, dtype=np.int16))
    caps = np.concatenate(parts)
    cnt = np.concatenate(counts)
    if len(caps) != n_img:
        raise SystemExit(f"caption rows {len(caps)} != image rows {n_img}")
    return caps, cnt


def main():
    a = parse()
    dev = "cuda"
    torch.manual_seed(a.seed)

    img = np.load(a.img)
    n_img, d_in = img.shape
    caps, cnt = load_caps(Path(a.caps_dir), n_img)
    print(f"images {img.shape}  captions {caps.shape}  "
          f"mean caps/img {cnt.mean():.2f}", flush=True)

    # Same permutation the readers use, so the holdout is shared end to end.
    order = np.random.default_rng(0).permutation(n_img)
    test_idx, train_idx = order[: a.holdout], order[a.holdout:]
    print(f"{len(train_idx)} train / {len(test_idx)} held out of head training",
          flush=True)

    img_t = torch.from_numpy(img).float()
    caps_t = torch.from_numpy(caps)          # fp16, (n, 5, d_in)
    cnt_t = torch.from_numpy(cnt.astype(np.int64))
    tr = torch.from_numpy(train_idx.copy())
    te = torch.from_numpy(test_idx.copy())

    # Centering means come from the TRAINING split only. Chunked: casting the
    # whole caption tensor to fp32 at once would need ~12 GB of scratch.
    mu_img = img_t[tr].mean(0)
    acc = torch.zeros(d_in, dtype=torch.float64)
    seen = 0
    for i in range(0, len(tr), 8192):
        sl = tr[i:i + 8192]
        block = caps_t[sl].float()
        m = (torch.arange(caps_t.shape[1])[None, :] < cnt_t[sl][:, None])
        acc += (block * m[..., None]).sum((0, 1)).double()
        seen += int(m.sum())
    mu_txt = (acc / max(seen, 1)).float()
    print(f"centering on {seen} training captions", flush=True)

    img_head = torch.nn.Linear(d_in, a.proj_dim).to(dev)
    txt_head = torch.nn.Linear(d_in, a.proj_dim).to(dev)
    opt = torch.optim.AdamW(
        list(img_head.parameters()) + list(txt_head.parameters()), lr=a.lr)
    steps_per_epoch = (len(tr) + a.batch - 1) // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch)

    mu_img_d, mu_txt_d = mu_img.to(dev), mu_txt.to(dev)
    g = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    for ep in range(a.epochs):
        perm = tr[torch.randperm(len(tr), generator=g)]
        tot = nb = 0
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            if len(idx) < 8:
                continue
            if a.mode == "cap0":
                pick = torch.zeros(len(idx), dtype=torch.long)
            else:
                # Resample which of the five captions represents the image, so
                # the head has to match any description rather than one.
                pick = (torch.rand(len(idx), generator=g) * cnt_t[idx]).long()
            t = caps_t[idx, pick].float().to(dev, non_blocking=True)
            v = img_t[idx].to(dev, non_blocking=True)
            zi = F.normalize(img_head(v - mu_img_d), dim=-1)
            zt = F.normalize(txt_head(t - mu_txt_d), dim=-1)
            logits = zi @ zt.T / a.tau
            tgt = torch.arange(len(idx), device=dev)
            loss = 0.5 * (F.cross_entropy(logits, tgt)
                          + F.cross_entropy(logits.T, tgt))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item(); nb += 1
        print(f"epoch {ep + 1}/{a.epochs}  loss {tot / max(nb, 1):.4f}  "
              f"{(time.time() - t0) / 60:.1f}m", flush=True)

    # Held-out retrieval, scored the way the deployment scores: every held-out
    # caption against the full gallery of all 118,287 images.
    @torch.no_grad()
    def project(head, x, mu):
        out = []
        for i in range(0, len(x), 4096):
            out.append(F.normalize(
                head(x[i:i + 4096].float().to(dev) - mu), dim=-1).cpu())
        return torch.cat(out)

    img_head.eval(); txt_head.eval()
    gal = project(img_head, img_t, mu_img_d)
    res = {}
    for slot, name in ((0, "caption_0"), (1, "caption_1")):
        q = project(txt_head, caps_t[te, slot], mu_txt_d)
        s = q.to(dev) @ gal.T.to(dev)
        gold = s[torch.arange(len(te)), te.to(dev)]
        rank = (s > gold[:, None]).sum(1).cpu().numpy() + 1
        res[name] = {"r@1": round(float((rank <= 1).mean()), 4),
                     "r@10": round(float((rank <= 10).mean()), 4),
                     "median_rank": float(np.median(rank))}
        print(f"held-out {name}: R@1 {res[name]['r@1']:.4f} "
              f"median {res[name]['median_rank']:.0f}", flush=True)
        del s

    torch.save({"img": {k: v.cpu() for k, v in img_head.state_dict().items()},
                "txt": {k: v.cpu() for k, v in txt_head.state_dict().items()},
                "mu_img": mu_img, "mu_txt": mu_txt,
                "meta": {"mode": a.mode, "epochs": a.epochs, "batch": a.batch,
                         "lr": a.lr, "tau": a.tau, "proj_dim": a.proj_dim,
                         "n_train": int(len(tr)), "holdout": int(a.holdout),
                         "heldout_from_head_training": True,
                         "eval": res}}, a.out)
    print(f"wrote {a.out}", flush=True)
    if a.gallery_out:
        np.save(a.gallery_out, gal.numpy().astype(np.float32))
        print(f"wrote {a.gallery_out} {tuple(gal.shape)}", flush=True)
    json.dump(res, open(str(a.out) + ".eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
