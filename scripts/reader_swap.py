"""Reader swap: put a sentence encoder behind the shipped browser gallery.

The reader ladder fitted BOTH sides of the head on 4,000 val pairs and found
sentence encoders read the 27B's image space better than the 27B's own text
tower. That is not yet a drop-in: the deployed gallery (`gallery_123k_v3.srtidx`,
123,287 COCO images) was projected by a head fitted against Qwen3-0.6B text, and
a phone that already holds that gallery should not have to re-download it.

So here the image side is FROZEN to the shipped gallery rows and only a text
head is fitted: sentence-encoder embedding -> centre -> linear -> 1,024-d,
symmetric InfoNCE against the fixed gallery vectors, trained on COCO train2017
only (118,287 images, one of five captions sampled per epoch, the v3 recipe),
scored on the card's own replay set: 5,001 val2017 captions against all 123,287
gallery rows, plus the 1,000-image sub-pool those captions belong to. The 0.6B
v3 text tower on the identical set is t2i R@1 0.1092, median rank 36
(`replay_123k.json`, `srt-browser-head-118k` card).

If the sentence encoder matches or beats that with the gallery untouched, the
swap is a 2 MB head and a 45 to 220 MB encoder in place of a 382 MB 0.6B, and
nothing else in the deployment moves.

    .venv-tools/bin/python scripts/reader_swap.py --readers bge-small e5-base
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
from reader_ladder import ENCODERS, read_sentence_transformer  # noqa: E402

PROJ_DIM = 1024
TAU = 0.05
MAGIC_F16 = b"SRTIDX01"
MAGIC_I8 = b"SRTIDX02"
REFERENCE_06B = {"t2i_r@1": 0.1092, "t2i_r@5": 0.2442, "t2i_r@10": 0.3307,
                 "t2i_median_rank": 36, "source": "replay_123k.json pytorch_t2i; card row 'PyTorch fp16 (reference)'"}
# sentence-transformers pooling module of each checkpoint (1_Pooling/config.json);
# the Rust reader must pool the same way or mu_txt describes a different quantity.
POOLING = {"bge-small": "cls", "minilm": "mean", "e5-base": "mean", "gte-base": "mean", "mpnet": "mean"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gallery", type=Path, default=HERE / "artifacts/local/browser/gallery_123k_v3.srtidx")
    p.add_argument("--replay", type=Path, default=HERE / "artifacts/local/browser/replay_123k.json")
    p.add_argument("--train-ann", type=Path, default=HERE / "artifacts/local/coco_ann/captions_train2017.json")
    p.add_argument("--readers", nargs="*", default=["bge-small", "e5-base"])
    p.add_argument("--seeds", type=int, nargs="*", default=[0])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--fit-batch", type=int, default=1024)
    p.add_argument("--cache", type=Path, default=HERE / "artifacts/local/reader_swap")
    p.add_argument("--save-heads", type=Path, default=HERE / "artifacts/local/reader_swap/heads")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=HERE / "artifacts/nla/q4/reader_swap_123k.json")
    return p.parse_args()


# ---------------------------------------------------------------- gallery

def read_srtidx(path: Path):
    b = path.read_bytes()
    magic, (dim, n) = b[:8], struct.unpack("<II", b[8:16])
    off = 16
    if magic == MAGIC_I8:
        scale = np.frombuffer(b[off:off + 4 * n], dtype="<f4"); off += 4 * n
        q = np.frombuffer(b[off:off + n * dim], dtype=np.int8).reshape(n, dim); off += n * dim
        Z = q.astype(np.float32) * scale[:, None]
    elif magic == MAGIC_F16:
        Z = np.frombuffer(b[off:off + 2 * n * dim], dtype="<f2").reshape(n, dim).astype(np.float32)
        off += 2 * n * dim
    else:
        raise SystemExit(f"bad magic {magic!r}")
    keys = []
    for _ in range(n):
        (L,) = struct.unpack("<I", b[off:off + 4]); off += 4
        keys.append(b[off:off + L].decode()); off += L
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    return torch.from_numpy(Z), keys


def load_train_captions(ann: Path, keys: list[str]):
    d = json.loads(ann.read_text())
    id2file = {im["id"]: im["file_name"] for im in d["images"]}
    by_file: dict[str, list[str]] = {}
    for a in sorted(d["annotations"], key=lambda x: x["id"]):
        by_file.setdefault(id2file[a["image_id"]], []).append(a["caption"].strip())
    rows, groups = [], []
    for i, k in enumerate(keys):
        if k.startswith("train2017/"):
            caps = by_file.get(k.split("/", 1)[1])
            if caps:
                rows.append(i)
                groups.append(caps)
    return np.array(rows), groups


# ---------------------------------------------------------------- fit / score

class TextHead(torch.nn.Module):
    def __init__(self, d_txt, proj):
        super().__init__()
        self.txt = torch.nn.Linear(d_txt, proj)


def fit(Zg_rows, cap_groups_feat, offsets, args, seed, shuffle=False):
    """Zg_rows: fixed unit gallery vectors for the training images [n, 1024].
    cap_groups_feat: all training caption features [m, d]; offsets[i] = slice of image i's captions."""
    dev = args.device
    g = torch.Generator().manual_seed(seed)
    mu_t = cap_groups_feat.mean(0)
    Zi = Zg_rows.to(dev)
    Xt = (cap_groups_feat - mu_t).to(dev)
    n = len(Zi)
    starts = torch.tensor([o[0] for o in offsets]); lens = torch.tensor([o[1] - o[0] for o in offsets])
    img_of = torch.arange(n)
    if shuffle:
        img_of = img_of[torch.randperm(n, generator=g)]
    head = TextHead(Xt.shape[1], PROJ_DIM).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    for ep in range(args.epochs):
        pick = starts + (torch.rand(n, generator=g) * lens).long()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, args.fit_batch):
            idx = perm[i:i + args.fit_batch]
            if len(idx) < 8:
                continue
            a = Zi[img_of[idx].to(dev)]
            b = F.normalize(head.txt(Xt[pick[idx].to(dev)]), dim=-1)
            logits = a @ b.T / TAU
            tgt = torch.arange(len(idx), device=dev)
            loss = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))
            opt.zero_grad(); loss.backward(); opt.step()
        if not shuffle and (ep % 5 == 0 or ep == args.epochs - 1):
            print(f"    epoch {ep + 1}/{args.epochs} loss {loss.item():.4f}", flush=True)
    return head, mu_t


@torch.no_grad()
def project_text(head, mu_t, X, dev):
    return F.normalize(head.txt((X - mu_t).to(dev)), dim=-1).cpu()


@torch.no_grad()
def score_t2i(zt, Zg, gold):
    """Rank of each caption's gold gallery row over Zg."""
    S = zt @ Zg.T
    gold_s = S[torch.arange(len(gold)), gold]
    rank = (S > gold_s[:, None]).sum(1) + 1
    out = {f"t2i_r@{k}": round((rank <= k).float().mean().item(), 4) for k in (1, 5, 10)}
    out["t2i_median_rank"] = int(rank.float().median().item())
    return out


@torch.no_grad()
def score_pool(zt, Zg, gold):
    """Both directions inside the sub-pool the eval captions belong to."""
    pool = torch.unique(gold)
    remap = {int(r): j for j, r in enumerate(pool)}
    owner = torch.tensor([remap[int(g)] for g in gold])
    Zp = Zg[pool]
    S = Zp @ zt.T                                   # [imgs, caps]
    out = {}
    rank = S.argsort(dim=1, descending=True)
    hit = owner[rank] == torch.arange(len(Zp))[:, None]
    for k in (1, 5, 10):
        out[f"i2t_r@{k}"] = round(hit[:, :k].any(1).float().mean().item(), 4)
    out["i2t_median_rank"] = int(hit.float().argmax(1).median().item()) + 1
    t = score_t2i(zt, Zp, owner)
    out.update({k: v for k, v in t.items()})
    out["pool_images"] = len(Zp)
    return out


def save_head(head, mu_t, meta, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    W = head.txt.weight.detach().cpu(); b = head.txt.bias.detach().cpu()
    torch.save({"txt": {"weight": W, "bias": b}, "mu_txt": mu_t.cpu(), "meta": meta}, stem.with_suffix(".pt"))
    from safetensors.torch import save_file
    save_file({"txt.weight": W.half().contiguous(), "txt.bias": b.half().contiguous(),
               "mu_txt": mu_t.cpu().half().contiguous()},
              str(stem.with_suffix(".safetensors")), metadata={k: str(v) for k, v in meta.items()})


def mean_range(per_seed, key):
    v = [r[key] for r in per_seed]
    return round(float(np.mean(v)), 4), [min(v), max(v)]


# ---------------------------------------------------------------- main

def main() -> None:
    args = parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    dev = args.device

    Zg, keys = read_srtidx(args.gallery)
    print(f"gallery {tuple(Zg.shape)} from {args.gallery.name}", flush=True)
    replay = json.loads(args.replay.read_text())
    ev_texts = [c["text"] for c in replay["captions"]]
    ev_gold = torch.tensor([c["gold"] for c in replay["captions"]])
    assert all(keys[g].startswith("val2017/") for g in ev_gold.tolist())

    tr_rows, tr_groups = load_train_captions(args.train_ann, keys)
    flat, offsets = [], []
    for caps in tr_groups:
        offsets.append((len(flat), len(flat) + len(caps)))
        flat.extend(caps)
    print(f"{len(tr_rows)} train images, {len(flat)} train captions, "
          f"{len(ev_texts)} eval captions over {len(torch.unique(ev_gold))} val images, "
          f"gallery {len(keys)}", flush=True)

    results = {
        "protocol": {
            "gallery": args.gallery.name, "gallery_rows": len(keys), "image_side": "frozen to shipped v3 gallery",
            "train": "COCO train2017, one of five captions sampled per epoch", "n_train_images": int(len(tr_rows)),
            "n_train_captions": len(flat), "eval": "replay_123k.json: 5,001 val2017 captions, gold = gallery row",
            "n_eval_captions": len(ev_texts), "proj_dim": PROJ_DIM, "tau": TAU, "epochs": args.epochs,
            "lr": args.lr, "fit_batch": args.fit_batch, "seeds": args.seeds, "device": dev,
            "chance_t2i_r@1_full": round(1 / len(keys), 6),
        },
        "reference_0.6B_v3_text_tower": REFERENCE_06B,
        "rungs": {},
    }

    for name in args.readers:
        model_id, prefix = ENCODERS[name]
        t0 = time.time()
        ctr, cev = args.cache / f"{name}_train.npy", args.cache / f"{name}_eval.npy"
        if ctr.exists() and cev.exists():
            Xtr = torch.from_numpy(np.load(ctr).astype(np.float32))
            Xev = torch.from_numpy(np.load(cev).astype(np.float32))
            desc = json.loads((args.cache / f"{name}_desc.json").read_text())
            print(f"{name}: cache hit {tuple(Xtr.shape)}", flush=True)
        else:
            print(f"{name}: encoding {len(flat)} + {len(ev_texts)} captions", flush=True)
            X, desc = read_sentence_transformer(model_id, flat + ev_texts, dev, prefix)
            Xtr, Xev = X[:len(flat)], X[len(flat):]
            np.save(ctr, Xtr.numpy().astype(np.float16)); np.save(cev, Xev.numpy().astype(np.float16))
            (args.cache / f"{name}_desc.json").write_text(json.dumps(desc))
            print(f"  encoded in {time.time() - t0:.0f}s", flush=True)

        per_seed, per_seed_sh, per_seed_pool = [], [], []
        for seed in args.seeds:
            head, mu_t = fit(Zg[tr_rows], Xtr, offsets, args, seed)
            zt = project_text(head, mu_t, Xev, dev)
            full = score_t2i(zt, Zg, ev_gold)
            pool = score_pool(zt, Zg, ev_gold)
            sh, smu = fit(Zg[tr_rows], Xtr, offsets, args, seed, shuffle=True)
            shuf = score_t2i(project_text(sh, smu, Xev, dev), Zg, ev_gold)
            per_seed.append(full); per_seed_pool.append(pool); per_seed_sh.append(shuf)
            print(f"  seed {seed}: full-gallery t2i R@1 {full['t2i_r@1']:.4f} R@5 {full['t2i_r@5']:.4f} "
                  f"R@10 {full['t2i_r@10']:.4f} median {full['t2i_median_rank']} | "
                  f"pool-{pool['pool_images']} i2t R@1 {pool['i2t_r@1']:.4f} t2i R@1 {pool['t2i_r@1']:.4f} | "
                  f"shuffled R@1 {shuf['t2i_r@1']:.4f} median {shuf['t2i_median_rank']}", flush=True)
            if seed == args.seeds[0]:
                meta = {"reader": model_id, "prefix": prefix, "pooling": POOLING[name], "normalize": True,
                        "max_len": 64, "gallery": args.gallery.name,
                        "proj_dim": PROJ_DIM, "seed": seed, "epochs": args.epochs, **full}
                save_head(head, mu_t, meta, args.save_heads / f"text_head_{name}_v3gallery")

        r1, r1_range = mean_range(per_seed, "t2i_r@1")
        rung = {
            **desc, "d_txt": int(Xtr.shape[1]),
            "text_head_mb_fp16": round((Xtr.shape[1] + 1) * PROJ_DIM * 2 / 1e6, 2),
            "full_gallery": {k: (round(float(np.mean([r[k] for r in per_seed])), 4)) for k in per_seed[0]},
            "full_gallery_t2i_r@1_range": r1_range,
            "pool": {k: (round(float(np.mean([r[k] for r in per_seed_pool])), 4)) for k in per_seed_pool[0]},
            "shuffled_control": {k: (round(float(np.mean([r[k] for r in per_seed_sh])), 4)) for k in per_seed_sh[0]},
            "per_seed": [{"seed": s, "full_gallery": a, "pool": b, "shuffled": c}
                         for s, a, b, c in zip(args.seeds, per_seed, per_seed_pool, per_seed_sh)],
            "vs_0.6B_t2i_r@1": round(r1 - REFERENCE_06B["t2i_r@1"], 4),
            "seconds": round(time.time() - t0, 1),
        }
        results["rungs"][name] = rung
        print(f"{name:10s} full-gallery t2i R@1 {r1:.4f} {r1_range} (0.6B v3: {REFERENCE_06B['t2i_r@1']}) "
              f"median {rung['full_gallery']['t2i_median_rank']:.0f} | {rung['seconds']}s\n", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
