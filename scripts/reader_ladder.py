"""Reader ladder: how small can the text side get and still land in the 27B's space?

The browser rung showed a 0.6B text tower reads a 27B's image gallery at 82% of
the matched 27B text tower (i2t R@1 0.356 vs 0.433). This runs the same protocol
with progressively smaller readers, down to one with no pretrained weights:

  minilm       sentence-transformers/all-MiniLM-L6-v2, 22.7M params, 384-d
  bge-small    BAAI/bge-small-en-v1.5, 33M params, 384-d
  embed-table  mean of Qwen3-0.6B's input-embedding rows, no transformer layers
  hash-bow     hashed uni+bigram counts, 4,096 features, no pretrained weights
  hash-bow-1k  the same at 1,024 features

Protocol is byte-for-byte the browser_text_tower.py one: Qwen3.8-27B image
states at L52, 5,000 COCO val2017 images, first caption of the first 4,000
images to fit, all captions of the last 1,000 to score, two-sided linear head to
1,024-d trained with symmetric InfoNCE at tau 0.05 for 60 epochs, and a
shuffled-pairing control fitted the same way. Median rank is reported beside R@k
because R@1 says nothing about an arm near zero.

    .venv-tools/bin/python scripts/reader_ladder.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent.parent
PROJ_DIM = 1024
TAU = 0.05
N_EVAL_IMAGES = 1000
REFERENCE = HERE / "artifacts/nla/q4/browser_text_tower.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-states", type=Path, default=None)
    p.add_argument("--image-layer-index", type=int, default=5)
    p.add_argument("--ann", type=Path, default=HERE / "artifacts/local/coco_ann/captions_val2017.json")
    p.add_argument("--readers", nargs="*",
                   default=["minilm", "bge-small", "embed-table", "hash-bow", "hash-bow-1k"])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--fit-batch", type=int, default=256)
    p.add_argument("--seeds", type=int, nargs="*", default=[0],
                   help="head init + batch order seeds; the split is fixed by file order")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=HERE / "artifacts/nla/q4/reader_ladder.json")
    return p.parse_args()


class Head(torch.nn.Module):
    def __init__(self, d_img, d_txt, proj):
        super().__init__()
        self.img = torch.nn.Linear(d_img, proj)
        self.txt = torch.nn.Linear(d_txt, proj)


def fit(Xi, Xt, args, shuffle=False):
    dev = args.device
    g = torch.Generator().manual_seed(args.seed)
    mu_i, mu_t = Xi.mean(0), Xt.mean(0)
    Zi, Zt = (Xi - mu_i).to(dev), (Xt - mu_t).to(dev)
    if shuffle:
        Zt = Zt[torch.randperm(len(Zt), generator=g).to(dev)]
    head = Head(Xi.shape[1], Xt.shape[1], PROJ_DIM).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    n = len(Zi)
    for _ in range(args.epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, args.fit_batch):
            idx = perm[i:i + args.fit_batch].to(dev)
            if len(idx) < 8:
                continue
            a = F.normalize(head.img(Zi[idx]), dim=-1)
            b = F.normalize(head.txt(Zt[idx]), dim=-1)
            logits = a @ b.T / TAU
            tgt = torch.arange(len(idx), device=dev)
            loss = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))
            opt.zero_grad(); loss.backward(); opt.step()
    return head, mu_i, mu_t


@torch.no_grad()
def retrieval(head, mu_i, mu_t, img, cap, owner_idx, dev):
    zi = F.normalize(head.img((img - mu_i).to(dev)), dim=-1)
    zt = F.normalize(head.txt((cap - mu_t).to(dev)), dim=-1)
    S = (zi @ zt.T).cpu()
    owner = torch.as_tensor(owner_idx)
    out = {}
    rank = S.argsort(dim=1, descending=True)
    hit = owner[rank] == torch.arange(len(zi))[:, None]
    for k in (1, 5, 10):
        out[f"i2t_r@{k}"] = round(hit[:, :k].any(1).float().mean().item(), 4)
    out["i2t_median_rank"] = int(hit.float().argmax(1).median().item()) + 1
    rank_t = S.T.argsort(dim=1, descending=True)
    correct = owner[:, None] == rank_t
    for k in (1, 5, 10):
        out[f"t2i_r@{k}"] = round(correct[:, :k].any(1).float().mean().item(), 4)
    out["t2i_median_rank"] = int(correct.float().argmax(1).median().item()) + 1
    return out


# ---------------------------------------------------------------- readers
# Each returns (features [n, d] float32, description dict with a payload estimate).

def read_sentence_transformer(name, texts, dev, prefix=""):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(name, device=dev if dev != "mps" else "mps")
    X = st.encode([prefix + t for t in texts], batch_size=256, convert_to_numpy=True,
                  show_progress_bar=False, normalize_embeddings=False).astype(np.float32)
    n_params = sum(p.numel() for p in st.parameters())
    return torch.from_numpy(X), {"reader": name, "params": n_params,
                                 "payload_mb_fp16": round(n_params * 2 / 1e6, 1)}


ENCODERS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "bge-small": ("BAAI/bge-small-en-v1.5", ""),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", ""),
    "gte-base": ("thenlper/gte-base", ""),
    # e5 is trained with an instruction prefix; captions are queries here.
    "e5-base": ("intfloat/e5-base-v2", "query: "),
}


def read_embed_table(texts, dev, model_name="Qwen/Qwen3-0.6B"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    E = model.get_input_embeddings().weight.detach()
    del model
    rows = []
    used = set()
    for t in texts:
        ids = tok(t, truncation=True, max_length=64, add_special_tokens=False).input_ids
        used.update(ids)
        rows.append(E[ids].mean(0))
    X = torch.stack(rows).float()
    full = E.shape[0] * E.shape[1]
    pruned = len(used) * E.shape[1]
    return X, {"reader": f"{model_name} input-embedding mean", "params": full,
               "payload_mb_fp16": round(full * 2 / 1e6, 1),
               "payload_mb_fp16_pruned_to_seen_vocab": round(pruned * 2 / 1e6, 1),
               "vocab_seen": len(used)}


def read_hash_bow(texts, n_features):
    from sklearn.feature_extraction.text import HashingVectorizer
    hv = HashingVectorizer(n_features=n_features, ngram_range=(1, 2), alternate_sign=False,
                           norm="l2", lowercase=True)
    X = hv.transform(texts).toarray().astype(np.float32)
    return torch.from_numpy(X), {"reader": f"hashed uni+bigram counts, {n_features} features",
                                 "params": 0, "payload_mb_fp16": 0.0}


def main() -> None:
    args = parse_args()
    dev = args.device
    if args.image_states is None:
        from huggingface_hub import hf_hub_download
        args.image_states = Path(hf_hub_download(
            "RiverRider/srt-qwen38-coco-states", "qwen38_coco_ls/images.pt", repo_type="dataset"))
    I = torch.load(args.image_states, map_location="cpu", weights_only=True)
    files = I["files"]
    img = I["base"][:, args.image_layer_index].float()
    image_layer = I["layers"][args.image_layer_index]
    print(f"image states {tuple(img.shape)} L{image_layer}", flush=True)

    ann = json.loads(args.ann.read_text())
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
    n_img = len(files)
    n_train = n_img - N_EVAL_IMAGES
    first = {}
    for j in np.where(owner < n_train)[0]:
        first.setdefault(int(owner[j]), int(j))
    tr_imgs = np.array([i for i in range(n_train) if i in first])
    tr_pairs = np.array([first[i] for i in tr_imgs])
    ev_img = np.arange(n_train, n_img)
    ev_cap = np.where(owner >= n_train)[0]
    print(f"{len(texts)} captions, {len(tr_imgs)} train pairs, {len(ev_img)} eval images, "
          f"{len(ev_cap)} eval captions", flush=True)

    ref = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else None
    results = {
        "protocol": {"image_states": str(args.image_states), "image_layer": image_layer,
                     "n_train": int(len(tr_imgs)), "n_eval_images": int(len(ev_img)),
                     "n_eval_captions": int(len(ev_cap)), "proj_dim": PROJ_DIM, "tau": TAU,
                     "epochs": args.epochs, "seeds": args.seeds, "device": dev,
                     "chance_i2t_r@1": round(1.0 / len(ev_cap), 6)},
        "reference_0.6B": None if ref is None else {
            "reader": ref["text_tower"], "text_layer": ref["text_layer"], "params": 600_000_000,
            "bridge": ref["bridge"], "shuffled_control": ref["shuffled_control"]},
        "rungs": {},
    }

    for name in args.readers:
        t0 = time.time()
        if name in ENCODERS:
            X, desc = read_sentence_transformer(ENCODERS[name][0], texts, dev, ENCODERS[name][1])
        elif name == "embed-table":
            X, desc = read_embed_table(texts, dev)
        elif name == "hash-bow":
            X, desc = read_hash_bow(texts, 4096)
        elif name == "hash-bow-1k":
            X, desc = read_hash_bow(texts, 1024)
        else:
            raise SystemExit(f"unknown reader {name}")
        d_txt = X.shape[1]
        per_seed = []
        for seed in args.seeds:
            args.seed = seed
            head, mi, mt = fit(img[tr_imgs], X[tr_pairs], args)
            real = retrieval(head, mi, mt, img[ev_img], X[ev_cap], owner[ev_cap] - n_train, dev)
            sh, smi, smt = fit(img[tr_imgs], X[tr_pairs], args, shuffle=True)
            shuf = retrieval(sh, smi, smt, img[ev_img], X[ev_cap], owner[ev_cap] - n_train, dev)
            per_seed.append({"seed": seed, "bridge": real, "shuffled_control": shuf})
        keys = per_seed[0]["bridge"].keys()
        agg = lambda arm: {k: round(float(np.mean([s[arm][k] for s in per_seed])), 4) for k in keys}
        rng_ = lambda arm, k: [min(s[arm][k] for s in per_seed), max(s[arm][k] for s in per_seed)]
        real, shuf = agg("bridge"), agg("shuffled_control")
        desc.update({
            "d_txt": int(d_txt),
            "text_head_mb_fp16": round(d_txt * PROJ_DIM * 2 / 1e6, 2),
            "text_head_mb_int8": round(d_txt * PROJ_DIM / 1e6, 2),
            "bridge": real, "shuffled_control": shuf,
            "bridge_i2t_r@1_range": rng_("bridge", "i2t_r@1"),
            "per_seed": per_seed,
            "seconds": round(time.time() - t0, 1),
        })
        results["rungs"][name] = desc
        lo, hi = desc["bridge_i2t_r@1_range"]
        print(f"{name:12s} d={d_txt:5d}  i2t R@1 {real['i2t_r@1']:.4f} [{lo:.3f},{hi:.3f}] "
              f"R@5 {real['i2t_r@5']:.4f} median {real['i2t_median_rank']:>5.1f} | "
              f"t2i R@1 {real['t2i_r@1']:.4f} | shuffled R@1 {shuf['i2t_r@1']:.4f} "
              f"median {shuf['i2t_median_rank']:>6.1f} | {desc['seconds']}s", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))

    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
