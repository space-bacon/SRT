#!/usr/bin/env python
"""Answer for the §8 hardware-row critique: what does the head actually buy?

Runs on a CUDA box (A100/H100 80GB, or 2x48GB with device_map=auto).

Step 1: fresh bf16 CUDA encode of the 5000 calib captions at L47 last token
        (exact convention of gemma4_procrustes_xmodal.py / the calib artifact:
        BOS enforced, right-pad, hidden_states[47], last non-pad position).
Step 2: four-arm comparison of the fresh CUDA states against BOTH
        (a) the stored calib bf16 states (artifact integrity check) and
        (b) an MLX-Q4 states file if provided (--mlx-states, produced on the
            Mac via the sunstone server /encode_texts endpoint).

Image side (the vision tie-in, 2026-08-05): the same protocol for the
1000 eval images that carry captions5 rows. Encode convention is the
Sunstone one (gemma4_procrustes_xmodal.py image_v): chat template with
"Describe this image.", mean of hidden_states[47] over image-token
positions. --out-images downloads COCO val2017 and encodes fresh;
--analyze-images runs the four arms through the head's img branch
against stored calib["img"] (and optional --mlx-img-states from the
Mac), plus the end-task i2t retrieval R@1/R@5 against the cap5 caption
gallery (any-of-5 correct).

Arms (cross-runtime self-retrieval@1 over the full pool, plus R@5):
    A  raw states, as-is               (the never-shipped baseline)
    B  mean-centered raw               (what local_sunstone --validate measures)
    C  head-projected, no re-centering (head's own mu on both sides)
    D  head-projected, per-runtime mu  (full pipeline the row claims)

Usage on the box:
    pip install torch transformers accelerate huggingface_hub numpy pillow
    python encode_ref_cuda.py --out ref_cuda.pt
    python encode_ref_cuda.py --analyze ref_cuda.pt [--mlx-states mlx_q4.npz]
    python encode_ref_cuda.py --out-images ref_img_cuda.pt
    python encode_ref_cuda.py --analyze-images ref_img_cuda.pt \
        [--mlx-img-states mlx_img_q4.npz] [--cap-states ref_cuda.pt]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

CALIB_REPO = "RiverRider/srt-nla-gemma4-artifacts"
CALIB_FILE = "procrustes/encoded_L47_n5000.pt"
HEAD_REPO = "RiverRider/srt-sunstone-linear-head"
HEAD_FILE = "sunstone_linear_head.pt"
BACKBONE = "google/gemma-4-31B-it"
LAYER = 47
MAX_SEQ = 64
BATCH = 16
IMG_N = 1000            # eval images that carry captions5 rows
COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None, help="encode fresh CUDA states to this file")
    p.add_argument("--analyze", default=None, help="ref_cuda.pt to analyze")
    p.add_argument("--mlx-states", default=None,
                   help="npz with arrays 'states' (n,d) from the Mac (optional)")
    p.add_argument("--out-images", default=None,
                   help="encode fresh CUDA image states to this file")
    p.add_argument("--analyze-images", default=None,
                   help="ref_img_cuda.pt to analyze (image four-arm + i2t)")
    p.add_argument("--mlx-img-states", default=None,
                   help="npz with 'states' (n,d) image states from the Mac")
    p.add_argument("--cap-states", default=None,
                   help="ref_cuda.pt caption states for the i2t gallery "
                        "(default: stored calib cap5)")
    p.add_argument("--img-dir", default="val2017",
                   help="local COCO val2017 dir (downloaded if missing)")
    p.add_argument("--head-file", default=HEAD_FILE)
    return p.parse_args()


def load_calib():
    from huggingface_hub import hf_hub_download
    return torch.load(hf_hub_download(CALIB_REPO, CALIB_FILE),
                      map_location="cpu", weights_only=True)


def encode(out_path: str) -> None:
    from transformers import AutoProcessor, AutoModelForImageTextToText

    calib = load_calib()
    texts = [c for caps in calib["captions5"] for c in caps]
    print(f"{len(texts)} captions to encode")

    proc = AutoProcessor.from_pretrained(BACKBONE)
    tok = getattr(proc, "tokenizer", proc)
    model = AutoModelForImageTextToText.from_pretrained(
        BACKBONE, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    lm = model  # full multimodal forward, text-only inputs

    bos = tok.bos_token_id
    states = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i + BATCH]
            ids_list = []
            for t in chunk:
                ids = tok(t, truncation=True, max_length=MAX_SEQ,
                          add_special_tokens=True).input_ids
                if bos is not None and ids[0] != bos:
                    ids = [bos] + ids[: MAX_SEQ - 1]
                ids_list.append(ids)
            T = max(len(x) for x in ids_list)
            pad = tok.pad_token_id if tok.pad_token_id is not None else bos
            input_ids = torch.full((len(ids_list), T), pad, dtype=torch.long)
            attn = torch.zeros((len(ids_list), T), dtype=torch.long)
            for j, ids in enumerate(ids_list):
                input_ids[j, : len(ids)] = torch.tensor(ids)
                attn[j, : len(ids)] = 1
            out = lm(input_ids=input_ids.cuda(), attention_mask=attn.cuda(),
                     output_hidden_states=True, use_cache=False)
            h = out.hidden_states[LAYER]
            last = attn.sum(-1) - 1
            rows = torch.arange(h.size(0))
            states.append(h[rows.cuda(), last.cuda()].float().cpu())
            if (i // BATCH) % 20 == 0:
                print(f"  {i}/{len(texts)}")
    X = torch.cat(states)
    torch.save({"states": X, "layer": LAYER, "backbone": BACKBONE}, out_path)
    print(f"saved {tuple(X.shape)} -> {out_path}")


def _ensure_images(img_dir: str, files: list[str]) -> list[str]:
    """Return local paths for the needed val2017 files, downloading the
    public COCO zip once if absent."""
    import os
    import subprocess

    names = [f.rsplit("/", 1)[-1] for f in files]
    if not all(os.path.exists(os.path.join(img_dir, n)) for n in names):
        print(f"downloading {COCO_VAL_URL} ...")
        subprocess.run(["curl", "-sLO", COCO_VAL_URL], check=True)
        subprocess.run(["unzip", "-qo", "val2017.zip"], check=True)
    missing = [n for n in names if not os.path.exists(os.path.join(img_dir, n))]
    if missing:
        raise SystemExit(f"{len(missing)} images missing after download, "
                         f"e.g. {missing[:3]}")
    return [os.path.join(img_dir, n) for n in names]


def encode_images(out_path: str, img_dir: str) -> None:
    """Fresh CUDA bf16 image states: mean of hidden_states[LAYER] over
    image-token positions (gemma4_procrustes_xmodal.py image_v convention)."""
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    calib = load_calib()
    paths = _ensure_images(img_dir, calib["files"][:IMG_N])
    print(f"{len(paths)} images to encode")

    proc = AutoProcessor.from_pretrained(BACKBONE)
    model = AutoModelForImageTextToText.from_pretrained(
        BACKBONE, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    img_tok = getattr(model.config, "image_token_id", None)
    assert img_tok is not None, "backbone config lacks image_token_id"

    states = []
    with torch.no_grad():
        for i, path in enumerate(paths):
            img = Image.open(path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Describe this image."},
            ]}]
            enc = proc.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to("cuda")
            out = model(**enc, output_hidden_states=True, use_cache=False)
            mask = enc["input_ids"][0] == img_tok
            states.append(
                out.hidden_states[LAYER][0][mask].float().mean(0).cpu())
            if i % 50 == 0:
                print(f"  {i}/{len(paths)}")
    X = torch.stack(states)
    torch.save({"states": X, "layer": LAYER, "backbone": BACKBONE,
                "files": calib["files"][:IMG_N]}, out_path)
    print(f"saved {tuple(X.shape)} -> {out_path}")


def _retrieval(a: np.ndarray, b: np.ndarray,
               dup_groups: np.ndarray | None = None) -> tuple[float, float]:
    """a rows query b rows; self-match = correct. Returns (R@1, R@5).
    If dup_groups is given (int group id per row, identical texts share a
    group), retrieving any row with the same group id counts as correct."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    sims = an @ bn.T
    order = np.argsort(-sims, axis=1)
    tgt = np.arange(len(a))
    if dup_groups is not None:
        hit = dup_groups[order] == dup_groups[tgt][:, None]
    else:
        hit = order == tgt[:, None]
    r1 = float(hit[:, 0].mean())
    r5 = float(hit[:, :5].any(1).mean())
    return r1, r5


def _pca_basis(Y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k PCA basis of Y: returns (mu, Vk) with Vk [d, k]."""
    mu = Y.mean(0)
    _, _, vt = np.linalg.svd(Y - mu, full_matrices=False)
    return mu, vt[:k].T


def _rand_basis(d: int, k: int, seed: int = 0) -> np.ndarray:
    """Random orthonormal k-dim basis [d, k], fixed seed."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    return q


def analyze(ref_path: str, mlx_path: str | None, head_file: str) -> None:
    from huggingface_hub import hf_hub_download

    ref = torch.load(ref_path, map_location="cpu", weights_only=True)["states"].numpy()
    calib = load_calib()
    cal = calib["cap5"].float().numpy()

    # duplicate-aware grouping: identical caption strings share a group id
    # (13 captions occur more than once in captions5; without this, rows
    # that lose to an identical twin are scored as misses and pollute the
    # same-runtime floor).
    texts = [c for caps in calib["captions5"] for c in caps]
    seen: dict[str, int] = {}
    dup_groups = np.array([seen.setdefault(t, len(seen)) for t in texts])

    d = torch.load(hf_hub_download(HEAD_REPO, head_file),
                   map_location="cpu", weights_only=True)
    W = d["txt"]["weight"].float().numpy()
    b = d["txt"]["bias"].float().numpy()
    mu = d["mu_txt"].float().numpy()

    def head(X, mu_):
        return (X - mu_) @ W.T + b

    # Subspace-control arms (2026-08-05 reader follow-up): is the head's
    # +3.7 a fact about where the drift lives, or would any 5376->1024
    # projection gain? E projects both sides onto the top-1024 PCA basis
    # of the stored reference (high-variance subspace); F onto a random
    # orthonormal 1024-dim basis. If E lands at/below raw while the head
    # gains, the drift is pinned to the high-variance complement of the
    # head's row space.
    k = W.shape[0]
    mu_pca, Vk = _pca_basis(cal, k)
    Qk = _rand_basis(cal.shape[1], k)

    pairs = [("fresh-CUDA vs stored-calib", ref, cal)]
    if mlx_path:
        mlx = np.load(mlx_path)["states"]
        pairs.append(("MLX-Q4 vs fresh-CUDA", mlx, ref))
        pairs.append(("MLX-Q4 vs stored-calib", mlx, cal))

    results = {}
    for name, X, Y in pairs:
        n = min(len(X), len(Y))
        X, Y = X[:n], Y[:n]
        arms = {
            "A_raw_asis": (X, Y),
            "B_mean_centered": (X - X.mean(0), Y - Y.mean(0)),
            "C_head_shared_mu": (head(X, mu), head(Y, mu)),
            "D_head_per_runtime_mu": (head(X, X.mean(0)), head(Y, Y.mean(0))),
            "E_pca_top1024": ((X - mu_pca) @ Vk, (Y - mu_pca) @ Vk),
            "F_random_1024": ((X - mu_pca) @ Qk, (Y - mu_pca) @ Qk),
        }
        results[name] = {}
        print(f"\n=== {name} (n={n}) ===")
        for arm, (xa, ya) in arms.items():
            r1, r5 = _retrieval(xa, ya, dup_groups[:n])
            r1b, r5b = _retrieval(ya, xa, dup_groups[:n])
            r1s, _ = _retrieval(xa, ya)          # strict, no dup credit
            results[name][arm] = {"r@1": r1, "r@5": r5,
                                  "r@1_rev": r1b, "r@5_rev": r5b,
                                  "r@1_strict": r1s}
            print(f"  {arm:24s} R@1 {r1:.4f} / R@5 {r5:.4f}   "
                  f"(reverse {r1b:.4f} / {r5b:.4f})")
    print("\n" + json.dumps(results, indent=2))


def _i2t(Zi: np.ndarray, Zt: np.ndarray) -> tuple[float, float]:
    """i2t retrieval: image i's correct captions are rows 5i..5i+4 of the
    caption gallery. Returns (R@1, R@5)."""
    zi = Zi / (np.linalg.norm(Zi, axis=1, keepdims=True) + 1e-8)
    zt = Zt / (np.linalg.norm(Zt, axis=1, keepdims=True) + 1e-8)
    order = np.argsort(-(zi @ zt.T), axis=1)
    correct = order // 5 == np.arange(len(Zi))[:, None]
    r1 = float(correct[:, 0].mean())
    r5 = float(correct[:, :5].any(1).mean())
    return r1, r5


def analyze_images(ref_path: str, mlx_path: str | None,
                   cap_path: str | None, head_file: str) -> None:
    from huggingface_hub import hf_hub_download

    ref = torch.load(ref_path, map_location="cpu",
                     weights_only=True)["states"].numpy()
    calib = load_calib()
    cal = calib["img"].float().numpy()[:IMG_N]

    d = torch.load(hf_hub_download(HEAD_REPO, head_file),
                   map_location="cpu", weights_only=True)
    Wi = d["img"]["weight"].float().numpy()
    bi = d["img"]["bias"].float().numpy()
    mu_i = d["mu_img"].float().numpy()
    Wt = d["txt"]["weight"].float().numpy()
    bt = d["txt"]["bias"].float().numpy()
    mu_t = d["mu_txt"].float().numpy()

    def head_img(X, mu_):
        return (X - mu_) @ Wi.T + bi

    # ---- four-arm cross-runtime agreement (image states) ----
    pairs = [("img fresh-CUDA vs stored-calib", ref, cal)]
    if mlx_path:
        mlx = np.load(mlx_path)["states"]
        pairs.append(("img MLX-Q4 vs fresh-CUDA", mlx, ref))
        pairs.append(("img MLX-Q4 vs stored-calib", mlx, cal))

    results = {}
    for name, X, Y in pairs:
        n = min(len(X), len(Y))
        X, Y = X[:n], Y[:n]
        arms = {
            "A_raw_asis": (X, Y),
            "B_mean_centered": (X - X.mean(0), Y - Y.mean(0)),
            "C_head_shared_mu": (head_img(X, mu_i), head_img(Y, mu_i)),
            "D_head_per_runtime_mu": (head_img(X, X.mean(0)),
                                      head_img(Y, Y.mean(0))),
        }
        results[name] = {}
        print(f"\n=== {name} (n={n}) ===")
        for arm, (xa, ya) in arms.items():
            r1, r5 = _retrieval(xa, ya)
            r1b, r5b = _retrieval(ya, xa)
            results[name][arm] = {"r@1": r1, "r@5": r5,
                                  "r@1_rev": r1b, "r@5_rev": r5b}
            print(f"  {arm:24s} R@1 {r1:.4f} / R@5 {r5:.4f}   "
                  f"(reverse {r1b:.4f} / {r5b:.4f})")

    # ---- end-task i2t: image queries vs the 5000-caption gallery ----
    if cap_path:
        cap = torch.load(cap_path, map_location="cpu",
                         weights_only=True)["states"].numpy()
    else:
        cap = calib["cap5"].float().numpy()
    Zt = (cap - mu_t) @ Wt.T + bt

    i2t_arms = [("CUDA fresh, shared mu", head_img(ref, mu_i)),
                ("CUDA fresh, per-runtime mu", head_img(ref, ref.mean(0))),
                ("stored calib, shared mu", head_img(cal, mu_i))]
    if mlx_path:
        mlx = np.load(mlx_path)["states"][:IMG_N]
        i2t_arms += [("MLX-Q4, shared mu", head_img(mlx, mu_i)),
                     ("MLX-Q4, per-runtime mu", head_img(mlx, mlx.mean(0)))]
    results["i2t_end_task"] = {}
    print(f"\n=== i2t end task (n_img={len(ref)}, gallery={len(cap)}) ===")
    for name, Zi in i2t_arms:
        r1, r5 = _i2t(Zi, Zt)
        results["i2t_end_task"][name] = {"r@1": r1, "r@5": r5}
        print(f"  {name:28s} R@1 {r1:.4f} / R@5 {r5:.4f}")

    print("\n" + json.dumps(results, indent=2))


def main():
    args = parse_args()
    if args.out:
        encode(args.out)
    if args.analyze:
        analyze(args.analyze, args.mlx_states, args.head_file)
    if args.out_images:
        encode_images(args.out_images, args.img_dir)
    if args.analyze_images:
        analyze_images(args.analyze_images, args.mlx_img_states,
                       args.cap_states, args.head_file)


if __name__ == "__main__":
    main()
