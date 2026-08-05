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

Arms (cross-runtime self-retrieval@1 over the full pool, plus R@5):
    A  raw states, as-is               (the never-shipped baseline)
    B  mean-centered raw               (what local_sunstone --validate measures)
    C  head-projected, no re-centering (head's own mu on both sides)
    D  head-projected, per-runtime mu  (full pipeline the row claims)

Usage on the box:
    pip install torch transformers accelerate huggingface_hub numpy
    python encode_ref_cuda.py --out ref_cuda.pt
    python encode_ref_cuda.py --analyze ref_cuda.pt [--mlx-states mlx_q4.npz]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

CALIB_REPO = "RiverRider/srt-nla-gemma4-artifacts"
CALIB_FILE = "procrustes/encoded_L47_n5000.pt"
HEAD_REPO = "RiverRider/srt-sunstone-linear-head"
HEAD_FILE = "gemma4_linear_head.pt"
BACKBONE = "google/gemma-4-31B-it"
LAYER = 47
MAX_SEQ = 64
BATCH = 16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None, help="encode fresh CUDA states to this file")
    p.add_argument("--analyze", default=None, help="ref_cuda.pt to analyze")
    p.add_argument("--mlx-states", default=None,
                   help="npz with arrays 'states' (n,d) from the Mac (optional)")
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


def _retrieval(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """a rows query b rows; self-match = correct. Returns (R@1, R@5)."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    sims = an @ bn.T
    order = np.argsort(-sims, axis=1)
    tgt = np.arange(len(a))
    r1 = float((order[:, 0] == tgt).mean())
    r5 = float((order[:, :5] == tgt[:, None]).any(1).mean())
    return r1, r5


def analyze(ref_path: str, mlx_path: str | None, head_file: str) -> None:
    from huggingface_hub import hf_hub_download

    ref = torch.load(ref_path, map_location="cpu", weights_only=True)["states"].numpy()
    calib = load_calib()
    cal = calib["cap5"].float().numpy()

    d = torch.load(hf_hub_download(HEAD_REPO, head_file),
                   map_location="cpu", weights_only=True)
    W = d["txt"]["weight"].float().numpy()
    b = d["txt"]["bias"].float().numpy()
    mu = d["mu_txt"].float().numpy()

    def head(X, mu_):
        return (X - mu_) @ W.T + b

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
    print("\n" + json.dumps(results, indent=2))


def main():
    args = parse_args()
    if args.out:
        encode(args.out)
    if args.analyze:
        analyze(args.analyze, args.mlx_states, args.head_file)


if __name__ == "__main__":
    main()
