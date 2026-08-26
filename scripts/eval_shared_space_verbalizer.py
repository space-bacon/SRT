#!/usr/bin/env python
"""Score the shared-space verbalizer with the demo's own instrument.

Generate a caption from a held-out image vector, re-encode it through the
SHIPPED text head, and retrieve against the full 123,287-image gallery. If the
words carry the 27B's reading, the gold image comes back.

Arms, because a caption prior alone will produce plausible COCO sentences and
plausible is not the same as informative:

    real        the image's own vector
    foreign     another held-out image's vector: does the caption follow the
                vector it was given, or the prior it learned
    mean        the mean of all held-out vectors: pure typicality
    gold        the human caption, through the same encode-and-retrieve path,
                which is the ceiling this can be measured against
"""
from __future__ import annotations

import argparse
import json
import struct

import numpy as np
import torch
from safetensors.torch import load_file


def load_index(path: str):
    raw = open(path, "rb").read()
    dim, n = struct.unpack_from("<II", raw, 8)
    off = 16
    sc = np.frombuffer(raw, "<f4", n, off); off += 4 * n
    q = np.frombuffer(raw, np.int8, n * dim, off).reshape(n, dim); off += n * dim
    keys = []
    for _ in range(n):
        (l,) = struct.unpack_from("<I", raw, off); off += 4
        keys.append(raw[off:off + l].decode()); off += l
    z = q.astype(np.float32) * sc[:, None]
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
    return z, {k: i for i, k in enumerate(keys)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="artifacts/local/browser/shared_verbalizer.pt")
    p.add_argument("--vecs", default="/tmp/val_img_vecs.npy")
    p.add_argument("--caps", default="/tmp/val_caps.json")
    p.add_argument("--head", default="/Users/burtron/development/BlackWindow/wasm/dist/head.safetensors")
    p.add_argument("--index", default="artifacts/local/browser/gallery_123k_v3.srtidx")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--layer", type=int, default=28)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--max-new", type=int, default=24)
    p.add_argument("--device", default="auto")
    p.add_argument("--img-prefix", default="val2017/")
    p.add_argument("--all-rows", action="store_true",
                   help="score every row, for a set that is held out by construction")
    p.add_argument("--out", default="artifacts/nla/q4/shared_space_verbalizer.json")
    a = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from train_shared_space_verbalizer import Prefix

    dev = (a.device if a.device != "auto"
           else "cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    vecs = np.load(a.vecs)
    meta = json.load(open(a.caps))
    gal, pos = load_index(a.index)

    test = list(range(len(vecs)))[: a.n] if a.all_rows else ck["test_idx"][: a.n]
    gold_rows = np.array([pos[f"{a.img_prefix}{meta['images'][i]}"] for i in test])

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    emb = model.get_input_embeddings()
    pre = Prefix(ck["d_in"], ck["d_model"], ck["n_tok"]).to(dev)
    pre.load_state_dict(ck["prefix"]); pre.eval()

    h = load_file(a.head)
    W = h["txt.weight"].float().numpy(); b = h["txt.bias"].float().numpy()
    mu = h["mu_txt"].float().numpy()

    @torch.no_grad()
    def project(texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], return_tensors="pt", padding=True,
                      truncation=True, max_length=64).to(dev)
            hs = model(**enc, output_hidden_states=True).hidden_states[a.layer].float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = ((hs * m).sum(1) / m.sum(1)).cpu().numpy()
            v = (pooled - mu) @ W.T + b
            out.append(v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8))
        return np.concatenate(out)

    @torch.no_grad()
    def speak(v: np.ndarray) -> list[str]:
        # Same frame the prefix was trained in, carried in the checkpoint.
        v = (v - ck["mu"]) / ck["sd"] if "mu" in ck else v
        said = []
        for i in range(0, len(v), 16):
            soft = pre(torch.tensor(v[i:i + 16], device=dev, dtype=torch.float32))
            gen = model.generate(inputs_embeds=soft, max_new_tokens=a.max_new,
                                 do_sample=False,
                                 attention_mask=torch.ones(soft.shape[:2], dtype=torch.long, device=dev),
                                 pad_token_id=tok.eos_token_id)
            said += [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)]
        return said

    real = vecs[test]
    # Roll rather than permute: a permutation can leave an image paired with
    # its own vector, which would blunt the control.
    foreign = np.roll(real, 1, axis=0)
    meanv = np.repeat(real.mean(0, keepdims=True), len(test), 0)

    arms = {}
    for name, v in (("real", real), ("foreign", foreign), ("mean", meanv)):
        said = speak(v.astype(np.float32))
        q = project(said)
        s = q @ gal.T
        ranks = (s > s[np.arange(len(s)), gold_rows][:, None]).sum(1)
        arms[name] = {"r@1": round(float((ranks < 1).mean()), 4),
                      "r@5": round(float((ranks < 5).mean()), 4),
                      "r@10": round(float((ranks < 10).mean()), 4),
                      "median_rank": float(np.median(ranks)) + 1,
                      "sample": said[:5]}
        print(f"{name:8s} R@1 {arms[name]['r@1']:.4f}  median {arms[name]['median_rank']:.0f}", flush=True)

    gold = [meta["captions"][i][0] for i in test]
    q = project(gold)
    s = q @ gal.T
    ranks = (s > s[np.arange(len(s)), gold_rows][:, None]).sum(1)
    arms["gold_caption"] = {"r@1": round(float((ranks < 1).mean()), 4),
                            "r@5": round(float((ranks < 5).mean()), 4),
                            "r@10": round(float((ranks < 10).mean()), 4),
                            "median_rank": float(np.median(ranks)) + 1}
    print(f"{'gold':8s} R@1 {arms['gold_caption']['r@1']:.4f}  "
          f"median {arms['gold_caption']['median_rank']:.0f}", flush=True)

    doc = {"question": "can the 0.6B put words to a vector only the 27B saw",
           "n_heldout_scored": len(test), "gallery": len(gal),
           "n_prefix_tokens": ck["n_tok"], "arms": arms}
    json.dump(doc, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
