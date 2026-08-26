#!/usr/bin/env python
"""Can the reader speak any point in the shared space, or only image records?

The verbalizer was trained on image records. But the head puts text and images
in one space, so a query vector, a midpoint between two photographs, and a
steered vector are all points in the same 1024-d room. If the reader only works
on the image manifold it is a captioner. If it speaks anywhere, the space
itself is legible and the demo is "hear what is at this location", not
"describe this picture".

    python scripts/probe_space_readable.py --index gallery.srtidx --head head.safetensors
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_shared_space_verbalizer import load_index  # noqa: E402
from train_shared_space_verbalizer import Prefix  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/fullstate_verbalizer/gallery_verbalizer.pt")
    p.add_argument("--index", default="artifacts/local/browser/gallery_123k_v3.srtidx")
    p.add_argument("--head", default="/Users/burtron/development/BlackWindow/wasm/dist/head.safetensors")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--layer", type=int, default=28)
    p.add_argument("--max-new", type=int, default=32)
    return p.parse_args()


def main():
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    a = parse()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    pre = Prefix(ck["d_in"], ck["d_model"], ck["n_tok"], ck.get("hidden", 2048))
    pre.load_state_dict(ck["prefix"])
    pre.eval()

    gal, pos = load_index(a.index)
    keys = list(pos)
    h = load_file(a.head)
    W = h["txt.weight"].float().numpy()
    b = h["txt.bias"].float().numpy()
    mu_txt = h["mu_txt"].float().numpy()

    @torch.no_grad()
    def encode_text(text: str) -> np.ndarray:
        enc = tok([text], return_tensors="pt", truncation=True, max_length=64)
        hs = lm(**enc, output_hidden_states=True).hidden_states[a.layer].float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        pooled = ((hs * m).sum(1) / m.sum(1)).numpy()[0]
        v = (pooled - mu_txt) @ W.T + b
        return v / (np.linalg.norm(v) + 1e-8)

    @torch.no_grad()
    def say(v: np.ndarray) -> str:
        v = v / (np.linalg.norm(v) + 1e-8)          # the index stores unit rows
        x = torch.tensor((v - ck["mu"]) / ck["sd"], dtype=torch.float32).unsqueeze(0)
        soft = pre(x)
        out = lm.generate(inputs_embeds=soft, max_new_tokens=a.max_new,
                          do_sample=False, pad_token_id=tok.eos_token_id,
                          attention_mask=torch.ones(soft.shape[:2], dtype=torch.long))
        t = tok.decode(out[0], skip_special_tokens=True).strip()
        cut = t.rfind(".")
        return t[: cut + 1] if cut > 0 else t

    def nearest(v: np.ndarray) -> str:
        v = v / (np.linalg.norm(v) + 1e-8)
        return keys[int(np.argmax(gal @ v))]

    print("=" * 72)
    print("1. AN IMAGE RECORD (in distribution, the trained case)")
    rng = np.random.default_rng(0)
    rows = [int(r) for r in rng.integers(0, len(keys), 2)]
    for row in rows:
        print(f"   {keys[row]}: {say(gal[row])}")

    print("=" * 72)
    print("2. A SENTENCE YOU TYPED, encoded by the head then read back")
    print("   (round trip through the shared space: text in, text out)")
    for q in ["a dog asleep in the afternoon sun",
              "two people arguing in a kitchen",
              "a red bus crossing a bridge in the rain"]:
        v = encode_text(q)
        print(f"   you: {q}")
        print(f"   it : {say(v)}")
        print(f"   nearest photo: {nearest(v)}")

    print("=" * 72)
    print("3. THE MIDPOINT BETWEEN TWO PHOTOGRAPHS (no photograph is here)")
    a_row, b_row = rows
    print(f"   from {keys[a_row]} to {keys[b_row]}")
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = (1 - t) * gal[a_row] + t * gal[b_row]
        print(f"   t={t:.2f}: {say(v)}")

    print("=" * 72)
    print("4. SEMANTIC ARITHMETIC: a query moved along an axis built from words")
    pos_txt = ["a dog", "a horse", "a bear", "a zebra", "a bird"]
    neg_txt = ["a car", "a bus", "a train", "a motorcycle", "a boat"]
    axis = np.mean([encode_text(t) for t in pos_txt], 0) - \
        np.mean([encode_text(t) for t in neg_txt], 0)
    axis /= np.linalg.norm(axis) + 1e-8
    base = encode_text("something moving fast down a street")
    for alpha in (-1.0, -0.5, 0.0, 0.5, 1.0):
        v = base + alpha * axis
        print(f"   alpha={alpha:+.1f}: {say(v)}")

    print("=" * 72)
    print("5. THE CENTRE OF THE GALLERY (the average of 123,287 photographs)")
    print(f"   {say(gal.mean(0))}")


if __name__ == "__main__":
    main()
