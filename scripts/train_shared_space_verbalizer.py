#!/usr/bin/env python
"""Can the 0.6B put words to a vector only the 27B ever saw?

The demo already shows the two models' spaces are comparable: a 0.6B text state
and a 27B image state land close enough to retrieve each other. That is
matching. This asks the harder question, whether the small model can SPEAK the
large model's reading.

A prefix module maps one 1024-d shared-space vector to `--np` soft tokens in the
0.6B's embedding space. The backbone is frozen; only the prefix trains, on
cross-entropy against COCO captions. Nothing about the image reaches the model
except through that vector, which the 27B produced and the 0.6B has never seen
an image to make.

The scoring is deliberately the same instrument the demo uses. Generate a
caption for a held-out image, re-encode it with the SHIPPED text head, and see
whether it retrieves its own image out of 123,287. If the words carry the
27B's reading, the gold image comes back.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn


class Prefix(nn.Module):
    """Shared-space vector -> soft prompt. The only trained parameters."""

    def __init__(self, d_in: int, d_model: int, n_tok: int, hidden: int = 2048):
        super().__init__()
        self.n_tok, self.d_model = n_tok, d_model
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, n_tok * d_model),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.net(v).view(v.size(0), self.n_tok, self.d_model)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--vecs", default="/tmp/val_img_vecs.npy")
    p.add_argument("--caps", default="/tmp/val_caps.json")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--np", dest="n_tok", type=int, default=16)
    p.add_argument("--holdout", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-len", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--out", default="artifacts/local/browser/shared_verbalizer.pt")
    return p.parse_args()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    a = parse()
    if a.device != "auto":
        dev = a.device
    elif torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    vecs = np.load(a.vecs)
    meta = json.load(open(a.caps))
    caps = meta["captions"]

    # Split by IMAGE, so no caption of a held-out image is ever trained on.
    n_img = len(caps)
    rng = np.random.default_rng(0)
    order = rng.permutation(n_img)
    test_idx, train_idx = order[: a.holdout], order[a.holdout:]
    pairs = [(i, c) for i in train_idx for c in caps[i]]
    print(f"{n_img} images -> {len(train_idx)} train / {len(test_idx)} held out; "
          f"{len(pairs)} training captions", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    wdtype = torch.bfloat16 if a.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=wdtype).to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    emb = model.get_input_embeddings()
    d_model = emb.weight.size(1)

    # Raw large-model states are strongly anisotropic; centre and scale on the
    # train split only, and carry the statistics in the checkpoint so eval and
    # the mean-vector control use exactly the same frame.
    mu = vecs[train_idx].mean(0)
    sd = float(np.linalg.norm(vecs[train_idx] - mu, axis=1).mean()) or 1.0
    print(f"input d={vecs.shape[1]} ||mu||={np.linalg.norm(mu):.2f} mean_radius={sd:.2f}", flush=True)

    pre = Prefix(vecs.shape[1], d_model, a.n_tok).to(dev)
    opt = torch.optim.AdamW(pre.parameters(), lr=a.lr)
    V = torch.tensor((vecs - mu) / sd, device=dev, dtype=torch.float32)

    def save(tag: str) -> None:
        torch.save({"prefix": pre.state_dict(), "n_tok": a.n_tok, "d_in": vecs.shape[1],
                    "d_model": d_model, "mu": mu, "sd": sd,
                    "test_idx": test_idx.tolist(),
                    "images": [meta["images"][i] for i in test_idx]}, a.out)
        print(f"saved {a.out} ({tag})", flush=True)

    step = 0
    for ep in range(a.epochs):
        rng.shuffle(pairs)
        for i in range(0, len(pairs) - a.batch + 1, a.batch):
            chunk = pairs[i:i + a.batch]
            ids = tok([c for _, c in chunk], return_tensors="pt", padding=True,
                      truncation=True, max_length=a.max_len).to(dev)
            v = V[[j for j, _ in chunk]]
            wte = emb(ids["input_ids"])
            soft = pre(v).to(wte.dtype)
            inp = torch.cat([soft, wte], 1)
            attn = torch.cat([torch.ones(soft.shape[:2], device=dev, dtype=ids["attention_mask"].dtype),
                              ids["attention_mask"]], 1)
            labels = torch.cat([torch.full(soft.shape[:2], -100, device=dev, dtype=torch.long),
                                ids["input_ids"].masked_fill(ids["attention_mask"] == 0, -100)], 1)
            loss = model(inputs_embeds=inp, attention_mask=attn, labels=labels).loss
            loss.backward()
            opt.step(); opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"  ep{ep} step {step} loss {loss.item():.4f}", flush=True)
            if step % a.save_every == 0:
                save(f"step {step}")

    save("final")


if __name__ == "__main__":
    main()
