"""A continuous field-strength dial for the chat template.

`paper_format_susceptibility.md` finds the template acts as a one-sided floor on
output order: below roughly 0.88 it lifts hard, at or above it does almost
nothing. That was measured with four discrete arms, which is a ladder, not a
control parameter. A floor is exactly the shape that should show a threshold as
the field is weakened continuously, and a threshold in an order parameter against
a continuous control is what the bifurcation reading needs and does not have.

The dial. Build the templated sequence as prefix + stem + suffix so the template
positions are known exactly, then replace the embedding at each template position
with

    e'  =  neutral + alpha * (e - neutral)

where `neutral` is the mean of the embedding matrix. At alpha = 1 this is exactly
the chat arm. At alpha = 0 the template positions are still there, in the same
places, carrying the average token instead of turn structure.

That control is better than comparing against the raw arm, because it holds
sequence length and position fixed and varies only what the template positions
mean. Any change across the sweep is the template's content, not its presence.

    python scripts/format_field_sweep.py --model Qwen/Qwen2.5-Coder-3B-Instruct --tag coder3B
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

SENTINEL = "\u0001STEM\u0001"


def embed_matrix(path):
    """Read embed_tokens.weight without instantiating the model."""
    from safetensors import safe_open
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(idx):
        m = json.load(open(idx))["weight_map"]
        key = next(k for k in m if k.endswith("embed_tokens.weight"))
        shards = [os.path.join(path, m[key])]
    else:
        shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        key = None
    for s in shards:
        with safe_open(s, framework="pt") as f:
            for k in f.keys():
                if k.endswith("embed_tokens.weight"):
                    return f.get_tensor(k)
    raise SystemExit("embed_tokens.weight not found")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--tag", default="coder3B")
    ap.add_argument("--alphas", nargs="*", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--prompts", type=int, default=164)
    ap.add_argument("--probs", default="data/humaneval.json")
    ap.add_argument("--out", default="artifacts/nla/field_sweep")
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    a = ap.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.makedirs(a.out, exist_ok=True)
    path = snapshot_download(a.model)
    tok = AutoTokenizer.from_pretrained(a.model)
    stems = [p["prompt"] for p in json.load(open(a.probs))[: a.prompts]]

    full = tok.apply_chat_template([{"role": "user", "content": SENTINEL}],
                                   tokenize=False, add_generation_prompt=True)
    pre, suf = full.split(SENTINEL)
    pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
    suf_ids = tok(suf, add_special_tokens=False)["input_ids"]

    seqs = []
    for s in stems:
        mid = tok(s, add_special_tokens=False)["input_ids"]
        seqs.append((pre_ids, mid, suf_ids))

    # The engine must be built before the embedding matrix is read. vLLM forks a
    # worker, and forking a parent that already holds a multi-GB torch tensor
    # deadlocks before any weights load.
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_frac,
              max_model_len=4096, enable_prompt_embeds=True, enforce_eager=True)
    sp = SamplingParams(n=a.k, temperature=1.0, top_p=0.9, max_tokens=a.max_new)

    W = embed_matrix(path).float()
    neutral = W.mean(0)
    print(f"{a.tag}: vocab {tuple(W.shape)}, template = {len(pre_ids)} + stem + "
          f"{len(suf_ids)} tokens", flush=True)

    for al in a.alphas:
        p = os.path.join(a.out, f"{a.tag}_a{al:.2f}__chat.json")
        if os.path.isfile(p):
            print(f"  alpha {al:.2f} cached", flush=True)
            continue
        batch = []
        for pre_i, mid_i, suf_i in seqs:
            e_pre = neutral + al * (W[pre_i] - neutral)
            e_suf = neutral + al * (W[suf_i] - neutral)
            e = torch.cat([e_pre, W[mid_i], e_suf], 0).to(torch.bfloat16)
            batch.append({"prompt_embeds": e})
        outs = llm.generate(batch, sp)
        rows = [[c.text for c in o.outputs] for o in outs]
        json.dump(rows, open(p, "w"))
        cut = sum(1 for o in outs for c in o.outputs if c.finish_reason == "length")
        n = sum(len(r) for r in rows)
        print(f"  alpha {al:.2f}  {n} cands  truncated {cut / max(1, n):.1%}  -> {p}",
              flush=True)

    print("field sweep complete", flush=True)


if __name__ == "__main__":
    main()
