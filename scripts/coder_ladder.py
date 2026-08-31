#!/usr/bin/env python3
"""Qwen2.5-Coder ladder: the section-5 scaling curve, in code.

paper_hivemind section 5 rests on six matched base/instruct pairs between 0.36B
and 2B, and the obvious attack is that the chat-template effect is a small-model
artifact. This runs the identical protocol on a ladder that spans 66x inside one
family, so lab, recipe and tokenizer are all held fixed and only scale moves.

  0.5B  1.5B  3B  7B  14B  32B, base and instruct at every rung

Code rather than prose for three reasons. It closes the paper's stated limitation
that transport was only measured on COCO captions. It gives ground truth, so
pass@k can sit beside similarity and answer whether homogeneity costs capability.
And HumanEval prompts are real function signatures, not hand-written stems.

Arms follow section 5.1: base models get `raw` only, since they have no template.
Instruct models get all five, because that is where the decomposition lives.

One process owns one GPU. Cards are independent here (PCIe, no NVLink, and GPU0
sits on a different NUMA node), so model-per-card beats tensor parallel.

    CUDA_VISIBLE_DEVICES=0 python scripts/coder_ladder.py --shard 0 --shards 4
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hivemind_gen import run_model, Progress

HUB = "/root/.hf_home/hub"
OUT = "artifacts/nla/coder_ladder"
PERSONA = "You are a helpful assistant."
SIZES = ["0.5B", "1.5B", "3B", "7B", "14B", "32B"]
BASE_ARMS = ["raw"]
INST_ARMS = ["raw", "persona", "shared", "shared_persona", "chat"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--prompts", type=int, default=164)
    p.add_argument("--max-new", type=int, default=192)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--sizes", nargs="*", default=SIZES)
    p.add_argument("--arms", nargs="*", default=None,
                   help="restrict instruct arms; base arms are dropped unless 'raw' is kept")
    # Generations are named by tag alone, so a different K must write elsewhere
    # or it silently overwrites the published K=8 pools.
    p.add_argument("--out", default=OUT)
    return p.parse_args()


def snap(repo):
    g = glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*/")
    return g[0] if g else None


def render(tok, stem, mode):
    """Same five arms as section 5.1, as plain string transforms."""
    if mode == "raw":
        return stem, True, False
    if mode == "persona":
        return f"{PERSONA}\n\n{stem}", True, False
    if mode == "shared":
        return f"### User:\n{stem}\n\n### Assistant:\n", True, False
    if mode == "shared_persona":
        return f"{PERSONA}\n\n### User:\n{stem}\n\n### Assistant:\n", True, False
    return tok.apply_chat_template([{"role": "user", "content": stem}],
                                   tokenize=False, add_generation_prompt=True), False, False


def main():
    a = parse_args()
    out = a.out
    he = json.load(open("data/humaneval.json"))[:a.prompts]
    prompts = [p["prompt"] for p in he]
    os.makedirs(out, exist_ok=True)
    json.dump([p["task_id"] for p in he], open(f"{out}/task_ids.json", "w"))

    inst_arms = a.arms or INST_ARMS
    base_arms = [m for m in (a.arms or BASE_ARMS) if m in BASE_ARMS]
    jobs = []
    for s in a.sizes:
        if base_arms:
            jobs.append((f"coder{s}_base", f"Qwen/Qwen2.5-Coder-{s}", base_arms))
        jobs.append((f"coder{s}_inst", f"Qwen/Qwen2.5-Coder-{s}-Instruct", inst_arms))
    # Largest first so the slow model starts immediately rather than last.
    order = {s: i for i, s in enumerate(reversed(a.sizes))}
    jobs.sort(key=lambda j: order[j[0].replace("coder", "").split("_")[0]])
    mine = [j for i, j in enumerate(jobs) if i % a.shards == a.shard]

    total = sum(len(arms) * len(prompts) for _, _, arms in mine)
    prog = Progress(total)
    print(f"shard {a.shard}/{a.shards}  {len(mine)} models  {len(prompts)} prompts  "
          f"K={a.k}  batch={a.batch}  -> {out}", flush=True)
    print(f"  {', '.join(t for t, _, _ in mine)}\n", flush=True)

    for tag, repo, arms in mine:
        path = snap(repo)
        if path is None:
            prog.tick(len(arms) * len(prompts))
            print(f"  {tag:16s} MISSING {repo}", flush=True)
            continue
        try:
            run_model(tag, path, prompts, arms, render, out, prog,
                      k=a.k, max_new=a.max_new, batch=a.batch, dtype=torch.bfloat16)
        except Exception as e:
            prog.tick(len(arms) * len(prompts))
            print(f"  {tag:16s} FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
    print(f"\nshard {a.shard} complete  {prog.fmt()}", flush=True)


if __name__ == "__main__":
    main()
