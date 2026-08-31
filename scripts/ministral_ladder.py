#!/usr/bin/env python3
"""Ministral-3 ladder: an independent replication of the section-5 scaling curve.

The Qwen2.5-Coder ladder answers "does the chat-template effect scale" inside one
lab, one recipe, one tokenizer, in code. That is a controlled result and also a
narrow one: if it is a Qwen artifact, or a code artifact, the curve proves nothing
about the claim in paper_hivemind section 5.

Mistral publishes base and instruct at 3B, 8B and 14B. Same protocol, different
lab, general-domain prompts rather than code. Two independent ladders agreeing is
a much harder result to dismiss than one.

Prompts are the section-5 stems, deliberately, so this arm is directly comparable
to the six small pairs already in the paper rather than to the code ladder.

    CUDA_VISIBLE_DEVICES=0 python scripts/ministral_ladder.py --shard 0 --shards 3
"""
import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hivemind_gen import run_model, Progress
from hivemind_mechanism_link import STEMS

HUB = "/root/.hf_home/hub"
OUT = "artifacts/nla/ministral_ladder"
PERSONA = "You are a helpful assistant."
SIZES = ["3B", "8B", "14B"]
BASE_ARMS = ["raw"]
INST_ARMS = ["raw", "persona", "shared", "shared_persona", "chat"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--prompts", type=int, default=60)
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--batch", type=int, default=16)
    return p.parse_args()


def snap(repo):
    g = glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*/")
    return g[0] if g else None


def render(tok, stem, mode):
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
    prompts = STEMS[:a.prompts]
    jobs = []
    for s in SIZES:
        jobs.append((f"ministral{s}_base", f"mistralai/Ministral-3-{s}-Base-2512", BASE_ARMS))
        jobs.append((f"ministral{s}_inst", f"mistralai/Ministral-3-{s}-Instruct-2512", INST_ARMS))
    order = {s: i for i, s in enumerate(reversed(SIZES))}
    jobs.sort(key=lambda j: order[j[0].replace("ministral", "").split("_")[0]])
    mine = [j for i, j in enumerate(jobs) if i % a.shards == a.shard]

    prog = Progress(sum(len(arms) * len(prompts) for _, _, arms in mine))
    print(f"shard {a.shard}/{a.shards}  {len(mine)} models  {len(prompts)} prompts  "
          f"K={a.k}  batch={a.batch}", flush=True)
    print(f"  {', '.join(t for t, _, _ in mine)}\n", flush=True)

    os.makedirs(OUT, exist_ok=True)
    for tag, repo, arms in mine:
        path = snap(repo)
        if path is None:
            prog.tick(len(arms) * len(prompts))
            print(f"  {tag:18s} MISSING {repo}", flush=True)
            continue
        try:
            run_model(tag, path, prompts, arms, render, OUT, prog,
                      k=a.k, max_new=a.max_new, batch=a.batch, dtype=torch.bfloat16)
        except Exception as e:
            prog.tick(len(arms) * len(prompts))
            print(f"  {tag:18s} FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
    print(f"\nshard {a.shard} complete  {prog.fmt()}", flush=True)


if __name__ == "__main__":
    main()
