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
# HF_HOME moved; earlier models are still under /root. Search both.
HUBS = [h for h in (os.environ.get("HF_HOME", "") and
                    os.path.join(os.environ["HF_HOME"], "hub"),
                    "/workspace/.hf_home/hub", HUB) if h]
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
    p.add_argument("--sizes", nargs="*", default=SIZES)
    p.add_argument("--arms", nargs="*", default=None)
    # Generations are named by tag alone, so a different K must write elsewhere.
    p.add_argument("--out", default=OUT)
    return p.parse_args()


def snap(repo):
    """First snapshot of `repo` in any known cache.

    HF_HOME moved to /workspace/.hf_home while 830 GB of earlier models stayed in
    /root/.hf_home, so a single hardcoded hub silently misses half the models.
    """
    stem = f"models--{repo.replace('/', '--')}/snapshots/*/"
    for hub in HUBS:
        g = glob.glob(f"{hub}/{stem}")
        if g:
            return g[0]
    return None


def render(tok, stem, mode):
    if mode == "raw":
        return stem, True, False
    if mode == "persona":
        return f"{PERSONA}\n\n{stem}", True, False
    if mode == "shared":
        return f"### User:\n{stem}\n\n### Assistant:\n", True, False
    if mode == "shared_persona":
        return f"{PERSONA}\n\n### User:\n{stem}\n\n### Assistant:\n", True, False
    # MistralCommonBackend cannot round-trip special tokens through a string.
    return tok.apply_chat_template([{"role": "user", "content": stem}], tokenize=True,
                                   add_generation_prompt=True)["input_ids"], False, False


def main():
    a = parse_args()
    out = a.out
    prompts = STEMS[:a.prompts]
    inst_arms = a.arms or INST_ARMS
    base_arms = [m for m in (a.arms or BASE_ARMS) if m in BASE_ARMS]
    jobs = []
    for s in a.sizes:
        if base_arms:
            jobs.append((f"ministral{s}_base",
                         f"mistralai/Ministral-3-{s}-Base-2512", base_arms))
        jobs.append((f"ministral{s}_inst",
                     f"mistralai/Ministral-3-{s}-Instruct-2512", inst_arms))
    order = {s: i for i, s in enumerate(reversed(a.sizes))}
    jobs.sort(key=lambda j: order[j[0].replace("ministral", "").split("_")[0]])
    mine = [j for i, j in enumerate(jobs) if i % a.shards == a.shard]

    prog = Progress(sum(len(arms) * len(prompts) for _, _, arms in mine))
    print(f"shard {a.shard}/{a.shards}  {len(mine)} models  {len(prompts)} prompts  "
          f"K={a.k}  batch={a.batch}  -> {out}", flush=True)
    print(f"  {', '.join(t for t, _, _ in mine)}\n", flush=True)

    os.makedirs(out, exist_ok=True)
    failed = []
    for tag, repo, arms in mine:
        path = snap(repo)
        if path is None:
            prog.tick(len(arms) * len(prompts))
            print(f"  {tag:18s} MISSING {repo}", flush=True)
            failed.append(tag)
            continue
        try:
            run_model(tag, path, prompts, arms, render, out, prog,
                      k=a.k, max_new=a.max_new, batch=a.batch, dtype=torch.bfloat16)
        except Exception as e:
            prog.tick(len(arms) * len(prompts))
            print(f"  {tag:18s} FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
            failed.append(tag)
    print(f"\nshard {a.shard} complete  {prog.fmt()}", flush=True)
    if failed:
        print(f"FAILED: {', '.join(failed)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
