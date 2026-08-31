"""MBPP candidate pools, for testing whether the HumanEval read ranking transfers.

The read comparison so far lives entirely on HumanEval, so it cannot separate "this
method works" from "this method fits HumanEval". MBPP is a different distribution:
the prompt is a prose description plus a single assert rather than a signature and a
docstring, and the answer is a whole function rather than a body continuation.

Same models and same decoding as the coder ladder, so per-arm numbers stay comparable
across the two benchmarks.

    python scripts/mbpp_ladder.py --shard 0 --shards 3
"""
import argparse
import glob
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
from hivemind_gen import Progress, run_model  # noqa: E402

HUB = "/root/.hf_home/hub"
OUT = "artifacts/nla/mbpp_ladder"
SIZES = ["0.5B", "1.5B", "3B", "7B", "14B", "32B"]
ARMS = ["chat", "shared"]
INSTR = "Write a Python function for the task below. Reply with the full function only.\n\n"
SHARED = "### User:\n" + INSTR + "{p}\n\n### Assistant:\n"


def snap(repo):
    g = glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*/")
    return g[0] if g else None


def render(tok, stem, mode):
    if mode == "shared":
        return SHARED.format(p=stem), True, False
    o = tok.apply_chat_template([{"role": "user", "content": INSTR + stem}],
                                tokenize=True, add_generation_prompt=True)
    return (o["input_ids"] if hasattr(o, "keys") else o), False, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default="data/mbpp.json")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--sizes", nargs="*", default=SIZES)
    a = ap.parse_args()

    probs = json.load(open(os.path.join(HERE, a.probs)))
    stems = [p["prompt"] for p in probs]
    jobs = [(f"mbpp{s}_inst", f"Qwen/Qwen2.5-Coder-{s}-Instruct") for s in a.sizes]
    jobs.sort(key=lambda j: -float(j[0].replace("mbpp", "").replace("B_inst", "")))
    mine = [j for i, j in enumerate(jobs) if i % a.shards == a.shard]
    print(f"shard {a.shard}/{a.shards}  {len(mine)} models  {len(stems)} prompts  "
          f"K={a.k}  max_new={a.max_new}", flush=True)

    prog = Progress(len(mine) * len(ARMS) * len(stems))
    for tag, repo in mine:
        p = snap(repo)
        if not p:
            print(f"  {tag:16s} MISSING {repo}", flush=True)
            continue
        run_model(tag, p, stems, ARMS, render, os.path.join(HERE, OUT), prog,
                  k=a.k, max_new=a.max_new, batch=a.batch, dtype=torch.bfloat16)


if __name__ == "__main__":
    main()
