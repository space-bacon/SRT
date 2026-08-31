"""Union ceiling: is a multi-lab candidate pool worth more than the best single model?

Every headroom number we have is within one model, where the pool sits near 0.86
intra-similarity and a selector has almost nothing to discriminate. This measures
the ceiling of a pool drawn from several labs instead.

The gate is simple. If union pass@N is not clearly above the best single model's
pass@K, there is no ensemble to build and no selector is worth training. Selection
is a separate question and this script does not attempt it; it only establishes
whether the prize exists.

All members are generated at identical settings so per-model numbers are comparable.
Token budget is deliberately larger than the ladder's 192, because chat models spend
part of it on preamble and a truncated answer scores as a capability failure.

    python scripts/ensemble_ceiling.py --shard 0 --shards 3 --gen-only
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
from code_select import execute_file  # noqa: E402
from hivemind_gen import Progress, run_model  # noqa: E402

HUB = "/root/.hf_home/hub"
# HF_HOME moved; earlier models are still under /root. Search both.
HUBS = [h for h in (os.environ.get("HF_HOME", "") and
                    os.path.join(os.environ["HF_HOME"], "hub"),
                    "/workspace/.hf_home/hub", HUB) if h]
OUT = "artifacts/nla/ensemble"
MODELS = [
    ("qwen25c_32b", "Qwen/Qwen2.5-Coder-32B-Instruct", "Alibaba"),
    ("gemma4_31b", "google/gemma-4-31B-it", "Google"),
    ("ministral_14b", "mistralai/Ministral-3-14B-Instruct-2512", "Mistral"),
    ("glm47_flash", "zai-org/GLM-4.7-Flash", "Zhipu"),
    ("qwen3c_30b", "Qwen/Qwen3-Coder-30B-A3B-Instruct", "Alibaba"),
    # Ornith is a Qwen3.5-architecture post-train, not an independent pretrain.
    ("ornith_35b", "ornith-ai/Ornith-1.5-35B-A3B", "Ornith/Qwen-arch"),
]
INSTR = "Complete the following Python function. Reply with the full function only.\n\n"


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
    msgs = [{"role": "user", "content": INSTR + stem}]
    # GLM-family templates open a <think> block by default, which burns the whole
    # budget on analysis before any code. Templates without the flag ignore it.
    try:
        o = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                    enable_thinking=False)
    except Exception:
        o = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    return (o["input_ids"] if hasattr(o, "keys") else o), False, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default="data/humaneval.json")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--gen-only", action="store_true")
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--out", default=f"{OUT}/union_ceiling.json")
    a = ap.parse_args()

    probs = json.load(open(os.path.join(HERE, a.probs)))
    stems = [p["prompt"] for p in probs]
    mine = [m for i, m in enumerate(MODELS) if i % a.shards == a.shard]
    print(f"shard {a.shard}/{a.shards}  {len(mine)} models  {len(stems)} prompts  "
          f"K={a.k}  max_new={a.max_new}", flush=True)

    prog = Progress(len(mine) * len(stems))
    for tag, repo, lab in mine:
        p = snap(repo)
        if not p:
            print(f"  {tag:16s} MISSING {repo}", flush=True)
            continue
        run_model(tag, p, stems, ["chat"], render, os.path.join(HERE, OUT), prog,
                  k=a.k, max_new=a.max_new, batch=a.batch, dtype=torch.bfloat16)
    if a.gen_only:
        return

    ok, res = {}, {"question": "is a multi-lab pool worth more than the best single model",
                   "k": a.k, "max_new": a.max_new, "n_problems": len(probs),
                   "members": {}, }
    for tag, repo, lab in MODELS:
        f = os.path.join(HERE, OUT, f"{tag}__chat.json")
        if not os.path.isfile(f):
            print(f"  {tag:16s} no generations, skipped", flush=True)
            continue
        rows = json.load(open(f))
        kmin = min(len(c) for c in rows)
        m = execute_file([c[:kmin] for c in rows], probs, a.timeout, a.workers)
        ok[tag] = m
        res["members"][tag] = {
            "lab": lab, "repo": repo, "k": int(m.shape[1]),
            "pass_at_1": round(float(m.mean()), 4),
            "pass_at_k": round(float(m.any(1).mean()), 4)}
        v = res["members"][tag]
        print(f"  {tag:16s} {lab:9s} pass@1 {v['pass_at_1']:.4f}  "
              f"pass@{v['k']} {v['pass_at_k']:.4f}", flush=True)

    if len(ok) < 2:
        print("need at least two members for a union", flush=True)
        return
    tags = sorted(ok)
    solved = {t: ok[t].any(1) for t in tags}
    union = np.zeros_like(solved[tags[0]])
    for t in tags:
        union |= solved[t]
    best = max(res["members"][t]["pass_at_k"] for t in tags)
    best_tag = max(tags, key=lambda t: res["members"][t]["pass_at_k"])
    only = {t: int((solved[t] & ~np.logical_or.reduce(
        [solved[u] for u in tags if u != t])).sum()) for t in tags}
    res["union"] = {
        "union_pass": round(float(union.mean()), 4),
        "best_single_pass_at_k": round(best, 4),
        "best_single": best_tag,
        "union_minus_best": round(float(union.mean()) - best, 4),
        "mean_single_pass_at_1": round(float(np.mean(
            [res["members"][t]["pass_at_1"] for t in tags])), 4),
        "solved_by_exactly_one_model": only,
        "solved_by_none": int((~union).sum()),
        "pairwise_jaccard": {f"{x}|{y}": round(float((solved[x] & solved[y]).sum()
                                                     / max((solved[x] | solved[y]).sum(), 1)), 4)
                             for x, y in itertools.combinations(tags, 2)}}
    u = res["union"]
    print(f"\n  union {u['union_pass']:.4f}  best single {u['best_single_pass_at_k']:.4f} "
          f"({u['best_single']})  gain {u['union_minus_best']:+.4f}", flush=True)
    print(f"  solved by exactly one model: {u['solved_by_exactly_one_model']}", flush=True)
    print(f"  solved by none: {u['solved_by_none']}", flush=True)

    op = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(res, open(op, "w"), indent=2)
    print(f"\nwrote {op}", flush=True)


if __name__ == "__main__":
    main()
