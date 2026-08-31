"""Cross-lab suppression: does the persona lever work across vendors, not just scale?

Section 5.5 measured suppression twice. On six models of 0.36B to 2B, where the
inter-model term is genuinely cross-lab, and on three Ministral rungs, where it is
cross-scale inside one lab. Neither is a cross-lab test at deployable size.

This runs the same arms over four instruct models from four labs at 14B to 31B, so
the inter-model term crosses a vendor boundary at a size people actually ship.

    python scripts/crosslab_suppression.py --shard 0 --shards 4 --gen-only
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
from hivemind_gen import Progress, run_model  # noqa: E402
from ministral_ladder import STEMS  # noqa: E402
from ministral_suppression import DEPLOYED, PERSONAS, embed, snap, stats  # noqa: E402

OUT = "artifacts/nla/crosslab_suppression"
MODELS = [
    ("qwen25c_14b", "Qwen/Qwen2.5-Coder-14B-Instruct", "Alibaba"),
    ("ministral_14b", "mistralai/Ministral-3-14B-Instruct-2512", "Mistral"),
    ("mistralsmall_24b", "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "Mistral"),
    ("gemma4_31b", "google/gemma-4-31B-it", "Google"),
]
MODES = ["chat", "persona_model", "persona_sample", "deployed_model", "deployed_sample"]


def make_render(model_idx, k):
    def render(tok, stem, mode):
        if mode == "chat":
            people, per_sample = [None], False
        else:
            pool = PERSONAS if mode.startswith("persona") else DEPLOYED
            if mode.endswith("_model"):
                people, per_sample = [pool[model_idx % len(pool)]], False
            else:
                people = [pool[j % len(pool)] for j in range(k)]
                per_sample = True
        outs = []
        for p in people:
            msgs = ([{"role": "user", "content": stem}] if p is None else
                    [{"role": "system", "content": p}, {"role": "user", "content": stem}])
            try:
                o = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
            except Exception:
                # Some templates reject a system role; fold the persona into the turn.
                o = tok.apply_chat_template(
                    [{"role": "user", "content": stem if p is None else f"{p}\n\n{stem}"}],
                    tokenize=True, add_generation_prompt=True)
            outs.append(o["input_ids"] if hasattr(o, "keys") else o)
        return (outs if per_sample else outs[0]), False, per_sample
    return render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--prompts", type=int, default=60)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--gen-only", action="store_true")
    ap.add_argument("--out", default=f"{OUT}/results.json")
    a = ap.parse_args()

    prompts = STEMS[:a.prompts]
    mine = [(i, m) for i, m in enumerate(MODELS) if i % a.shards == a.shard]
    print(f"shard {a.shard}/{a.shards}  {len(mine)} models  {len(prompts)} prompts  "
          f"K={a.k}", flush=True)

    prog = Progress(len(mine) * len(MODES) * len(prompts))
    for idx, (tag, repo, lab) in mine:
        p = snap(repo)
        if not p:
            print(f"  {tag} MISSING {repo}", flush=True)
            continue
        run_model(tag, p, prompts, MODES, make_render(idx, a.k),
                  os.path.join(HERE, OUT), prog, k=a.k, max_new=a.max_new,
                  batch=a.batch, dtype=torch.bfloat16)
    if a.gen_only:
        return

    rng = np.random.default_rng(0)
    res = {"question": "does persona suppression work across labs at deployable size",
           "models": {t: {"repo": r, "lab": l} for t, r, l in MODELS},
           "arms": {}}
    for mode in MODES:
        per = {}
        for tag, _, _ in MODELS:
            f = os.path.join(HERE, OUT, f"{tag}__{mode}.json")
            if not os.path.isfile(f):
                break
            g = json.load(open(f))
            flat = [t if t.strip() else " " for row in g for t in row]
            per[tag] = embed(flat).reshape(len(g), len(g[0]), -1)
        if len(per) == len(MODELS):
            res["arms"][mode] = stats(per, rng)
            v = res["arms"][mode]
            print(f"  {mode:16s} intra {v['intra_mean']:.4f}  inter {v['inter_mean']:.4f}  "
                  f"floor {v['floor']:.4f}", flush=True)

    if "chat" in res["arms"]:
        b = res["arms"]["chat"]
        res["suppression"] = {
            m: {"intra_drop": round(b["intra_mean"] - res["arms"][m]["intra_mean"], 4),
                "inter_drop": round(b["inter_mean"] - res["arms"][m]["inter_mean"], 4)}
            for m in MODES[1:] if m in res["arms"]}
        for m, v in res["suppression"].items():
            print(f"  {m:16s} intra drop {v['intra_drop']:+.4f}  "
                  f"inter drop {v['inter_drop']:+.4f}", flush=True)

    op = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(res, open(op, "w"), indent=2)
    print(f"\nwrote {op}", flush=True)


if __name__ == "__main__":
    main()
