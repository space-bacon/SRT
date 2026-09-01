#!/usr/bin/env python3
"""Frontier generation through vLLM, using each vendor's own turn format.

`device_map="auto"` splits a model into sequential pipeline stages, so three of
four cards idle at any instant, and for FP8 checkpoints spanning devices in one
process it also forces the slow kernel path. Tensor parallelism keeps every card
computing on every token, which is what these models want.

Two of the three frontier models are not in vLLM's registry at 0.28.0
(`Glm5NextForConditionalGeneration`, `Qwen4ExpForConditionalGeneration`), so this
covers the one that is.

DeepSeek-V4 ships no HF `chat_template`. It ships something better: a reference
encoder at `encoding/encoding_dsv4.py` that defines the deployed format exactly.
Using it is the vendor's format, not one we invented, which is the distinction
that decides whether a `chat` arm measures anything. Its `chat` mode closes the
thinking block immediately, so the budget goes to the answer rather than to
reasoning that we then truncate.

Truncation is reported, never assumed. A frontier model that spends its budget
before emitting code scores near zero for a reason that is not capability.

    python scripts/vllm_frontier_gen.py --model deepseek-ai/DeepSeek-V4-Flash --tp 4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUBS = [h for h in (os.environ.get("HF_HOME", "") and
                    os.path.join(os.environ["HF_HOME"], "hub"),
                    "/workspace/.hf_home/hub", "/root/.hf_home/hub") if h]
INSTR = "Complete the following Python function. Reply with the full function only.\n\n"


def snap(repo):
    stem = f"models--{repo.replace('/', '--')}/snapshots/*/"
    for hub in HUBS:
        g = glob.glob(f"{hub}/{stem}")
        if g:
            return g[0]
    return None


def build_prompts(path, stems):
    """Vendor's own turn format, preferring a shipped encoder over the HF template."""
    enc = os.path.join(path, "encoding")
    if os.path.isfile(os.path.join(enc, "encoding_dsv4.py")):
        sys.path.insert(0, enc)
        from encoding_dsv4 import encode_messages
        return [encode_messages([{"role": "user", "content": INSTR + s}],
                                thinking_mode="chat") for s in stems], "encoding_dsv4/chat"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    return [tok.apply_chat_template([{"role": "user", "content": INSTR + s}],
                                    tokenize=False, add_generation_prompt=True)
            for s in stems], "hf_chat_template"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--probs", default="data/humaneval.json")
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    ap.add_argument("--out-dir", default="artifacts/nla/frontier")
    a = ap.parse_args()

    path = snap(a.model)
    if path is None:
        print(f"MISSING {a.model}", flush=True)
        return 1
    tag = a.tag or a.model.split("/")[-1].replace(".", "").replace("-", "_").lower()

    probs = json.load(open(os.path.join(HERE, a.probs)))
    stems = [p["prompt"] for p in probs]
    prompts, fmt = build_prompts(path, stems)
    print(f"{tag}  {len(prompts)} prompts  K={a.k}  max_new={a.max_new}  format={fmt}", flush=True)
    print(f"  first prompt: {prompts[0][:120]!r}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=path, tensor_parallel_size=a.tp, trust_remote_code=True,
              gpu_memory_utilization=a.gpu_frac, max_model_len=4096)
    sp = SamplingParams(n=a.k, temperature=1.0, top_p=0.9, max_tokens=a.max_new)
    outs = llm.generate(prompts, sp)

    rows, truncated, total = [], 0, 0
    for o in outs:
        rows.append([c.text for c in o.outputs])
        for c in o.outputs:
            total += 1
            truncated += (c.finish_reason == "length")

    os.makedirs(os.path.join(HERE, a.out_dir), exist_ok=True)
    out = os.path.join(HERE, a.out_dir, f"{tag}__chat.json")
    json.dump(rows, open(out, "w"))
    frac = truncated / total if total else 0.0
    json.dump({"model": a.model, "tag": tag, "format": fmt, "k": a.k,
               "max_new": a.max_new, "tensor_parallel": a.tp,
               "n_prompts": len(prompts), "n_samples": total,
               "truncated": truncated, "truncated_frac": round(frac, 4),
               "reading": "a high truncated_frac means the budget, not the model, "
                          "is what any low pass rate measures"},
              open(os.path.join(HERE, a.out_dir, f"{tag}__meta.json"), "w"), indent=2)
    print(f"\n  {out}")
    print(f"  truncated {truncated}/{total} = {frac:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
