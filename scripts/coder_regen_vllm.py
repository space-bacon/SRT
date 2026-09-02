"""Regenerate the coder ladder on one GPU with a budget that lets it finish.

The banked ladder was generated at max_new=192 and is 45% to 78% truncated
mid-fence, so those arms measure the token budget more than the model. See
`scripts/coder_regen_mlx.py` for the Apple Silicon version of the same fix.

This runs the whole ladder in one precision instead of mixing 4-bit MLX arms
with full-precision ones, so the rungs stay comparable to each other. Sampling
matches the MLX path: same HumanEval prompts, same chat template, same K,
temperature 1.0, top_p 0.9.

    python scripts/coder_regen_vllm.py --model Qwen/Qwen2.5-Coder-32B-Instruct --tag coder32B_inst
"""
from __future__ import annotations

import argparse
import json
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--prompts", type=int, default=164)
    ap.add_argument("--probs", default="data/humaneval.json")
    ap.add_argument("--out", default="artifacts/nla/coder_regen")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    ap.add_argument("--max-len", type=int, default=4096)
    a = ap.parse_args()

    path = os.path.join(a.out, f"{a.tag}__chat.json")
    os.makedirs(a.out, exist_ok=True)
    if os.path.isfile(path) and len(json.load(open(path))) >= a.prompts:
        print(f"{a.tag}: already complete", flush=True)
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    he = json.load(open(a.probs))[: a.prompts]
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}],
            add_generation_prompt=True, tokenize=False)
        for p in he
    ]

    llm = LLM(model=a.model, dtype=a.dtype, gpu_memory_utilization=a.gpu_frac,
              max_model_len=a.max_len, trust_remote_code=True)
    sp = SamplingParams(n=a.k, temperature=a.temp, top_p=a.top_p,
                        max_tokens=a.max_new)

    print(f"{a.tag}: {len(prompts)} problems, K={a.k}, max_new={a.max_new}",
          flush=True)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    rows = [[c.text for c in o.outputs] for o in outs]

    json.dump(rows, open(path, "w"))
    n = sum(len(r) for r in rows)
    unclosed = sum(1 for r in rows for c in r if c.count("```") % 2 == 1)
    print(f"wrote {path}  {time.time() - t0:.0f}s  {n} candidates  "
          f"unclosed {unclosed / max(1, n):.1%}", flush=True)


if __name__ == "__main__":
    main()
