"""Regenerate the full 36-arm HumanEval format matrix with a budget that lets it finish.

The banked matrix was generated at max_new=192, and the budget did not bind
equally across arms: wrapper formats truncate about twice as often as raw
(shared_persona 70.0%, chat 66.4%, shared 62.4%, persona 55.1%, raw 30.6%).
That is the same axis the format effect is measured along, so the banked
comparison cannot separate "this format helps" from "this format ran out of
tokens first".

Arms, prompts, chat template, K, top_p and temperature all match
`scripts/coder_ladder.py`. Precision matches too (bfloat16). Only `max_new`
differs.

    python scripts/coder_matrix_vllm.py --sizes 0.5B 1.5B --max-new 1024
"""
from __future__ import annotations

import argparse
import json
import os
import time

PERSONA = "You are a helpful assistant."
SIZES = ["0.5B", "1.5B", "3B", "7B", "14B", "32B"]
BASE_ARMS = ["raw"]
INST_ARMS = ["raw", "persona", "shared", "shared_persona", "chat"]


def render(tok, stem: str, mode: str) -> tuple[str, bool]:
    """Same five arms as section 5.1. Returns (text, add_special_tokens)."""
    if mode == "raw":
        return stem, True
    if mode == "persona":
        return f"{PERSONA}\n\n{stem}", True
    if mode == "shared":
        return f"### User:\n{stem}\n\n### Assistant:\n", True
    if mode == "shared_persona":
        return f"{PERSONA}\n\n### User:\n{stem}\n\n### Assistant:\n", True
    # The chat template already emits its own specials, so adding more corrupts it.
    return tok.apply_chat_template([{"role": "user", "content": stem}],
                                   tokenize=False, add_generation_prompt=True), False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", nargs="*", default=SIZES)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--prompts", type=int, default=164)
    ap.add_argument("--probs", default="data/humaneval.json")
    ap.add_argument("--prefix", default="coder")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--out", default="artifacts/nla/coder_matrix1024")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    ap.add_argument("--max-len", type=int, default=4096)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    he = json.load(open(a.probs))[: a.prompts]
    stems = [p["prompt"] for p in he]
    os.makedirs(a.out, exist_ok=True)
    # Kept out of `out` because the pass-matrix globber treats every *.json with a
    # "__" in its name as an arm.
    meta_dir = os.path.join(a.out, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    sp = SamplingParams(n=a.k, temperature=a.temp, top_p=a.top_p,
                        max_tokens=a.max_new)

    jobs = []
    inst_arms = a.arms or INST_ARMS
    base_arms = [m for m in (a.arms or BASE_ARMS) if m in BASE_ARMS]
    for s in a.sizes:
        if base_arms:
            jobs.append((f"Qwen/Qwen2.5-Coder-{s}", f"{a.prefix}{s}_base", base_arms))
        jobs.append((f"Qwen/Qwen2.5-Coder-{s}-Instruct", f"{a.prefix}{s}_inst", inst_arms))

    for mid, tag, arms in jobs:
        todo = [m for m in arms
                if not os.path.isfile(os.path.join(a.out, f"{tag}__{m}.json"))]
        if not todo:
            print(f"{tag}: all arms cached", flush=True)
            continue

        tok = AutoTokenizer.from_pretrained(mid)
        llm = LLM(model=mid, dtype=a.dtype, gpu_memory_utilization=a.gpu_frac,
                  max_model_len=a.max_len, trust_remote_code=True)

        for mode in todo:
            t0 = time.time()
            texts, add_special = zip(*(render(tok, s, mode) for s in stems))
            ids = tok(list(texts), add_special_tokens=add_special[0])["input_ids"]
            outs = llm.generate([TokensPrompt(prompt_token_ids=i) for i in ids], sp)
            rows = [[c.text for c in o.outputs] for o in outs]

            # finish_reason == "length" is the only ground truth for truncation;
            # fence parity misreads arms that answer in bare code.
            reasons = [c.finish_reason for o in outs for c in o.outputs]
            n = len(reasons)
            cut = sum(1 for r in reasons if r == "length")

            p = os.path.join(a.out, f"{tag}__{mode}.json")
            json.dump(rows, open(p, "w"))
            json.dump({"tag": tag, "arm": mode, "model": mid, "k": a.k,
                       "max_new": a.max_new, "n_candidates": n,
                       "truncated": cut, "truncated_frac": round(cut / max(1, n), 4)},
                      open(os.path.join(meta_dir, f"{tag}.{mode}.meta.json"), "w"))
            print(f"wrote {tag}__{mode}  {time.time() - t0:.0f}s  {n} cands  "
                  f"truncated {cut / max(1, n):.1%}", flush=True)

        del llm
        import gc

        import torch
        gc.collect()
        torch.cuda.empty_cache()

    print("matrix complete", flush=True)


if __name__ == "__main__":
    main()
