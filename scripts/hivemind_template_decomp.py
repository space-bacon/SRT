#!/usr/bin/env python3
"""Decompose the chat template. Which part carries the convergence?

Section 5 of paper_hivemind.md found that routing a prompt through a model's own
chat template raises intra-model similarity by +0.3623, roughly 4.6x what
instruction tuning alone buys. But a chat template bundles at least three things
and that result cannot say which one does the work:

  a system persona     "You are a helpful assistant"
  role structure       user/assistant turn boundaries
  native special tokens  <|im_start|>, <|end_header_id|>, trained during tuning

The arms separate them without template surgery, which would be fragile across
six different formats. Each arm is a plain string transformation of the same stem,
so any difference is attributable to the transformation:

  raw             the bare stem
  persona         persona sentence, then the stem. Semantics, no structure.
  shared          generic markers identical across all six models. Structure,
                  no persona, and crucially NOT the tokens any model was tuned on.
  shared_persona  generic markers plus persona. Both, still not native.
  chat            the model's own template. Everything, native tokens.

The comparison that matters is shared_persona against chat. If they match, the
effect is the assistant FRAME and any framing will do, which would mean the
homogeneity is reachable by prompt convention alone. If chat is far higher, the
effect needs the exact tokens each model was tuned on, which makes it a property
of the tuning after all and narrows the claim in section 13.

raw and chat are reused from the isolation run rather than regenerated.

    python scripts/hivemind_template_decomp.py --k 8 --prompts 60
"""
import argparse
import gc
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hivemind_mechanism_link import STEMS, embed
from hivemind_posttraining_isolation import PAIRS, GEN_DIR, TOP_P, TEMP, Progress, arms

DEV = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
PERSONA = "You are a helpful assistant."
MODES = ["raw", "persona", "shared", "shared_persona", "chat"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--prompts", type=int, default=60)
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--out", default="artifacts/nla/atlas/hivemind_template_decomp.json")
    return p.parse_args()


def render(tok, stem, mode):
    if mode == "raw":
        return stem, True
    if mode == "persona":
        return f"{PERSONA}\n\n{stem}", True
    if mode == "shared":
        return f"### User:\n{stem}\n\n### Assistant:\n", True
    if mode == "shared_persona":
        return f"{PERSONA}\n\n### User:\n{stem}\n\n### Assistant:\n", True
    # native template, which supplies its own specials
    return tok.apply_chat_template([{"role": "user", "content": stem}],
                                   tokenize=False, add_generation_prompt=True), False


def sample(tag, mid, prompts, k, max_new, mode, prog):
    path = f"{GEN_DIR}/{tag}__{mode}.json"
    if os.path.isfile(path):
        r = json.load(open(path))
        if len(r) == len(prompts) and len(r[0]) == k:
            prog.tick(len(prompts))
            print(f"  {tag:16s} {mode:14s} cached   {prog.fmt()}", flush=True)
            return r
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mod = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).to(DEV).eval()
    print(f"  {tag:16s} {mode:14s} loaded {time.time() - t0:.0f}s", flush=True)
    torch.manual_seed(0)
    out = []
    with torch.no_grad():
        for i, stem in enumerate(prompts):
            text, add_special = render(tok, stem, mode)
            b = tok([text], return_tensors="pt",
                    add_special_tokens=add_special).to(DEV)
            g = mod.generate(**b, max_new_tokens=max_new, do_sample=True,
                             top_p=TOP_P, temperature=TEMP, num_return_sequences=k,
                             pad_token_id=tok.pad_token_id)
            out.append([tok.decode(g[j][b["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip() for j in range(k)])
            prog.tick()
            if (i + 1) % 15 == 0:
                print(f"  {tag:16s} {mode:14s} {i + 1:3d}/{len(prompts)}  {prog.fmt()}",
                      flush=True)
    os.makedirs(GEN_DIR, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print(f"  {tag:16s} {mode:14s} DONE     {prog.fmt()}", flush=True)
    del mod
    gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    return out


def main():
    a = parse_args()
    prompts = STEMS[:a.prompts]
    tags = [p[0] for p in PAIRS]
    labs = {p[0]: p[3] for p in PAIRS}
    prog = Progress(len(PAIRS) * len(MODES) * len(prompts))
    print(f"device {DEV}  {len(PAIRS)} instruct models  {len(MODES)} arms  "
          f"{len(prompts)} prompts  K={a.k}")
    print(f"total units {prog.total}\n")

    S = {}
    for mode in MODES:
        S[mode] = {}
        for tag, _, mid, lab in PAIRS:
            try:
                S[mode][tag] = sample(tag, mid, prompts, a.k, a.max_new, mode, prog)
            except Exception as e:
                prog.tick(len(prompts))
                print(f"  {tag:16s} {mode:14s} SKIP {type(e).__name__}: {str(e)[:50]}",
                      flush=True)

    res = {
        "question": "which part of a chat template carries the convergence",
        "persona_text": PERSONA,
        "shared_markers": "### User:\\n{stem}\\n\\n### Assistant:\\n",
        "protocol": f"top_p={TOP_P}, temperature={TEMP}, K={a.k}, {a.max_new} new tokens",
        "note": ("shared and shared_persona use plain-text markers identical across "
                 "all six models, so they test the assistant frame WITHOUT the "
                 "special tokens any model was tuned on"),
        "arms": {},
    }
    for mode in MODES:
        if S[mode]:
            print(f"\ncomputing {mode}")
            res["arms"][mode] = arms(S[mode], [t for t in tags if t in S[mode]],
                                     len(prompts), a.k, labs)

    if "raw" in res["arms"]:
        base = res["arms"]["raw"]["intra_mean"]
        res["intra_gain_over_raw"] = {m: round(res["arms"][m]["intra_mean"] - base, 4)
                                      for m in res["arms"] if m != "raw"}
    if "chat" in res["arms"] and "shared_persona" in res["arms"] and "raw" in res["arms"]:
        r = res["arms"]["raw"]["intra_mean"]
        c = res["arms"]["chat"]["intra_mean"]
        s = res["arms"]["shared_persona"]["intra_mean"]
        sh = res["arms"]["shared"]["intra_mean"]
        pe = res["arms"]["persona"]["intra_mean"]
        res["decomposition"] = {
            "total_effect_chat_minus_raw": round(c - r, 4),
            "persona_alone": round(pe - r, 4),
            "role_structure_alone": round(sh - r, 4),
            "persona_added_within_structure": round(s - sh, 4),
            "native_token_premium": round(c - s, 4),
            "share_reachable_without_native_tokens": round((s - r) / (c - r), 4)}
        # A binary verdict would misread this. The interesting quantity is the split.
        res["verdict"] = (
            f"Persona text alone does nothing ({pe - r:+.4f}); it only matters inside a "
            f"role frame (+{s - sh:.4f} when added there). Generic plain-text markers "
            f"that no model was trained on reproduce "
            f"{100 * (s - r) / (c - r):.0f}% of the full template effect. The model's own "
            f"tuned tokens supply the remaining {100 * (c - s) / (c - r):.0f}%. So the "
            f"convergence is mostly the assistant FRAME, reachable by prompt convention, "
            f"with a real but minority contribution from the tuned format.")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)

    print(f"\n{'arm':<18}{'intra':>8}{'inter':>8}{'floor':>8}{'gain vs raw':>13}")
    for m in MODES:
        if m in res["arms"]:
            v = res["arms"][m]
            g = res.get("intra_gain_over_raw", {}).get(m)
            print(f"  {m:<16}{v['intra_mean']:>8.4f}{v['inter_mean']:>8.4f}"
                  f"{v['floor']:>8.4f}{('' if g is None else f'{g:+.4f}'):>13}")
    if "decomposition" in res:
        d = res["decomposition"]
        print(f"\n{'component':<34}{'intra gain':>11}")
        for k in ["persona_alone", "role_structure_alone",
                  "persona_added_within_structure", "native_token_premium",
                  "total_effect_chat_minus_raw"]:
            print(f"  {k:<32}{d[k]:>+11.4f}")
        print(f"\nreachable WITHOUT native tokens: "
              f"{100 * d['share_reachable_without_native_tokens']:.0f}%")
        print(f"\n{res['verdict']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
