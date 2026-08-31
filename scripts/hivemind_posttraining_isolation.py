#!/usr/bin/env python3
"""Isolate post-training: matched base/instruct pairs, one protocol.

Section 4 of paper_hivemind.md found that base models reproduce the STRUCTURE of
arXiv:2510.22954 (a model's resamples are barely more distinctive than a rival's
output) at half the LEVEL, on a matched floor. Two explanations survive that:
post-training raises the level, or our models are simply smaller than theirs.

This separates them. Every model here is the instruct sibling of a base model
already measured, so pretrained weights, prompts, decoding and scorer are all held
fixed and only post-training moves.

  base intra 0.3644, inter 0.3401   (paper section 4, all 12 models)
  their instruct models             (>0.8 on 79% of queries, inter 0.71-0.82)

Two arms, because instruction tuning and chat formatting are different things and
conflating them would repeat the mistake that produced the greedy null:

  raw   the stem continued directly, byte-identical input to the base run, so the
        only difference from base is the weights
  chat  the stem as a user turn through the model's own chat template, which is
        how these models are actually used and how their study prompts them

The base side is RECOMPUTED over only the six matched tags. Comparing six instruct
models against all twelve base models would not be a controlled comparison.

    python scripts/hivemind_posttraining_isolation.py --k 8 --prompts 60
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
from hivemind_mechanism_link import STEMS, embed, spearman, pval

DEV = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
BASE_SAMPLES = "artifacts/nla/atlas/samples"
GEN_DIR = "artifacts/nla/atlas/samples_instruct"
TOP_P, TEMP = 0.9, 1.0

# base tag, base id, instruct id, lab
PAIRS = [
    ("qwen25_05b", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "Alibaba"),
    ("qwen3_06b", "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-0.6B", "Alibaba"),
    ("gemma2_2b", "google/gemma-2-2b", "google/gemma-2-2b-it", "Google"),
    ("llama32_1b", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct", "Meta"),
    ("smollm2_360m", "HuggingFaceTB/SmolLM2-360M",
     "HuggingFaceTB/SmolLM2-360M-Instruct", "HuggingFace"),
    ("olmo2_1b", "allenai/OLMo-2-0425-1B", "allenai/OLMo-2-0425-1B-Instruct", "AI2"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--prompts", type=int, default=60)
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--out", default="artifacts/nla/atlas/hivemind_posttraining_isolation.json")
    return p.parse_args()


class Progress:
    """Wall-clock ETA over the whole run, since per-model counters hide total time."""

    def __init__(self, total):
        self.total, self.done, self.t0 = total, 0, time.time()

    def tick(self, n=1):
        self.done += n

    def fmt(self):
        el = time.time() - self.t0
        if not self.done:
            return f"[   0/{self.total}  ETA --:--]"
        rate = self.done / el
        rem = (self.total - self.done) / rate if rate else 0
        return (f"[{self.done:4d}/{self.total} {100 * self.done / self.total:5.1f}%  "
                f"elapsed {int(el // 60)}:{int(el % 60):02d}  "
                f"ETA {int(rem // 60)}:{int(rem % 60):02d}  "
                f"{rate * 60:.1f}/min]")


def sample(tag, mid, prompts, k, max_new, mode, prog):
    path = f"{GEN_DIR}/{tag}__{mode}.json"
    if os.path.isfile(path):
        r = json.load(open(path))
        if len(r) == len(prompts) and len(r[0]) == k:
            prog.tick(len(prompts))
            print(f"  {tag:16s} {mode:4s} cached        {prog.fmt()}", flush=True)
            return r
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mod = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).to(DEV).eval()
    print(f"  {tag:16s} {mode:4s} loaded {time.time() - t_load:.0f}s", flush=True)
    torch.manual_seed(0)
    out = []
    with torch.no_grad():
        for i, pr in enumerate(prompts):
            if mode == "chat":
                text = tok.apply_chat_template([{"role": "user", "content": pr}],
                                               tokenize=False, add_generation_prompt=True)
                b = tok([text], return_tensors="pt", add_special_tokens=False).to(DEV)
            else:
                b = tok([pr], return_tensors="pt").to(DEV)
            g = mod.generate(**b, max_new_tokens=max_new, do_sample=True,
                             top_p=TOP_P, temperature=TEMP, num_return_sequences=k,
                             pad_token_id=tok.pad_token_id)
            out.append([tok.decode(g[j][b["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip() for j in range(k)])
            prog.tick()
            if (i + 1) % 10 == 0:
                print(f"  {tag:16s} {mode:4s} {i + 1:3d}/{len(prompts)}      {prog.fmt()}",
                      flush=True)
    os.makedirs(GEN_DIR, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print(f"  {tag:16s} {mode:4s} DONE          {prog.fmt()}", flush=True)
    del mod
    gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    return out


def arms(S, tags, n_prompts, k, labs):
    """intra, inter and a different-prompt floor, identical to the base protocol."""
    flat = [t for tg in tags for row in S[tg] for t in row]
    E = embed(flat).reshape(len(tags), n_prompts, k, -1)
    iu, ju = zip(*itertools.combinations(range(k), 2))
    intra = {}
    for m, tg in enumerate(tags):
        v = (E[m][:, iu, :] * E[m][:, ju, :]).sum(-1).mean(1)
        intra[tg] = {"mean": round(float(v.mean()), 4),
                     "frac_prompts_above_0.8": round(float((v > 0.8).mean()), 4)}
    inter, cross = [], []
    for n, (i, j) in enumerate(itertools.combinations(range(len(tags)), 2)):
        inter.append(float((E[i][:, :, None, :] * E[j][:, None, :, :]).sum(-1).mean()))
        if labs[tags[i]] != labs[tags[j]]:
            cross.append(n)
    inter = np.array(inter)
    rng = np.random.default_rng(0)
    F = E.reshape(-1, E.shape[-1])
    per = n_prompts * k
    p = rng.integers(0, len(F), 20000); q = rng.integers(0, len(F), 20000)
    ok = ((p % per) // k) != ((q % per) // k)
    im = np.array([intra[t]["mean"] for t in tags])
    return {"per_model": intra,
            "intra_mean": round(float(im.mean()), 4),
            "intra_range": [round(float(im.min()), 4), round(float(im.max()), 4)],
            "frac_models_any_prompt_above_0.8": round(
                float(np.mean([v["frac_prompts_above_0.8"] > 0 for v in intra.values()])), 4),
            "inter_mean": round(float(inter.mean()), 4),
            "inter_cross_lab": round(float(inter[cross].mean()), 4),
            "intra_minus_inter": round(float(im.mean() - inter.mean()), 4),
            "floor": round(float((F[p[ok]] * F[q[ok]]).sum(-1).mean()), 4)}


def main():
    a = parse_args()
    prompts = STEMS[:a.prompts]
    tags = [p[0] for p in PAIRS]
    labs = {p[0]: p[3] for p in PAIRS}
    print(f"device {DEV}  {len(PAIRS)} matched pairs  {len(prompts)} prompts  K={a.k}")
    prog = Progress(len(PAIRS) * 2 * len(prompts))
    print(f"total units {prog.total} (model x mode x prompt)\n")

    inst = {}
    for mode in ("raw", "chat"):
        inst[mode] = {}
        for tag, _, mid, lab in PAIRS:
            try:
                inst[mode][tag] = sample(tag, mid, prompts, a.k, a.max_new, mode, prog)
            except Exception as e:
                prog.tick(len(prompts))
                print(f"  {tag:16s} {mode:4s} SKIP {type(e).__name__}: {str(e)[:60]}",
                      flush=True)

    # Base side restricted to the SAME six models, or it is not a matched comparison.
    base = {t: json.load(open(f"{BASE_SAMPLES}/{t}.json")) for t in tags
            if os.path.isfile(f"{BASE_SAMPLES}/{t}.json")}
    print(f"\nbase arm: {len(base)} matched models")
    res = {
        "question": "does post-training produce the hivemind level",
        "design": "matched base/instruct pairs, identical prompts, decoding and scorer",
        "protocol": f"top_p={TOP_P}, temperature={TEMP}, K={a.k}, {a.max_new} new tokens",
        "pairs": [{"tag": p[0], "base": p[1], "instruct": p[2], "lab": p[3]} for p in PAIRS],
        "paper_reference": {"their_intra": "above 0.8 on 79% of queries",
                            "their_inter": [0.71, 0.82], "their_floor": [0.1, 0.2]},
    }
    print("\ncomputing base arm (6 matched models)")
    res["base"] = arms(base, [t for t in tags if t in base], len(prompts), a.k, labs)
    for mode in ("raw", "chat"):
        if inst[mode]:
            print(f"computing instruct arm ({mode})")
            res[f"instruct_{mode}"] = arms(inst[mode], [t for t in tags if t in inst[mode]],
                                           len(prompts), a.k, labs)

    b = res["base"]
    rows = [("base (6 matched)", b)]
    for mode in ("raw", "chat"):
        if f"instruct_{mode}" in res:
            rows.append((f"instruct {mode}", res[f"instruct_{mode}"]))
    res["delta"] = {}
    for name, v in rows[1:]:
        res["delta"][name.replace(" ", "_")] = {
            "intra": round(v["intra_mean"] - b["intra_mean"], 4),
            "inter": round(v["inter_mean"] - b["inter_mean"], 4),
            "floor": round(v["floor"] - b["floor"], 4)}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)

    print(f"\n{'arm':<20}{'intra':>8}{'inter':>8}{'i-i':>8}{'floor':>8}")
    for name, v in rows:
        print(f"  {name:<18}{v['intra_mean']:>8.4f}{v['inter_mean']:>8.4f}"
              f"{v['intra_minus_inter']:>8.4f}{v['floor']:>8.4f}")
    print(f"  {'THEIRS':<18}{'>0.80':>8}{'0.71-0.82':>8}{'~0':>8}{'0.1-0.2':>8}")
    print("\ndeltas vs matched base:")
    for k, v in res["delta"].items():
        print(f"  {k:<20} intra {v['intra']:+.4f}   inter {v['inter']:+.4f}   "
              f"floor {v['floor']:+.4f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
