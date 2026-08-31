#!/usr/bin/env python3
"""Can varied framing drive the convergence back DOWN?

Everything so far adds framing and watches similarity rise. Section 5.1 showed the
assistant turn structure is what does it. That is an explanation, not a remedy,
and the question anyone trying to fix this would actually ask is the reverse one:
if the shared frame causes the convergence, does breaking the frame undo it?

Two directions, because they are different problems with different fixes.

  persona_model    each MODEL gets a different persona, all 8 of its samples share
                   it. If inter-model similarity falls, a deployer can differentiate
                   their product from a competitor's by changing one string.

  persona_sample   each of the 8 SAMPLES gets a different persona within one model.
                   If intra-model similarity falls, a single model can be made to
                   explore, which is the diversity problem the original paper
                   raises about resampling being useless.

Baseline is the native chat arm at intra 0.7272 and inter 0.6464, reused rather
than regenerated. Personas are eight genuinely different voices, not eight
rewordings of "helpful assistant", since section 5.1 showed a neutral persona
contributes nothing and near-identical personas would test nothing.

Models whose template rejects a system role, such as gemma-2, get the persona
prepended to the user turn instead. The frame is what matters, not the slot.

    python scripts/hivemind_suppression.py --k 8 --prompts 60
"""
import argparse
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hivemind_mechanism_link import STEMS
from hivemind_posttraining_isolation import PAIRS, GEN_DIR, TOP_P, TEMP, Progress, arms

DEV = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")

PERSONAS = [
    "You are a blunt, opinionated essayist who despises cliches.",
    "You are a working plumber giving plain practical advice.",
    "You are a poet who answers in vivid concrete images.",
    "You are a skeptical scientist who hedges every claim carefully.",
    "You are an enthusiastic teenager texting a close friend.",
    "You are a nineteenth-century letter writer, formal and unhurried.",
    "You are a terse expert who answers in one short sentence.",
    "You are a stand-up comedian working the answer into a bit.",
]

# What the industry actually ships. Every one is a variation on helpful assistant,
# which is the point: this is the persona diversity that exists in the wild, and
# the exotic set above is an upper bound nobody deploys.
DEPLOYED = [
    "You are a helpful, harmless, and honest AI assistant.",
    "You are a large language model. Answer accurately and concisely.",
    "You are an AI assistant. Provide clear, accurate and helpful responses.",
    "You are a friendly and knowledgeable assistant. Be concise but thorough.",
    "You are an AI assistant designed to be maximally helpful while remaining safe.",
    "You are a helpful assistant. Answer directly and avoid unnecessary preamble.",
    "You are an AI language model. Respond helpfully and without verbosity.",
    "You are a thoughtful assistant. Give balanced, well-reasoned answers.",
]
SETS = {"persona": PERSONAS, "deployed": DEPLOYED}
MODES = ["chat", "persona_model", "persona_sample", "deployed_model", "deployed_sample"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--prompts", type=int, default=60)
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--out", default="artifacts/nla/atlas/hivemind_suppression.json")
    return p.parse_args()


def build(tok, stem, persona):
    """Persona in the system slot, or folded into the user turn if unsupported."""
    if persona is None:
        msgs = [{"role": "user", "content": stem}]
    else:
        msgs = [{"role": "system", "content": persona}, {"role": "user", "content": stem}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return tok.apply_chat_template(
            [{"role": "user", "content": f"{persona}\n\n{stem}"}],
            tokenize=False, add_generation_prompt=True)


def sample(tag, mid, idx, prompts, k, max_new, mode, prog):
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
    tok.padding_side = "left"
    mod = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).to(DEV).eval()
    print(f"  {tag:16s} {mode:14s} loaded {time.time() - t0:.0f}s", flush=True)
    torch.manual_seed(0)
    pool = SETS.get(mode.split("_")[0])
    out = []
    with torch.no_grad():
        for i, stem in enumerate(prompts):
            if mode.endswith("_sample"):
                # k prompts, one persona each, one sample apiece
                texts = [build(tok, stem, pool[j % len(pool)]) for j in range(k)]
                b = tok(texts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(DEV)
                g = mod.generate(**b, max_new_tokens=max_new, do_sample=True,
                                 top_p=TOP_P, temperature=TEMP,
                                 num_return_sequences=1, pad_token_id=tok.pad_token_id)
            else:
                p = pool[idx % len(pool)] if pool else None
                b = tok([build(tok, stem, p)], return_tensors="pt",
                        add_special_tokens=False).to(DEV)
                g = mod.generate(**b, max_new_tokens=max_new, do_sample=True,
                                 top_p=TOP_P, temperature=TEMP,
                                 num_return_sequences=k, pad_token_id=tok.pad_token_id)
            cut = b["input_ids"].shape[1]
            out.append([tok.decode(g[j][cut:], skip_special_tokens=True).strip()
                        for j in range(g.shape[0])])
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
    print(f"device {DEV}  {len(PAIRS)} models  {len(MODES)} arms  "
          f"{len(prompts)} prompts  K={a.k}")
    print(f"total units {prog.total}\n")

    S = {}
    for mode in MODES:
        S[mode] = {}
        for idx, (tag, _, mid, lab) in enumerate(PAIRS):
            try:
                S[mode][tag] = sample(tag, mid, idx, prompts, a.k, a.max_new, mode, prog)
            except Exception as e:
                prog.tick(len(prompts))
                print(f"  {tag:16s} {mode:14s} SKIP {type(e).__name__}: {str(e)[:50]}",
                      flush=True)

    res = {"question": "does varied framing suppress the convergence",
           "exotic_personas": PERSONAS,
           "deployed_personas": DEPLOYED,
           "why_two_sets": ("exotic voices are an upper bound nobody ships; the "
                            "deployed set is what the industry actually uses, every "
                            "one a variation on helpful assistant"),
           "native_template_note": ("only Qwen2.5 and SmolLM2 inject a branded persona "
                                    "by default; Qwen3, gemma-2 and OLMo-2 inject no "
                                    "system prompt at all, and Llama-3.2 injects only "
                                    "date metadata. They converge regardless."),
           "protocol": f"top_p={TOP_P}, temperature={TEMP}, K={a.k}, {a.max_new} tokens",
           "baseline": "the native chat arm from section 5",
           "arms": {}}
    for mode in MODES:
        if S[mode]:
            print(f"\ncomputing {mode}")
            res["arms"][mode] = arms(S[mode], [t for t in tags if t in S[mode]],
                                     len(prompts), a.k, labs)

    if "chat" in res["arms"]:
        b = res["arms"]["chat"]
        res["suppression"] = {}
        for m in res["arms"]:
            if m == "chat":
                continue
            v = res["arms"][m]
            res["suppression"][m] = {
                "intra_delta": round(v["intra_mean"] - b["intra_mean"], 4),
                "inter_delta": round(v["inter_mean"] - b["inter_mean"], 4),
                "floor_delta": round(v["floor"] - b["floor"], 4)}
        pm = res["suppression"].get("persona_model", {})
        ps = res["suppression"].get("persona_sample", {})
        dm = res["suppression"].get("deployed_model", {})
        ds = res["suppression"].get("deployed_sample", {})
        res["verdict"] = (
            f"Against a chat baseline of intra {b['intra_mean']} and inter "
            f"{b['inter_mean']}: exotic personas move intra by "
            f"{ps.get('intra_delta', 0):+.4f} per sample and inter by "
            f"{pm.get('inter_delta', 0):+.4f} per model. The personas the industry "
            f"actually deploys move intra by {ds.get('intra_delta', 0):+.4f} and inter "
            f"by {dm.get('inter_delta', 0):+.4f}. The gap between those two is how much "
            f"achievable diversity the field is leaving on the table.")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)

    print(f"\n{'arm':<18}{'intra':>8}{'inter':>8}{'floor':>8}{'d intra':>10}{'d inter':>10}")
    for m in MODES:
        if m not in res["arms"]:
            continue
        v = res["arms"][m]
        s = res.get("suppression", {}).get(m, {})
        di = f"{s['intra_delta']:+.4f}" if "intra_delta" in s else ""
        de = f"{s['inter_delta']:+.4f}" if "inter_delta" in s else ""
        print(f"  {m:<16}{v['intra_mean']:>8.4f}{v['inter_mean']:>8.4f}"
              f"{v['floor']:>8.4f}{di:>10}{de:>10}")
    if "verdict" in res:
        print(f"\n{res['verdict']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
