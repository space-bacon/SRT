"""Metacognition-Bench AXIS-1 (vulnerability / trap_rate) via reconstructed
4-option multiple choice.

The official leaderboard RANKS on trap_rate: render each problem as
{correct, tempting-trap, 2 distractors}, and trap_rate = fraction of problems
where the model picks the trap option (lower is better; chance = 0.25). The
dataset ships only the free-form fields (prompt / expected_behavior /
hidden_trap), NOT ginigen's exact option sets, so we RECONSTRUCT the four
options with a strong external model (Claude) from expected_behavior (correct)
and hidden_trap (trap) plus two synthesized distractors. This is a faithful
reconstruction of the axis-1 methodology, not the official option set, so the
absolute trap_rate is indicative rather than leaderboard-official.

Two phases (cached):
  options - Claude builds {correct, trap, distractor1, distractor2} per task.
  mc      - gemma-4 answers the shuffled 4-option MC with a single letter.

Metrics: trap_rate (picked==trap), accuracy (picked==correct), by ticos_type /
grade. Creds from .env: API_KEY, API_BASE (Anthropic). Judge/option model via
--opt-model (default claude-opus-4-8).

Usage:
    set -a && source .env && set +a
    python scripts/metacog_mc_eval.py --model google/gemma-4-31B-it --n 300 \
        --opt-model claude-opus-4-8 --out artifacts/nla/gemma4/metacog_mc_gemma4.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import time
from collections import defaultdict

LETTERS = ["A", "B", "C", "D"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-31B-it")
    p.add_argument("--bench", default="")
    p.add_argument("--out", default="artifacts/nla/gemma4/metacog_mc_gemma4.json")
    p.add_argument("--options", default="", help="options cache jsonl (default <out>.options.jsonl)")
    p.add_argument("--phase", default="all", choices=["all", "options", "mc"])
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--gen-bs", type=int, default=16)
    p.add_argument("--opt-model", default="claude-opus-4-8")
    p.add_argument("--retries", type=int, default=4)
    return p.parse_args()


def find_bench(explicit: str) -> str:
    if explicit:
        return explicit
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hits = glob.glob(os.path.join(hf, "hub", "**", "metacog_bench.jsonl"), recursive=True)
    if not hits:
        raise SystemExit("metacog_bench.jsonl not found; pass --bench")
    return hits[0]


# ---------------- option construction (Claude) ----------------
OPT_SYS = (
    "You build fair 4-option multiple-choice items for a metacognition benchmark. "
    "Given a problem, its correct reasoning outcome, and the hidden trap (the "
    "seductive wrong answer), produce four SHORT answer options: the correct "
    "answer, the trap answer, and two plausible but clearly-wrong distractors. "
    "Options must be mutually exclusive, similar in length/style, and give no "
    "away-cue as to which is correct. Reply STRICT JSON only."
)
OPT_TMPL = """PROBLEM:
{prompt}

CORRECT REASONING OUTCOME (rubric):
{expected}

HIDDEN TRAP (the tempting wrong answer):
{trap}

Return STRICT JSON:
{{"correct": "<short correct answer>",
  "trap": "<short answer that embodies the hidden trap>",
  "distractors": ["<wrong 1>", "<wrong 2>"]}}
Each option <= 25 words. JSON only."""


def anthropic_json(base, key, model, sys, user, retries):
    import requests
    url = base.rstrip("/") + "/messages"
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    body = {"model": model, "max_tokens": 500, "system": sys,
            "messages": [{"role": "user", "content": user}]}
    last = ""
    for a in range(retries):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=120)
            if r.status_code != 200:
                last = f"http {r.status_code}: {r.text[:150]}"; time.sleep(2 ** a); continue
            t = r.json()["content"][0]["text"].strip()
            if t.startswith("```"):
                t = t.split("```", 2)[1].removeprefix("json").strip()
            return json.loads(t)
        except Exception as e:  # noqa: BLE001
            last = str(e)[:150]; time.sleep(2 ** a)
    return {"error": last}


def build_options(args, items, opt_path):
    base = os.environ.get("API_BASE", "https://api.anthropic.com/v1")
    key = os.environ.get("API_KEY", "")
    if not key:
        raise SystemExit("API_KEY missing (source .env)")
    done = {}
    if os.path.exists(opt_path):
        for l in open(opt_path):
            if l.strip():
                r = json.loads(l); done[r["task_id"]] = r
        print(f"  resume options: {len(done)} done", flush=True)
    t0 = time.time()
    with open(opt_path, "a") as f:
        for i, it in enumerate(items, 1):
            if it["task_id"] in done:
                continue
            o = anthropic_json(base, key, args.opt_model, OPT_SYS,
                               OPT_TMPL.format(prompt=it["prompt"],
                                               expected=it.get("expected_behavior", ""),
                                               trap=it.get("hidden_trap", "")), args.retries)
            rec = {"task_id": it["task_id"], "ticos_type": it.get("ticos_type"),
                   "grade": it.get("grade"), "prompt": it["prompt"], "options": o}
            f.write(json.dumps(rec) + "\n"); f.flush()
            if i % 20 == 0 or i == len(items):
                print(f"  options {i}/{len(items)} ({i/(time.time()-t0):.2f} it/s)", flush=True)


# ---------------- MC answering (gemma-4) ----------------
def run_mc(args, opt_path, out_path):
    import torch
    from transformers import Gemma4ForConditionalGeneration, AutoTokenizer

    opts = [json.loads(l) for l in open(opt_path) if l.strip()]
    opts = [o for o in opts if isinstance(o.get("options"), dict)
            and "correct" in o["options"] and "trap" in o["options"]
            and len(o["options"].get("distractors", [])) >= 2]
    print(f"MC over {len(opts)} items", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    rng = random.Random(0)
    rendered = []
    for o in opts:
        oo = o["options"]
        choices = [("correct", oo["correct"]), ("trap", oo["trap"]),
                   ("distractor", oo["distractors"][0]), ("distractor", oo["distractors"][1])]
        rng.shuffle(choices)
        letter_of = {"correct": None, "trap": None}
        lines = []
        for li, (kind, text) in enumerate(choices):
            lines.append(f"{LETTERS[li]}. {text}")
            if kind in letter_of and letter_of[kind] is None:
                letter_of[kind] = LETTERS[li]
        q = (o["prompt"] + "\n\n" + "\n".join(lines) +
             "\n\nAnswer with ONLY the single letter (A, B, C, or D) of the best answer.")
        rendered.append((o, q, letter_of["correct"], letter_of["trap"]))

    def build(q):
        return tok.apply_chat_template([{"role": "user", "content": q}],
                                       add_generation_prompt=True, tokenize=False)

    results = []
    t0 = time.time()
    for i in range(0, len(rendered), args.gen_bs):
        chunk = rendered[i:i + args.gen_bs]
        texts = [build(q) for (_, q, _, _) in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        outs = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for (o, q, cl, tl), ans in zip(chunk, outs):
            m = re.search(r"[ABCD]", ans.upper())
            picked = m.group(0) if m else "?"
            results.append({"task_id": o["task_id"], "ticos_type": o["ticos_type"],
                            "grade": o["grade"], "correct_letter": cl, "trap_letter": tl,
                            "picked": picked, "raw": ans.strip()[:40]})
        done = min(i + args.gen_bs, len(rendered))
        print(f"  mc {done}/{len(rendered)} ({done/(time.time()-t0):.2f} it/s)", flush=True)

    n = len(results)
    trap = sum(r["picked"] == r["trap_letter"] for r in results)
    corr = sum(r["picked"] == r["correct_letter"] for r in results)
    unparsed = sum(r["picked"] == "?" for r in results)

    def bd(key):
        g = defaultdict(lambda: [0, 0, 0])
        for r in results:
            k = r.get(key)
            g[k][0] += r["picked"] == r["trap_letter"]
            g[k][1] += r["picked"] == r["correct_letter"]
            g[k][2] += 1
        return {str(k): {"trap_rate": t / n_, "accuracy": c / n_, "n": n_}
                for k, (t, c, n_) in sorted(g.items())}

    summary = {"model": args.model, "n": n, "chance_trap": 0.25,
               "trap_rate": trap / n, "accuracy": corr / n, "unparsed": unparsed,
               "note": "Reconstructed 4-option MC (options built by Claude from "
                       "expected_behavior/hidden_trap); indicative of axis-1 "
                       "methodology, NOT ginigen's official option set.",
               "by_ticos_type": bd("ticos_type"), "by_grade": bd("grade")}
    json.dump({"summary": summary, "results": results}, open(out_path, "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nwrote {out_path}", flush=True)


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    opt_path = args.options or (args.out.rsplit(".", 1)[0] + ".options.jsonl")

    if args.phase in ("all", "options"):
        bench = find_bench(args.bench)
        items = [json.loads(l) for l in open(bench) if l.strip()][: args.n]
        print(f"bench {bench} ({len(items)} items) | option model {args.opt_model}", flush=True)
        build_options(args, items, opt_path)
        print(f"wrote options -> {opt_path}", flush=True)

    if args.phase in ("all", "mc"):
        run_mc(args, opt_path, args.out)


if __name__ == "__main__":
    main()
