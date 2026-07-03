"""Metacognition-Bench (ginigen-ai) evaluation with an Anthropic LLM judge.

The ginigen Metacognition leaderboard scores a model on 300 free-form reasoning
TRAPS: each item embeds a seductive-but-wrong line of reasoning, and a
metacognitive model must NOT fall for it. This harness produces the baseline
trap-escape score for a target model, which is the ground the "improve from
there" claim needs.

Two phases (both run by default; use --phase to split):
  generate - target model answers each `prompt` (chat template, greedy). Saved to
             a rows file so judging can be re-run without regeneration.
  judge    - Anthropic Claude grades each answer against the item's
             `expected_behavior` + `hidden_trap`, returning strict JSON
             {escaped, correct, score, reason}.

Aggregates: overall trap-escape rate + mean score, broken down by ticos_type,
grade, and difficulty tertile.

Judge creds from env (source .env first): API_KEY, API_BASE, MODEL_NAME
(Anthropic native /messages endpoint).

Usage (venv transformers>=5):
    set -a && source .env && set +a
    python scripts/metacog_bench_eval.py --model google/gemma-4-31B-it --n 300 \
        --out artifacts/nla/gemma4/metacog_bench_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

BENCH_GLOB = "*Metacognition-Bench*/**/metacog_bench.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-31B-it")
    p.add_argument("--bench", default="", help="path to metacog_bench.jsonl (auto if empty)")
    p.add_argument("--out", default="artifacts/nla/gemma4/metacog_bench_eval.json")
    p.add_argument("--rows", default="", help="answers jsonl (default: <out>.rows.jsonl)")
    p.add_argument("--phase", default="all", choices=["all", "generate", "judge"])
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--gen-bs", type=int, default=8)
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--judge-model", default="", help="override MODEL_NAME env")
    p.add_argument("--judge-retries", type=int, default=4)
    return p.parse_args()


def find_bench(explicit: str) -> str:
    if explicit:
        return explicit
    import glob
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hits = glob.glob(os.path.join(hf, "hub", "**", "metacog_bench.jsonl"), recursive=True)
    if not hits:
        raise SystemExit("metacog_bench.jsonl not found; pass --bench")
    return hits[0]


def load_bench(path: str, n: int) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:n]


# ---------------- generation ----------------
def generate_answers(args, items: list[dict], rows_path: str) -> None:
    import torch
    from transformers import Gemma4ForConditionalGeneration, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    def build(prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    t0 = time.time()
    with open(rows_path, "w") as out:
        for i in range(0, len(items), args.gen_bs):
            chunk = items[i:i + args.gen_bs]
            texts = [build(it["prompt"]) for it in chunk]
            enc = tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            new = gen[:, enc.input_ids.shape[1]:]
            answers = tok.batch_decode(new, skip_special_tokens=True)
            for it, ans in zip(chunk, answers):
                rec = {"task_id": it.get("task_id"), "domain": it.get("domain"),
                       "grade": it.get("grade"), "ticos_type": it.get("ticos_type"),
                       "difficulty": it.get("difficulty"), "prompt": it["prompt"],
                       "expected_behavior": it.get("expected_behavior", ""),
                       "hidden_trap": it.get("hidden_trap", ""),
                       "answer": ans.strip()}
                out.write(json.dumps(rec) + "\n")
            done = min(i + args.gen_bs, len(items))
            print(f"  generated {done}/{len(items)} "
                  f"({done/(time.time()-t0):.2f} it/s)", flush=True)


# ---------------- judging ----------------
JUDGE_SYS = (
    "You are a strict, careful grader for a metacognition benchmark. Each item "
    "contains a reasoning TRAP: a seductive but wrong line of reasoning. A good "
    "metacognitive answer must NOT fall for the trap and must satisfy the stated "
    "expected behavior (reach the correct conclusion with sound reasoning). Judge "
    "only the substance, not style or verbosity. Reply with STRICT JSON only."
)

JUDGE_TMPL = """PROBLEM:
{prompt}

THE HIDDEN TRAP (what a weak model falls for):
{trap}

EXPECTED CORRECT BEHAVIOR (rubric for a passing answer):
{expected}

MODEL'S ANSWER:
{answer}

Grade the model's answer. Return STRICT JSON with keys:
  "escaped_trap": true/false  (did it avoid the specific wrong reasoning?)
  "correct": true/false       (does it satisfy the expected correct behavior?)
  "score": 0.0-1.0            (partial credit for correctness/soundness)
  "reason": "<= 40 words"
JSON only, no prose."""


def judge_one(base: str, key: str, model: str, item: dict, retries: int) -> dict:
    import requests
    url = base.rstrip("/") + "/messages"
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    user = JUDGE_TMPL.format(prompt=item["prompt"], trap=item.get("hidden_trap", ""),
                             expected=item.get("expected_behavior", ""),
                             answer=item["answer"][:8000])
    body = {"model": model, "max_tokens": 400, "system": JUDGE_SYS,
            "messages": [{"role": "user", "content": user}]}
    last = ""
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=120)
            if r.status_code != 200:
                last = f"http {r.status_code}: {r.text[:200]}"
                time.sleep(2 ** attempt)
                continue
            text = r.json()["content"][0]["text"].strip()
            if text.startswith("```"):
                text = text.split("```", 2)[1].removeprefix("json").strip()
            v = json.loads(text)
            return {"escaped_trap": bool(v.get("escaped_trap", False)),
                    "correct": bool(v.get("correct", False)),
                    "score": float(v.get("score", 0.0)),
                    "reason": str(v.get("reason", ""))[:300]}
        except Exception as e:  # noqa: BLE001
            last = str(e)[:200]
            time.sleep(2 ** attempt)
    return {"escaped_trap": False, "correct": False, "score": 0.0,
            "error": last, "reason": "JUDGE_FAILED"}


def judge_answers(args, rows_path: str) -> list[dict]:
    base = os.environ.get("API_BASE", "https://api.anthropic.com/v1")
    key = os.environ.get("API_KEY", "")
    model = args.judge_model or os.environ.get("MODEL_NAME", "")
    if not key or not model:
        raise SystemExit("judge creds missing: set API_KEY + MODEL_NAME (source .env)")
    print(f"judge: {model} @ {base}", flush=True)
    rows = [json.loads(l) for l in open(rows_path) if l.strip()]
    # resume: verdicts are persisted incrementally so a crash never re-pays API
    judged_path = rows_path.rsplit(".", 1)[0] + ".judged.jsonl"
    done: dict = {}
    if os.path.exists(judged_path):
        for l in open(judged_path):
            if l.strip():
                jr = json.loads(l); done[jr["task_id"]] = jr
        print(f"  resume: {len(done)} already judged", flush=True)
    out = []
    t0 = time.time()
    with open(judged_path, "a") as jf:
        for i, it in enumerate(rows, 1):
            if it.get("task_id") in done:
                out.append(done[it["task_id"]]); continue
            v = judge_one(base, key, model, it, args.judge_retries)
            it["verdict"] = v
            jf.write(json.dumps(it) + "\n"); jf.flush()
            out.append(it)
            if i % 10 == 0 or i == len(rows):
                passed = sum(x["verdict"]["correct"] for x in out)
                print(f"  judged {i}/{len(rows)} | running pass {passed/i:.3f} "
                      f"({i/(time.time()-t0):.2f} it/s)", flush=True)
    return out


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(r["verdict"]["correct"] for r in rows)
    escaped = sum(r["verdict"]["escaped_trap"] for r in rows)
    score = sum(r["verdict"]["score"] for r in rows) / max(1, n)
    failed = sum(1 for r in rows if r["verdict"].get("reason") == "JUDGE_FAILED")

    def breakdown(key):
        g = defaultdict(lambda: [0, 0])
        for r in rows:
            k = r.get(key)
            g[k][0] += r["verdict"]["correct"]; g[k][1] += 1
        return {str(k): {"pass_rate": c / t, "n": t} for k, (c, t) in sorted(g.items())}

    # difficulty tertiles (difficulty may be str or float in the bench)
    def as_float(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0
    diffs = sorted(as_float(r.get("difficulty")) for r in rows)
    lo, hi = (diffs[len(diffs)//3], diffs[2*len(diffs)//3]) if diffs else (0.33, 0.66)
    dt = defaultdict(lambda: [0, 0])
    for r in rows:
        d = as_float(r.get("difficulty"))
        b = "easy" if d <= lo else ("hard" if d > hi else "medium")
        dt[b][0] += r["verdict"]["correct"]; dt[b][1] += 1

    return {"n": n, "trap_escape_rate": escaped / max(1, n),
            "pass_rate": correct / max(1, n), "mean_score": score,
            "judge_failed": failed,
            "by_ticos_type": breakdown("ticos_type"),
            "by_grade": breakdown("grade"),
            "by_difficulty": {k: {"pass_rate": c/t, "n": t} for k, (c, t) in dt.items()}}


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows_path = args.rows or (args.out.rsplit(".", 1)[0] + ".rows.jsonl")

    if args.phase in ("all", "generate"):
        bench = find_bench(args.bench)
        items = load_bench(bench, args.n)
        print(f"bench: {bench} ({len(items)} items) | model {args.model}", flush=True)
        generate_answers(args, items, rows_path)
        print(f"wrote answers -> {rows_path}", flush=True)

    if args.phase in ("all", "judge"):
        rows = judge_answers(args, rows_path)
        summary = aggregate(rows)
        result = {"model": args.model, "bench_n": len(rows), **summary}
        with open(args.out, "w") as f:
            json.dump({"summary": result, "rows": rows}, f, indent=2)
        print(json.dumps(result, indent=2), flush=True)
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
