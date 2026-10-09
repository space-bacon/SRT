#!/usr/bin/env python
"""Forced-step variants on saved traces: does the sampling temperature or the closing sentence change how many cut traces end in a correct program?

Every saved trace (g4, g5; code-arm prompt, xhigh effort) longer than the cap is cut at the cap, closed, and --n programs are sampled. The control (closing sentence A,
temperature 0.7) is already on disk, so only the variants are generated here; analyze_forced_variants.py scores them against it on the same outputs. Because the reasoning is
identical across arms, the paired contrast is far less noisy than a contrast between two full runs.

  T10  closing sentence A, temperature 1.0
  T04  closing sentence A, temperature 0.4
  C2   closing sentence B, temperature 0.7 (asks for a complete program that defines transform and has been checked against the examples in the reasoning)
  C3   closing sentence C, temperature 0.7 (asks for compact code with no comments or docstrings: about 21 percent of the characters of a forced program are comments)

One JSON line per (run, key, variant) is appended to --out as soon as it finishes; reruns skip finished units.
"""
import argparse, asyncio, gzip, json, os, time

from openai import AsyncOpenAI
from transformers import AutoTokenizer

import run_llm_arc as R

CLOSE_A = "\n\nI have run out of thinking time. I will now write my best final code.\n</think>\n\n```python\n"
CLOSE_B = ("\n\nI have run out of thinking time. I will now write the complete final Python program, with imports and a function named transform(grid), "
           "based on the best hypothesis in my reasoning so far, which I have checked against the training examples as far as I could.\n</think>\n\n```python\n")
CLOSE_C = "\n\nI have run out of thinking time. I will now write my best final code. The code must be compact: no comments and no docstrings.\n</think>\n\n```python\n"
VARIANTS = {"T10": (CLOSE_A, 1.0), "T04": (CLOSE_A, 0.4), "C2": (CLOSE_B, 0.7), "C3": (CLOSE_C, 0.7)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--runs", default="g5,g4")
    ap.add_argument("--cap", type=int, default=32768)
    ap.add_argument("--variants", default="T10,T04,C2")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ports", default="8001,8002,8003,8004")
    ap.add_argument("--concurrency", type=int, default=48)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_dir)
    data = json.load(open(a.challenges))
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "completions" in r:
                done.add((r["run"], r["key"], r["variant"]))
    out_f = open(a.out, "a")
    jobs = []
    for run in a.runs.split(","):
        for line in open(os.path.join(a.tables, run, "table.jsonl")):
            row = json.loads(line)
            key = row["key"]
            if row["tokens"] <= a.cap:
                continue
            tid, ti = key.rsplit("_", 1)
            head = tok.apply_chat_template([{"role": "user", "content": R.build_prompt(data[tid], int(ti), "code")}], tokenize=False, add_generation_prompt=True)
            if not head.rstrip().endswith("<think>"):
                head += "<think>\n"
            with gzip.open(os.path.join(a.tables, run, "traces", f"{key}__0.txt.gz"), "rt") as f:
                ids = tok(f.read().strip(), add_special_tokens=False)["input_ids"]
            prefix = head + tok.decode(ids[: a.cap])
            for v in a.variants.split(","):
                if (run, key, v) not in done:
                    jobs.append((run, key, v, prefix + VARIANTS[v][0], VARIANTS[v][1]))
    print("jobs", len(jobs), flush=True)
    clients = [AsyncOpenAI(base_url=f"http://127.0.0.1:{p}/v1", api_key="x", timeout=86400, max_retries=0) for p in a.ports.split(",")]
    load = [0] * len(clients)
    sem = asyncio.Semaphore(a.concurrency)

    async def one(run, key, v, prompt, temp):
        async with sem:
            r = min(range(len(clients)), key=lambda i: load[i])
            load[r] += 1
            try:
                for attempt in range(2):
                    try:
                        resp = await clients[r].completions.create(model="qwen", prompt=prompt, max_tokens=4000, temperature=temp, top_p=0.95, n=a.n)
                        rec = {"run": run, "key": key, "variant": v, "completions": ["```python\n" + c.text for c in resp.choices], "completion_tokens": resp.usage.completion_tokens}
                        break
                    except Exception as e:
                        rec = {"run": run, "key": key, "variant": v, "error": repr(e)[:200]}
                        await asyncio.sleep(5)
            finally:
                load[r] -= 1
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()

    t0 = time.time()
    await asyncio.gather(*(one(*j) for j in jobs))
    print("finished in", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
