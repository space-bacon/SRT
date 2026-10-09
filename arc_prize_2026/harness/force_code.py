#!/usr/bin/env python
"""Budget forcing for code-mode traces: cut the reasoning at a cap, close the think block, and sample several code completions.

Each sample longer than a cap gets --n completions from the truncated reasoning (temperature 0.7). Completions are stored as full
fenced code blocks so exec_code.py can run them. One JSON line per (sample, cap) with the list of completions.
"""
import argparse, asyncio, gzip, json, os, zlib
from openai import AsyncOpenAI
from transformers import AutoTokenizer
import run_llm_arc as R

CLOSE = "\n\nI have run out of thinking time. I will now write my best final code.\n</think>\n\n```python\n"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--caps", default="16384,32768")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ports", default="8001,8002,8003,8004")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--skip-file", default="")
    ap.add_argument("--effort", default="", help="reasoning effort the traces were generated with (xhigh when empty)")
    a = ap.parse_args()
    caps = [int(x) for x in a.caps.split(",")]
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    data = json.load(open(a.challenges))
    rows = [json.loads(l) for l in open(a.run) if '"status": "ok"' in l]
    skip = {tuple(x) for x in json.load(open(a.skip_file))} if a.skip_file else set()
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l)
                if "completions" in r:
                    done.add((r["key"], r["sample"], r["cap"]))
            except ValueError:
                pass
    clients = [AsyncOpenAI(base_url=f"http://127.0.0.1:{p}/v1", api_key="x", timeout=172800, max_retries=0) for p in a.ports.split(",")]
    sem = asyncio.Semaphore(a.concurrency)
    out = open(a.out, "a")
    jobs = []
    for r in rows:
        if (r["key"], r["sample"]) in skip:
            continue
        todo = [c for c in caps if r["completion_tokens"] > c and (r["key"], r["sample"], c) not in done]
        if not todo:
            continue
        tid, ti = r["key"].rsplit("_", 1)
        head = tok.apply_chat_template([{"role": "user", "content": R.build_prompt(data[tid], int(ti), "code")}], tokenize=False, add_generation_prompt=True,
                                       **({"reasoning_effort": a.effort} if a.effort else {}))
        if not head.rstrip().endswith("<think>"):
            head += "<think>\n"
        with gzip.open(os.path.join(a.traces, f"{r['key']}__{r['sample']}.txt.gz"), "rt") as f:
            ids = tok(f.read().strip(), add_special_tokens=False)["input_ids"]
        for c in sorted(todo):
            jobs.append((r["key"], r["sample"], c, head + tok.decode(ids[:c]) + CLOSE))

    async def one(client, key, s, cap, prompt):
        rec = None
        for attempt in range(2):
            try:
                resp = await client.completions.create(model="qwen", prompt=prompt, max_tokens=4000, temperature=0.7, top_p=0.95, n=a.n)
                rec = {"key": key, "sample": s, "cap": cap, "completions": ["```python\n" + ch.text for ch in resp.choices],
                       "completion_tokens": resp.usage.completion_tokens, "prompt_tokens": resp.usage.prompt_tokens}
                break
            except Exception as e:
                rec = {"key": key, "sample": s, "cap": cap, "error": repr(e)[:200], "attempt": attempt}
                await asyncio.sleep(5)
        out.write(json.dumps(rec) + "\n")
        out.flush()

    by_trace = {}
    for key, s, cap, prompt in jobs:
        by_trace.setdefault((key, s), []).append((cap, prompt))

    async def run_trace(ti, key, s, items):
        client = clients[zlib.crc32(f"{key}_{s}".encode()) % len(clients)]
        async with sem:
            for cap, prompt in sorted(items):
                await one(client, key, s, cap, prompt)

    await asyncio.gather(*(run_trace(i, k, s, items) for i, ((k, s), items) in enumerate(by_trace.items())))


if __name__ == "__main__":
    asyncio.run(main())
