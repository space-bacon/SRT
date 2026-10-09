#!/usr/bin/env python
"""Budget forcing on saved traces: cut the reasoning at a cap, close the think block and ask for the answer.

For each finished sample longer than a cap, the prompt is chat-template + first <cap> reasoning tokens + a closing line + </think>,
completed greedily on a vLLM replica. Samples at or under the cap keep their original answer. One JSON line per (sample, cap).
"""
import argparse, asyncio, gzip, json, os
from openai import AsyncOpenAI
from transformers import AutoTokenizer
import run_llm_arc as R

CLOSE = "\n\nI have run out of thinking time. I will now give my best final answer.\n</think>\n\n"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--caps", default="8192,16384,24576,32768,49152")
    ap.add_argument("--ports", default="8001,8002,8003,8004")
    ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()
    caps = [int(x) for x in a.caps.split(",")]
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    data = json.load(open(a.challenges))
    rows = [json.loads(l) for l in open(a.run) if '"status": "ok"' in l]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l)
                done.add((r["key"], r["sample"], r["cap"]))
            except ValueError:
                pass
    clients = [AsyncOpenAI(base_url=f"http://127.0.0.1:{p}/v1", api_key="x", timeout=3600, max_retries=0) for p in a.ports.split(",")]
    sem = asyncio.Semaphore(a.concurrency)
    out = open(a.out, "a")
    jobs = []
    for r in rows:
        todo = [c for c in caps if r["completion_tokens"] > c and (r["key"], r["sample"], c) not in done]
        if not todo:
            continue
        tid, ti = r["key"].rsplit("_", 1)
        head = tok.apply_chat_template([{"role": "user", "content": R.build_prompt(data[tid], int(ti))}], tokenize=False, add_generation_prompt=True)
        if not head.rstrip().endswith("<think>"):
            head += "<think>\n"
        with gzip.open(os.path.join(a.traces, f"{r['key']}__{r['sample']}.txt.gz"), "rt") as f:
            ids = tok(f.read().strip(), add_special_tokens=False)["input_ids"]
        for c in sorted(todo):
            jobs.append((r["key"], r["sample"], c, head + tok.decode(ids[:c]) + CLOSE))

    async def one(i, key, s, cap, prompt):
        async with sem:
            try:
                resp = await clients[i % len(clients)].completions.create(model="qwen", prompt=prompt, max_tokens=3000, temperature=0.0)
                rec = {"key": key, "sample": s, "cap": cap, "content": resp.choices[0].text, "completion_tokens": resp.usage.completion_tokens,
                       "prompt_tokens": resp.usage.prompt_tokens, "finish": resp.choices[0].finish_reason}
            except Exception as e:
                rec = {"key": key, "sample": s, "cap": cap, "error": repr(e)[:200]}
            out.write(json.dumps(rec) + "\n")
            out.flush()

    await asyncio.gather(*(one(i, *j) for i, j in enumerate(jobs)))


if __name__ == "__main__":
    asyncio.run(main())
