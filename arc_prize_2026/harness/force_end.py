#!/usr/bin/env python
"""Budget forcing at the end of traces that stopped on their own without an answer.

With reasoning effort medium, about half of the traces that finish with finish_reason 'stop' have an empty answer: the text ends in the middle of a sentence or a grid.
A deployed solver closes such a trace where it stopped and samples programs from it, exactly as for a trace cut at a cap. This script does that for every trace of a
run with finish 'stop' and an empty answer, with the same closing sentence and sampling as force_code.py, and records the trace length as `cap` so that
per_task_eval.py can use the result for every cap at or above that length.
"""
import argparse, asyncio, gzip, json, os
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
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ports", default="8001,8002,8003,8004")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--effort", default="")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    data = json.load(open(a.challenges))
    rows = [json.loads(l) for l in open(a.run) if '"status": "ok"' in l]
    rows = [r for r in rows if r["finish"] == "stop" and not (r.get("content") or "").strip()]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l)
                if "completions" in r:
                    done.add((r["key"], r["sample"]))
            except ValueError:
                pass
    clients = [AsyncOpenAI(base_url=f"http://127.0.0.1:{p}/v1", api_key="x", timeout=172800, max_retries=0) for p in a.ports.split(",")]
    sem = asyncio.Semaphore(a.concurrency)
    out = open(a.out, "a")
    jobs = []
    for r in rows:
        if (r["key"], r["sample"]) in done:
            continue
        tid, ti = r["key"].rsplit("_", 1)
        head = tok.apply_chat_template([{"role": "user", "content": R.build_prompt(data[tid], int(ti), "code")}], tokenize=False, add_generation_prompt=True,
                                       **({"reasoning_effort": a.effort} if a.effort else {}))
        if not head.rstrip().endswith("<think>"):
            head += "<think>\n"
        with gzip.open(os.path.join(a.traces, f"{r['key']}__{r['sample']}.txt.gz"), "rt") as f:
            text = f.read().strip()
        jobs.append((r["key"], r["sample"], r["completion_tokens"], head + text + CLOSE))
    print("jobs", len(jobs), flush=True)
    counter = {"i": 0}

    async def one(key, s, cap, prompt):
        async with sem:
            client = clients[counter["i"] % len(clients)]
            counter["i"] += 1
            rec = None
            for attempt in range(2):
                try:
                    resp = await client.completions.create(model="qwen", prompt=prompt, max_tokens=4000, temperature=0.7, top_p=0.95, n=a.n)
                    rec = {"key": key, "sample": s, "cap": cap, "completions": ["```python\n" + ch.text for ch in resp.choices], "completion_tokens": resp.usage.completion_tokens,
                           "prompt_tokens": resp.usage.prompt_tokens}
                    break
                except Exception as e:
                    rec = {"key": key, "sample": s, "cap": cap, "error": repr(e)[:200], "attempt": attempt}
                    await asyncio.sleep(5)
            out.write(json.dumps(rec) + "\n")
            out.flush()

    await asyncio.gather(*(one(*j) for j in jobs))


if __name__ == "__main__":
    asyncio.run(main())
