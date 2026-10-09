#!/usr/bin/env python
"""A second attempt that costs almost nothing: after a reasoning trace is cut at a cap, ask the model for the test output grid directly from the same truncated reasoning.

The program arm already forces eight programs from the truncated trace. Its second attempt adds 0.4, 0.0, 2.1 and 1.3 points over the first at caps 16K, 32K, 49K and 63K,
so a candidate from a different mechanism could fill that slot. Here every saved trace (g4, g5; code-arm prompt, 64K cap) that is longer than a cap is cut at that cap,
closed with a sentence that asks for the grid, and --n grids are sampled (rows of digits, stop at the first blank line). The program-arm candidates at the same cap are
already on disk, so analyze_direct.py scores 'program top-1 plus the direct answer' against 'program top-2' on the same outputs.

One JSON line per (run, key, cap) is appended to --out as soon as it finishes; reruns skip finished units.
"""
import argparse, asyncio, gzip, json, os, re, time, zlib

from openai import AsyncOpenAI
from transformers import AutoTokenizer

import run_llm_arc as R

CLOSE_D = "\n\nI have run out of thinking time. I will now write my best answer, the output grid for the test input, directly as rows of digits.\n</think>\n\nOUTPUT:\n"


def parse_grid(text):
    rows = []
    for line in (text or "").strip().splitlines():
        line = line.strip()
        if not re.fullmatch(r"[0-9]+", line):
            break
        rows.append([int(c) for c in line])
    if not rows or len({len(r) for r in rows}) != 1 or len(rows) > 30 or len(rows[0]) > 30:
        return None
    return rows


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--runs", default="g5,g4")
    ap.add_argument("--caps", default="32768")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--ports", default="8001,8002,8003,8004")
    ap.add_argument("--per-replica", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
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
            if r.get("kind") == "unit":
                done.add((r["run"], r["key"], r["cap"]))
    out_f = open(a.out, "a")

    def emit(rec):
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()

    units = []
    for run in a.runs.split(","):
        for line in open(os.path.join(a.tables, run, "table.jsonl")):
            row = json.loads(line)
            key = row["key"]
            tid, ti = key.rsplit("_", 1)
            task, ti = data[tid], int(ti)
            for cap in map(int, a.caps.split(",")):
                if row["tokens"] <= cap or (run, key, cap) in done:
                    continue
                with gzip.open(os.path.join(a.tables, run, "traces", f"{key}__0.txt.gz"), "rt") as f:
                    ids = tok(f.read().strip(), add_special_tokens=False)["input_ids"]
                head = tok.apply_chat_template([{"role": "user", "content": R.build_prompt(task, ti, "code")}], tokenize=False, add_generation_prompt=True)
                if not head.rstrip().endswith("<think>"):
                    head += "<think>\n"
                units.append({"run": run, "key": key, "cap": cap, "prompt": head + tok.decode(ids[:cap]).rstrip() + CLOSE_D})
    if a.limit:
        units = units[: a.limit]
    print("units to run:", len(units), flush=True)

    ports = [int(p) for p in a.ports.split(",")]
    clients = [AsyncOpenAI(base_url=f"http://127.0.0.1:{p}/v1", api_key="x", timeout=86400, max_retries=0) for p in ports]
    stats = {"done": 0, "tokens": 0, "err": 0}

    async def one(u, r):
        for attempt in range(3):
            try:
                resp = await clients[r].completions.create(model="qwen", prompt=u["prompt"], max_tokens=1100, temperature=0.7, top_p=0.95, n=a.n, stop=["\n\n"],
                                                           extra_body={"seed": zlib.crc32(f"{u['run']}:{u['key']}:{u['cap']}:direct".encode())})
                grids = [parse_grid(c.text) for c in resp.choices]
                emit({"kind": "unit", "run": u["run"], "key": u["key"], "cap": u["cap"], "grids": grids, "tokens": resp.usage.completion_tokens, "finish": [c.finish_reason for c in resp.choices]})
                stats["done"] += 1
                stats["tokens"] += resp.usage.completion_tokens
                return
            except Exception as e:
                stats["err"] += 1
                emit({"kind": "error", "run": u["run"], "key": u["key"], "cap": u["cap"], "err": repr(e)[:200], "attempt": attempt})
                await asyncio.sleep(10 * (attempt + 1))

    queue = asyncio.Queue()
    for u in units:
        queue.put_nowait(u)

    async def worker(r):
        while True:
            try:
                u = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            await one(u, r)

    async def heartbeat():
        while True:
            await asyncio.sleep(60)
            emit({"kind": "heartbeat", "t": time.time(), **stats, "todo": len(units) - stats["done"]})

    hb = asyncio.create_task(heartbeat())
    emit({"kind": "run_start", "t": time.time(), "units": len(units), "caps": a.caps, "n": a.n})
    await asyncio.gather(*(worker(r) for r in range(len(ports)) for _ in range(a.per_replica)))
    hb.cancel()
    emit({"kind": "run_end", "t": time.time(), **stats})
    print("finished", stats, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
