#!/usr/bin/env python
"""Execution feedback injected into a truncated reasoning trace, evaluated against plain continuation (which already exists in the saved runs).

Population: every saved trace (g4, g5; code-arm, 64K cap) is cut at the 16K checkpoint. The eight programs sampled from that prefix (saved in table.jsonl)
are executed on the demo pairs. A trace with a program that passes every demo is finished (nothing to test). For the others the best program (most demos passed,
then fewest wrong cells) is run again to get its outputs, and the first failing demo is turned into feedback text. Two ways to use that feedback:

  A  the program, its result and the diff are appended inside the thinking block after the 16K prefix and the model keeps reasoning for --cont more tokens;
  B  a fresh chat turn: the original prompt, the program as the assistant answer, the feedback as the next user message, then --cont tokens of new reasoning;
  S  as A, but the closing think tag is forbidden (logit bias) for the whole continuation, because in A the model closes its thinking within about a thousand tokens.

After the continuation the trace is closed and --n programs are sampled (or its own program is used if it ended). Programs run on all test inputs and are voted
(demo-passing weigh 2, partial passes 0.25 x fraction). One JSON line per (run, key, variant) is appended to --out as soon as it finishes; reruns skip finished units.
"""
import argparse, asyncio, concurrent.futures as cf, gzip, hashlib, json, os, random, re, subprocess, sys, time, zlib
from collections import defaultdict

from openai import AsyncOpenAI
from transformers import AutoTokenizer

import run_llm_arc as R

HERE = os.path.dirname(os.path.abspath(__file__))
CLOSE = "\n\nI have run out of thinking time. I will now write my best final code.\n</think>\n\n```python\n"


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text or "", re.S)
    return blocks[-1] if blocks else None


def tkey(g):
    return tuple(map(tuple, g)) if g else None


def ghash(g):
    return hashlib.md5(json.dumps(g).encode()).hexdigest()[:10] if g is not None else None


def run_prog(code, task):
    req = {"code": code, "train": task["train"], "test_inputs": [t["input"] for t in task["test"]]}
    try:
        p = subprocess.run([sys.executable, os.path.join(HERE, "exec_demo.py")], input=json.dumps(req), capture_output=True, text=True, timeout=90)
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"n_train": len(task["train"]), "n_pass": 0, "demo": [], "preds": [None] * len(task["test"]), "error": "exec: " + repr(e)[:120]}


def vote(results, j):
    w = defaultdict(float)
    for r in results:
        ps = r.get("preds") or []
        p = ps[j] if j < len(ps) else None
        if p is None:
            continue
        w[tkey(p)] += 2.0 if r["n_pass"] == r["n_train"] else 0.25 * r["n_pass"] / max(r["n_train"], 1)
    return [g for g, _ in sorted(w.items(), key=lambda kv: -kv[1])]


def wrong_cells(out, exp):
    if out is None:
        return 10 ** 4
    if (len(out), len(out[0])) != (len(exp), len(exp[0])):
        return 10 ** 3
    return sum(1 for r in range(len(exp)) for c in range(len(exp[0])) if out[r][c] != exp[r][c])


def closeness(task, res):
    if res.get("error"):
        return 10 ** 6
    return sum(wrong_cells(d["out"], task["train"][i]["output"]) for i, d in enumerate(res["demo"]) if d["out"] is None or d["out"] != task["train"][i]["output"])


def detail_text(task, res):
    if res.get("error"):
        return f"The program does not run at all: {res['error']}."
    train = task["train"]
    fails = [i for i, d in enumerate(res["demo"]) if d["out"] is None or d["out"] != train[i]["output"]]
    i = min(fails, key=lambda i: len(train[i]["output"]) * len(train[i]["output"][0]))
    d, exp = res["demo"][i], train[i]["output"]
    if d["out"] is None:
        return f"On training example {i} the function failed: {d['err']}."
    out = d["out"]
    if (len(out), len(out[0])) != (len(exp), len(exp[0])):
        return (f"On training example {i} the output has shape {len(out)}x{len(out[0])} but the expected shape is {len(exp)}x{len(exp[0])}.\n"
                f"Program output:\n{R.grid_text(out)}\nExpected output:\n{R.grid_text(exp)}")
    cells = [(r, c, out[r][c], exp[r][c]) for r in range(len(exp)) for c in range(len(exp[0])) if out[r][c] != exp[r][c]]
    shown = ", ".join(f"({r},{c}) {o}->{e}" for r, c, o, e in cells[:12]) + (" ..." if len(cells) > 12 else "")
    return (f"On training example {i} the output has the right shape ({len(out)}x{len(out[0])}) but {len(cells)} cells are wrong.\n"
            f"Program output:\n{R.grid_text(out)}\nExpected output:\n{R.grid_text(exp)}\nWrong cells (row, col) program->expected: {shown}")


def inject_a(code, k, n, detail):
    return ("\n\nLet me stop here and test my current best program on the training examples.\n\n```python\n" + code.strip() + "\n```\n\n"
            f"I ran it. It reproduces {k} of {n} training examples.\n{detail}\n\n"
            "So this program is not right yet. Let me work out exactly why it fails and what the real rule is, and then fix it.\n")


def messages_b(task, ti, code, k, n, detail):
    user2 = (f"I ran your program on the training examples. It reproduces {k} of {n} of them.\n{detail}\n\n"
             "The program is not correct yet. Think carefully about what the rule really is, using this feedback, then give a corrected complete Python program "
             "(imports plus the function `transform(grid)`) inside one fenced ```python code block, and write nothing after the code block.")
    return [{"role": "user", "content": R.build_prompt(task, ti, "code")}, {"role": "assistant", "content": "```python\n" + code.strip() + "\n```"}, {"role": "user", "content": user2}]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True, help="directory holding g4/ and g5/ with table.jsonl and traces/")
    ap.add_argument("--runs", default="g5,g4")
    ap.add_argument("--variants", default="A,B")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--solutions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", type=int, default=16384)
    ap.add_argument("--cont", type=int, default=16384)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ports", default="8001,8002,8003,8004")
    ap.add_argument("--per-replica", type=int, default=16)
    ap.add_argument("--forced-per-replica", type=int, default=6)
    ap.add_argument("--exec-workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="only the first N units (testing)")
    ap.add_argument("--dry", action="store_true", help="prepare and print sample prompts, no generation")
    ap.add_argument("--samples-dir", default="")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_dir)
    THINK_END = tok.convert_tokens_to_ids("</think>")
    print("think-end token id", THINK_END, flush=True)
    data = json.load(open(a.challenges))
    sol = json.load(open(a.solutions))
    gold = lambda k: tkey(sol[k.rsplit("_", 1)[0]][int(k.rsplit("_", 1)[1])])
    pool = cf.ThreadPoolExecutor(a.exec_workers)
    loop = asyncio.get_running_loop()
    arun = lambda code, task: loop.run_in_executor(pool, run_prog, code, task)

    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if r.get("kind") == "unit":
                done.add((r["run"], r["key"], r["variant"]))
    out_f = open(a.out, "a")

    def emit(rec):
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()

    # phase 0: run the 16K programs, find verified outputs, build the feedback
    rows = []
    for run in a.runs.split(","):
        for l in open(os.path.join(a.tables, run, "table.jsonl")):
            r = json.loads(l)
            r["dir"] = os.path.join(a.tables, run)
            rows.append(r)
    random.Random(0).shuffle(rows)
    if a.limit:
        rows = rows[: a.limit]
    t0 = time.time()

    async def prep(row):
        key = row["key"]
        tid, ti = key.rsplit("_", 1)
        task = data[tid]
        codes = [c for c in (extract_code(x) for x in row["progs16"]) if c]
        res = await asyncio.gather(*(arun(c, task) for c in codes))
        j = int(ti)
        verified = [i for i, r in enumerate(res) if r["n_pass"] == r["n_train"] and r.get("preds") and j < len(r["preds"]) and r["preds"][j] is not None]
        rank = vote(res, j)
        g = gold(key)
        rec = {"kind": "prep", "run": row["run"], "key": key, "n_programs16": len(codes), "n_verified16": len(verified), "top1_16": bool(rank) and rank[0] == g, "top2_16": g in rank[:2]}
        best = None
        if codes and not verified:
            best = min(range(len(codes)), key=lambda i: (-res[i]["n_pass"], closeness(task, res[i]), i))
            rec["p_star_n_pass"] = res[best]["n_pass"]
            rec["n_train"] = res[best]["n_train"]
        emit(rec)
        return row, task, codes, res, best, verified

    preps = await asyncio.gather(*(prep(r) for r in rows))
    print(f"prep done in {time.time() - t0:.0f}s: {len(preps)} traces, verified at checkpoint {sum(1 for p in preps if p[5])}, without any program {sum(1 for p in preps if not p[2])}", flush=True)

    units = []
    for row, task, codes, res, best, verified in preps:
        if verified or best is None:
            continue
        key = row["key"]
        ti = int(key.rsplit("_", 1)[1])
        code = codes[best]
        k, n = res[best]["n_pass"], res[best]["n_train"]
        det = detail_text(task, res[best])
        with gzip.open(os.path.join(row["dir"], "traces", f"{key}__0.txt.gz"), "rt") as f:
            ids = tok(f.read().strip(), add_special_tokens=False)["input_ids"]
        head = tok.apply_chat_template([{"role": "user", "content": R.build_prompt(task, ti, "code")}], tokenize=False, add_generation_prompt=True)
        if not head.rstrip().endswith("<think>"):
            head += "<think>\n"
        prefix = tok.decode(ids[: a.ckpt])
        headb = tok.apply_chat_template(messages_b(task, ti, code, k, n, det), tokenize=False, add_generation_prompt=True)
        if not headb.rstrip().endswith("<think>"):
            headb += "<think>\n"
        prompts = {"A": head + prefix + inject_a(code, k, n, det), "B": headb}
        prompts["S"] = prompts["A"]
        for v in a.variants.split(","):
            if (row["run"], key, v) not in done:
                units.append({"run": row["run"], "key": key, "variant": v, "task": task, "ti": ti, "prompt": prompts[v], "k": k, "n": n})
    # keep both variants of an output next to each other
    by_out = defaultdict(list)
    for u in units:
        by_out[(u["run"], u["key"])].append(u)
    units = [u for grp in by_out.values() for u in grp]
    print("units to run:", len(units), flush=True)
    if a.samples_dir:
        os.makedirs(a.samples_dir, exist_ok=True)
        for u in units[:4]:
            open(os.path.join(a.samples_dir, f"{u['run']}_{u['key']}_{u['variant']}.txt"), "w").write(u["prompt"][-6000:] if u["variant"] == "A" else u["prompt"])
    if a.dry:
        return

    ports = [int(p) for p in a.ports.split(",")]
    clients = [AsyncOpenAI(base_url=f"http://127.0.0.1:{p}/v1", api_key="x", timeout=172800, max_retries=0) for p in ports]
    fsems = [asyncio.Semaphore(a.forced_per_replica) for _ in ports]
    stats = {"done": 0, "cont_tokens": 0, "forced_tokens": 0, "err": 0}

    async def one(u, r):
        key, task, ti = u["key"], u["task"], u["ti"]
        client = clients[r]
        seed = zlib.crc32(f"{u['run']}:{key}:{u['variant']}".encode())
        t_begin = time.time()
        text, finish, ntok = "", None, 0
        for attempt in range(3):
            try:
                kw = {"logit_bias": {str(THINK_END): -100}} if u["variant"] == "S" else {}
                resp = await client.completions.create(model="qwen", prompt=u["prompt"], max_tokens=a.cont, temperature=1.0, top_p=0.95, extra_body={"top_k": 20, "seed": seed}, **kw)
                text, finish, ntok = resp.choices[0].text, resp.choices[0].finish_reason, resp.usage.completion_tokens
                break
            except Exception as e:
                stats["err"] += 1
                emit({"kind": "error", "run": u["run"], "key": key, "variant": u["variant"], "err": repr(e)[:200], "attempt": attempt})
                await asyncio.sleep(10 * (attempt + 1))
        else:
            return
        stats["cont_tokens"] += ntok
        completions, f_tokens, own = [], 0, None
        if finish == "stop" and "</think>" in text:
            own = extract_code(text.split("</think>")[-1])
        if own:
            completions = ["```python\n" + own + "```"]
        else:
            reasoning = text.split("</think>")[0] if "</think>" in text else text
            async with fsems[r]:
                for attempt in range(3):
                    try:
                        resp = await client.completions.create(model="qwen", prompt=u["prompt"] + reasoning.rstrip() + CLOSE, max_tokens=4000, temperature=0.7, top_p=0.95, n=a.n)
                        completions = ["```python\n" + c.text for c in resp.choices]
                        f_tokens = resp.usage.completion_tokens
                        break
                    except Exception as e:
                        stats["err"] += 1
                        emit({"kind": "error", "run": u["run"], "key": key, "variant": u["variant"], "stage": "forced", "err": repr(e)[:200], "attempt": attempt})
                        await asyncio.sleep(10 * (attempt + 1))
        stats["forced_tokens"] += f_tokens
        codes = [c for c in (extract_code(x) for x in completions) if c]
        res = await asyncio.gather(*(arun(c, task) for c in codes))
        g = gold(key)
        rank = vote(res, ti)
        progs = []
        for rr in res:
            p = rr["preds"][ti] if rr.get("preds") and ti < len(rr["preds"]) else None
            progs.append({"n_pass": rr["n_pass"], "n_train": rr["n_train"], "h": ghash(p), "ok": p is not None and tkey(p) == g})
        emit({"kind": "unit", "run": u["run"], "key": key, "variant": u["variant"], "cont_tokens": ntok, "cont_finish": finish, "own": bool(own), "forced_tokens": f_tokens,
              "n_programs": len(codes), "n_verified": sum(1 for p in progs if p["n_pass"] == p["n_train"] and p["h"]), "top1": bool(rank) and rank[0] == g, "top2": g in rank[:2],
              "p_star_n_pass": u["k"], "n_train": u["n"], "progs": progs, "seconds": round(time.time() - t_begin, 1)})
        stats["done"] += 1

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
    emit({"kind": "run_start", "t": time.time(), "units": len(units), "ckpt": a.ckpt, "cont": a.cont, "variants": a.variants})
    await asyncio.gather(*(worker(r) for r in range(len(ports)) for _ in range(a.per_replica)))
    hb.cancel()
    emit({"kind": "run_end", "t": time.time(), **stats})


if __name__ == "__main__":
    asyncio.run(main())
