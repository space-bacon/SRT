#!/usr/bin/env python
"""ARC-AGI-2 solver for a single vLLM server: one reasoning trace per task, budget-forced program finalization, demo-verified voting.

For every task one reasoning trace is sampled from a Qwen3.8 server through the raw completions API (streaming). A trace that ends on its own
yields its own program. A trace that reaches its token cap (or the global deadline) is closed with a fixed sentence and --n-forced programs are sampled
from the truncated reasoning. Every program runs in a sandbox on the demo pairs and on all test inputs. Programs that reproduce every demo output weigh 2,
partial passes weigh 0.25 x fraction; the two grids with the largest weight become attempt_1 and attempt_2.

submission.json is rewritten atomically after every finished task and starts from a fallback (the test input itself), so an interrupted run still scores.
Per-task records go to --log as JSON lines.
"""
import argparse, asyncio, json, os, random, re, subprocess, sys, tempfile, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from openai import AsyncOpenAI
from transformers import AutoTokenizer

HEADER_CODE = (
    "You are participating in a puzzle solving competition. You are an expert programmer and puzzle solver.\n\n"
    "Below is a list of input and output grid pairs that share one transformation rule. Your goal is to find the rule "
    "and write a Python function `transform(grid)` that maps any input grid to its output grid. "
    "`grid` is a list of lists of ints (0-9); return a list of lists of ints. You may import numpy. "
    "The function must reproduce every training output exactly and must generalize to the test input.\n\n"
    "Grids are shown as rows of digits 0-9, one row per line, no separators. Each digit is a color.\n"
    "After your reasoning, give only the complete Python code (imports plus the function) inside one fenced ```python code block, "
    "and write nothing after the code block.\n\n")
CLOSE = "\n\nI have run out of thinking time. I will now write my best final code.\n</think>\n\n```python\n"

EXEC_ONE = r'''
import json, resource, signal, sys
try:
    resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))
except (ValueError, OSError):
    pass
req = json.load(sys.stdin)


class Timeout(Exception):
    pass


def on_alarm(*_):
    raise Timeout()


signal.signal(signal.SIGALRM, on_alarm)


def clean(out):
    import numpy as np
    a = np.asarray(out)
    if a.ndim != 2 or a.size == 0 or a.shape[0] > 30 or a.shape[1] > 30:
        return None
    if not np.issubdtype(a.dtype, np.integer):
        if np.issubdtype(a.dtype, np.floating) and np.all(a == np.round(a)):
            a = a.astype(int)
        else:
            return None
    if a.min() < 0 or a.max() > 9:
        return None
    return a.astype(int).tolist()


def run(f, g, limit=6):
    signal.alarm(limit)
    try:
        return clean(f([list(map(int, r)) for r in g]))
    except BaseException:
        return None
    finally:
        signal.alarm(0)


res = {"n_train": len(req["train"]), "n_pass": 0, "preds": [], "error": None}
try:
    ns = {"__name__": "candidate"}
    signal.alarm(10)
    exec(req["code"], ns)
    signal.alarm(0)
    f = ns["transform"]
except BaseException as e:
    signal.alarm(0)
    res["error"] = "compile: " + repr(e)[:160]
    print(json.dumps(res))
    sys.exit(0)
for ex in req["train"]:
    if run(f, ex["input"]) == ex["output"]:
        res["n_pass"] += 1
res["preds"] = [run(f, g) for g in req["test_inputs"]]
print(json.dumps(res))
'''


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text or "", re.S)
    return blocks[-1] if blocks else None


def grid_text(g):
    return "\n".join("".join(str(c) for c in r) for r in g)


def build_prompt(task):
    parts = [HEADER_CODE, "--Training Examples--\n"]
    for i, ex in enumerate(task["train"]):
        parts.append(f"--Example {i}--\nINPUT:\n{grid_text(ex['input'])}\nOUTPUT:\n{grid_text(ex['output'])}\n")
    parts.append(f"--Test Input--\n{grid_text(task['test'][0]['input'])}\n")
    return "\n".join(parts)


def tkey(g):
    return tuple(map(tuple, g)) if g else None


class Solver:
    def __init__(self, a):
        self.a = a
        self.t0 = a.t_start
        self.deadline = self.t0 + a.total_s
        self.main_deadline = self.deadline - a.reserve_s
        self.tok = AutoTokenizer.from_pretrained(a.model_dir)
        self.client = AsyncOpenAI(base_url=f"http://127.0.0.1:{a.port}/v1", api_key="x", timeout=86400, max_retries=0)
        self.tasks = json.load(open(a.challenges))
        if a.tasks:
            keep = set(json.load(open(a.tasks)))
            self.tasks = {k: v for k, v in self.tasks.items() if k in keep}
        self.sol = json.load(open(a.solutions)) if a.solutions else None
        self.submission = {tid: [{"attempt_1": t["input"], "attempt_2": t["input"]} for t in task["test"]] for tid, task in self.tasks.items()}
        self.pool = ThreadPoolExecutor(a.exec_workers)
        self.log = open(a.log, "a")
        self.stats = defaultdict(float)
        self.sub_lock = asyncio.Lock()

    def now(self):
        return time.time() - self.t0

    def emit(self, rec):
        rec["t"] = round(self.now(), 1)
        self.log.write(json.dumps(rec) + "\n")
        self.log.flush()

    async def write_submission(self):
        async with self.sub_lock:
            tmp = self.a.out + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.submission, f)
            os.replace(tmp, self.a.out)

    def head(self, task):
        h = self.tok.apply_chat_template([{"role": "user", "content": build_prompt(task)}], tokenize=False, add_generation_prompt=True)
        return h if h.rstrip().endswith("<think>") else h + "<think>\n"

    async def trace(self, tid, task):
        """Stream one reasoning trace; returns (text, finish, n_tokens)."""
        a = self.a
        remaining = self.main_deadline - time.time()
        cap = int(min(a.cap, max(a.min_cap, remaining * a.rate)))
        prompt = self.head(task)
        text, finish, ntok = [], None, 0
        stream = await self.client.completions.create(model="qwen", prompt=prompt, max_tokens=cap, temperature=1.0, top_p=0.95, stream=True,
                                                      stream_options={"include_usage": True}, extra_body={"top_k": 20})
        try:
            async for ch in stream:
                if ch.choices:
                    text.append(ch.choices[0].text or "")
                    if ch.choices[0].finish_reason:
                        finish = ch.choices[0].finish_reason
                if getattr(ch, "usage", None):
                    ntok = ch.usage.completion_tokens
                if time.time() > self.main_deadline:
                    finish = "deadline"
                    break
        finally:
            await stream.close()
        return prompt, "".join(text), finish, ntok, cap

    async def forced(self, prompt_head, reasoning):
        a = self.a
        p = prompt_head + reasoning.rstrip() + CLOSE
        for attempt in range(4):
            try:
                r = await self.client.completions.create(model="qwen", prompt=p, max_tokens=a.forced_max_tokens, temperature=0.7, top_p=0.95, n=a.n_forced)
                return ["```python\n" + c.text for c in r.choices], r.usage.completion_tokens
            except Exception as e:
                self.emit({"event": "forced_error", "err": repr(e)[:200], "attempt": attempt})
                await asyncio.sleep(30 * (attempt + 1))
                if time.time() > self.deadline - 300:
                    break
        return [], 0

    def run_program(self, code, task):
        req = {"code": code, "train": task["train"], "test_inputs": [t["input"] for t in task["test"]]}
        try:
            p = subprocess.run([sys.executable, "-c", EXEC_ONE], input=json.dumps(req), capture_output=True, text=True, timeout=90)
            return json.loads(p.stdout.strip().splitlines()[-1])
        except Exception as e:
            return {"n_train": len(task["train"]), "n_pass": 0, "preds": [None] * len(task["test"]), "error": repr(e)[:120]}

    async def execute(self, codes, task):
        loop = asyncio.get_running_loop()
        return await asyncio.gather(*(loop.run_in_executor(self.pool, self.run_program, c, task) for c in codes if c))

    def vote(self, results, task):
        out = []
        for j in range(len(task["test"])):
            w = defaultdict(float)
            for r in results:
                ps = r.get("preds") or []
                p = ps[j] if j < len(ps) else None
                if p is None:
                    continue
                w[tkey(p)] += 2.0 if r["n_pass"] == r["n_train"] else 0.25 * r["n_pass"] / max(r["n_train"], 1)
            ranked = [g for g, _ in sorted(w.items(), key=lambda kv: -kv[1])]
            out.append([[list(r) for r in g] for g in ranked[:2]])
        return out

    async def solve(self, tid, sem_main, sem_forced):
        task = self.tasks[tid]
        t_begin = self.now()
        async with sem_main:
            if time.time() > self.main_deadline - self.a.min_trace_s:
                self.emit({"event": "skipped", "task": tid})
                return
            for attempt in range(4):
                try:
                    prompt, text, finish, ntok, cap = await self.trace(tid, task)
                    break
                except Exception as e:
                    self.emit({"event": "trace_error", "task": tid, "err": repr(e)[:200], "attempt": attempt})
                    await asyncio.sleep(45 * (attempt + 1))
                    if time.time() > self.main_deadline - self.a.min_trace_s:
                        return
            else:
                return
        codes, f_tokens = [], 0
        own = extract_code(text.split("</think>")[-1]) if "</think>" in text else None
        if finish == "stop" and own:
            codes = [own]
        else:
            async with sem_forced:
                completions, f_tokens = await self.forced(prompt, text.split("</think>")[0] if "</think>" in text else text)
            codes = [c for c in (extract_code(x) for x in completions) if c]
        results = await self.execute(codes, task)
        picks = self.vote(results, task)
        for j, ranked in enumerate(picks):
            if ranked:
                att = [ranked[0], ranked[1] if len(ranked) > 1 else ranked[0]]
                self.submission[tid][j] = {"attempt_1": att[0], "attempt_2": att[1]}
        await self.write_submission()
        verified = sum(1 for r in results if r["n_pass"] == r["n_train"] and r.get("preds") and all(p is not None for p in r["preds"]))
        rec = {"event": "done", "task": tid, "finish": finish, "ntok": ntok, "cap": cap, "forced_tokens": f_tokens, "n_programs": len(codes), "n_verified": verified,
               "seconds": round(self.now() - t_begin, 1)}
        if self.sol is not None:
            ok = []
            for j, t in enumerate(task["test"]):
                gold = self.sol[tid][j]
                sub = self.submission[tid][j]
                ok.append(bool(sub["attempt_1"] == gold) or bool(sub["attempt_2"] == gold))
            rec["correct_top2"] = ok
            rec["correct_top1"] = [self.submission[tid][j]["attempt_1"] == self.sol[tid][j] for j in range(len(task["test"]))]
        self.emit(rec)

    async def run(self):
        a = self.a
        await self.write_submission()
        order = list(self.tasks)
        random.Random(a.seed).shuffle(order)
        sem_main = asyncio.Semaphore(a.concurrency)
        sem_forced = asyncio.Semaphore(a.forced_concurrency)
        self.emit({"event": "start", "n_tasks": len(order), "cap": a.cap, "concurrency": a.concurrency, "total_s": a.total_s})
        await asyncio.gather(*(self.solve(t, sem_main, sem_forced) for t in order))
        await self.write_submission()
        self.emit({"event": "end"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--solutions", default="")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", default="submission.json")
    ap.add_argument("--log", default="solver_log.jsonl")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cap", type=int, default=36864)
    ap.add_argument("--min-cap", type=int, default=6144)
    ap.add_argument("--rate", type=float, default=10.0, help="conservative tokens per second per stream used to shrink caps near the deadline")
    ap.add_argument("--n-forced", type=int, default=8)
    ap.add_argument("--forced-max-tokens", type=int, default=4000)
    ap.add_argument("--concurrency", type=int, default=30)
    ap.add_argument("--forced-concurrency", type=int, default=4)
    ap.add_argument("--exec-workers", type=int, default=12)
    ap.add_argument("--t-start", type=float, default=time.time())
    ap.add_argument("--total-s", type=float, default=11.6 * 3600)
    ap.add_argument("--reserve-s", type=float, default=2400)
    ap.add_argument("--min-trace-s", type=float, default=900)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    asyncio.run(Solver(a).run())


if __name__ == "__main__":
    main()
