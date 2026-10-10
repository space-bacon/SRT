#!/usr/bin/env python
"""ARC-AGI-2 solver v2 for one or more vLLM replicas: one reasoning trace per task, budget-forced program finalization, demo-verified voting.

Per task: stream one reasoning trace from a Qwen3.8 server through the raw completions API. A trace that ends on its own yields its own program; a trace that
reaches its token cap is closed with a fixed sentence and --n-forced programs are sampled from the truncated reasoning. Programs run in a sandbox on the demo pairs and on
every test input; programs that reproduce all demos weigh 2, partial passes 0.25 x fraction, and the two heaviest distinct grids become attempt_1 and attempt_2.

Differences from v1: several replicas (each task stays on one replica so a prefix cache can serve its forced step), a cap controller that picks the token cap of each new
trace from the measured throughput and the work left (so a slow or fast machine still finishes inside the budget), resume from the log, and an optional execution-feedback
stage (--fb A|B): at a checkpoint the best program is run, and if no program passes every demo the first failing demo is described to the model (A: inside the thinking
block, B: as a fresh chat turn) before the trace continues. submission.json is rewritten atomically after every task and starts from a fallback (the test input).
"""
import argparse, asyncio, json, os, random, re, subprocess, sys, time, urllib.request
from collections import defaultdict, deque
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
FENCE = "`" * 3

EXEC_DEMO = r'''
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
    if a.dtype == object:
        return None, "ragged or non-numeric nested list"
    if a.ndim != 2:
        return None, f"expected a 2D grid but got {a.ndim} dimensions"
    if a.size == 0:
        return None, "empty grid"
    if a.shape[0] > 30 or a.shape[1] > 30:
        return None, f"grid of shape {a.shape[0]}x{a.shape[1]} is larger than 30x30"
    if not np.issubdtype(a.dtype, np.integer):
        if np.issubdtype(a.dtype, np.floating) and np.all(a == np.round(a)):
            a = a.astype(int)
        else:
            return None, "non-integer values"
    if a.min() < 0 or a.max() > 9:
        return None, "values outside 0-9"
    return a.astype(int).tolist(), None


def run(f, g, limit=6):
    signal.alarm(limit)
    try:
        return clean(f([list(map(int, r)) for r in g]))
    except Timeout:
        return None, f"timed out after {limit} s"
    except BaseException as e:
        return None, (type(e).__name__ + ": " + str(e))[:160]
    finally:
        signal.alarm(0)


res = {"n_train": len(req["train"]), "n_pass": 0, "demo": [], "preds": [], "error": None}
try:
    ns = {"__name__": "candidate"}
    signal.alarm(10)
    exec(req["code"], ns)
    signal.alarm(0)
    f = ns["transform"]
except BaseException as e:
    signal.alarm(0)
    res["error"] = "compile: " + (type(e).__name__ + ": " + str(e))[:160]
    print(json.dumps(res))
    sys.exit(0)
for ex in req["train"]:
    out, err = run(f, ex["input"])
    res["demo"].append({"out": out, "err": err})
    if out is not None and out == ex["output"]:
        res["n_pass"] += 1
res["preds"] = [run(f, g)[0] for g in req["test_inputs"]]
print(json.dumps(res))
'''

# mean decoded tokens per task (reasoning plus forced programs) at each cap, measured on 120 public evaluation tasks (two runs agree within 0.2K)
COST_POINTS = [(8192, 17.0e3), (16384, 24.7e3), (32768, 42.0e3), (49152, 58.0e3), (63000, 70.0e3)]


def cost_of_cap(c):
    pts = COST_POINTS
    if c <= pts[0][0]:
        return pts[0][1] * c / pts[0][0] if c > 0 else 0.0
    for (c0, v0), (c1, v1) in zip(pts, pts[1:]):
        if c <= c1:
            return v0 + (v1 - v0) * (c - c0) / (c1 - c0)
    return pts[-1][1] + (c - pts[-1][0]) * 0.9


def extract_code(text):
    blocks = re.findall(FENCE + r"(?:python|py)?\n(.*?)" + FENCE, text or "", re.S)
    return blocks[-1] if blocks else None


def grid_text(g):
    return "\n".join("".join(str(c) for c in r) for r in g)


PROMPT_EXTRA = ""
PROMPT_SUMMARY = False


def summarize_text(train, test_inputs):
    out = []
    def _print(*a):
        out.append(" ".join(str(x) for x in a))

    import numpy as np
    from collections import Counter

    def comps(a, bg, diag=True):
        h, w = a.shape
        seen = np.zeros((h, w), bool)
        out = []
        for i in range(h):
            for j in range(w):
                if a[i, j] != bg and not seen[i, j]:
                    c = a[i, j]
                    st = [(i, j)]
                    seen[i, j] = True
                    cells = []
                    while st:
                        y, x = st.pop()
                        cells.append((y, x))
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                if (dy or dx) and (diag or not (dy and dx)):
                                    yy, xx = y + dy, x + dx
                                    if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and a[yy, xx] == c:
                                        seen[yy, xx] = True
                                        st.append((yy, xx))
                    ys = [c_[0] for c_ in cells]
                    xs = [c_[1] for c_ in cells]
                    out.append((int(c), len(cells), (min(ys), min(xs), max(ys), max(xs))))
        return out

    def desc(a, name):
        bg = Counter(a.ravel().tolist()).most_common(1)[0][0]
        cc = Counter(a.ravel().tolist())
        cs = comps(a, bg)
        t = name + ": " + str(a.shape[0]) + "x" + str(a.shape[1]) + ", colors " + ", ".join(str(k) + ":" + str(v) for k, v in sorted(cc.items())) + ", background guess " + str(bg)
        t += "; " + str(len(cs)) + " same-color objects (8-connected)"
        if cs:
            big = sorted(cs, key=lambda o: -o[1])[:6]
            t += ", largest: " + "; ".join("color " + str(c) + " size " + str(n) + " rows " + str(b[0]) + "-" + str(b[2]) + " cols " + str(b[1]) + "-" + str(b[3]) for c, n, b in big)
        sym = []
        if (a == a[::-1]).all(): sym.append("up-down symmetric")
        if (a == a[:, ::-1]).all(): sym.append("left-right symmetric")
        if a.shape[0] == a.shape[1] and (a == a.T).all(): sym.append("transpose symmetric")
        if sym: t += "; " + ", ".join(sym)
        return t

    pairs = train
    for i, ex in enumerate(pairs):
        a, b = np.array(ex["input"]), np.array(ex["output"])
        _print("Example", i)
        _print("  " + desc(a, "input"))
        _print("  " + desc(b, "output"))
        if a.shape == b.shape:
            d = a != b
            n = int(d.sum())
            tr = Counter((int(x), int(y)) for x, y in zip(a[d], b[d]))
            if n:
                ys, xs = np.where(d)
                _print("  same shape; " + str(n) + " cells change (rows " + str(ys.min()) + "-" + str(ys.max()) + ", cols " + str(xs.min()) + "-" + str(xs.max()) + "); color changes (from, to): " + ", ".join(str(k) + " x" + str(v) for k, v in tr.most_common(8)))
            else:
                _print("  same shape; no cell changes")
        else:
            note = []
            if b.shape[0] % a.shape[0] == 0 and b.shape[1] % a.shape[1] == 0:
                fy, fx = b.shape[0] // a.shape[0], b.shape[1] // a.shape[1]
                note.append("output is " + str(fy) + "x" + str(fx) + " times the input size" + (" and equals the input tiled" if (np.tile(a, (fy, fx)) == b).all() else "") + (" and equals the input scaled up" if (np.kron(a, np.ones((fy, fx), int)) == b).all() else ""))
            if b.shape[0] <= a.shape[0] and b.shape[1] <= a.shape[1]:
                hits = [(y, x) for y in range(a.shape[0] - b.shape[0] + 1) for x in range(a.shape[1] - b.shape[1] + 1) if (a[y:y + b.shape[0], x:x + b.shape[1]] == b).all()]
                if hits: note.append("output is a sub-grid of the input at row, col " + str(hits[0]))
            _print("  different shape" + ("; " + "; ".join(note) if note else ""))
    tests = test_inputs
    for k, t_ in enumerate(tests):
        _print("Test input" + (" %d" % k if len(tests) > 1 else ""))
        _print("  " + desc(np.array(t_), "input"))
    return "\n".join(out)


def build_prompt(task, ti=0):
    parts = [HEADER_CODE + PROMPT_EXTRA, "--Training Examples--\n"]
    for i, ex in enumerate(task["train"]):
        parts.append(f"--Example {i}--\nINPUT:\n{grid_text(ex['input'])}\nOUTPUT:\n{grid_text(ex['output'])}\n")
    parts.append(f"--Test Input--\n{grid_text(task['test'][ti]['input'])}\n")
    if PROMPT_SUMMARY:
        parts.append("--Facts about the grids, computed by code (colors with their cell counts; objects are connected groups of cells of one color, largest first, with their row and column ranges)--\n"
                     + summarize_text(task["train"], [task["test"][ti]["input"]]) + "\n")
    return "\n".join(parts)


def tkey(g):
    return tuple(map(tuple, g)) if g else None


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
                f"Program output:\n{grid_text(out)}\nExpected output:\n{grid_text(exp)}")
    cells = [(r, c, out[r][c], exp[r][c]) for r in range(len(exp)) for c in range(len(exp[0])) if out[r][c] != exp[r][c]]
    shown = ", ".join(f"({r},{c}) {o}->{e}" for r, c, o, e in cells[:12]) + (" ..." if len(cells) > 12 else "")
    return (f"On training example {i} the output has the right shape ({len(out)}x{len(out[0])}) but {len(cells)} cells are wrong.\n"
            f"Program output:\n{grid_text(out)}\nExpected output:\n{grid_text(exp)}\nWrong cells (row, col) program->expected: {shown}")


def inject_a(code, k, n, detail):
    return (f"\n\nLet me stop here and test my current best program on the training examples.\n\n{FENCE}python\n" + code.strip() + f"\n{FENCE}\n\n"
            f"I ran it. It reproduces {k} of {n} training examples.\n{detail}\n\n"
            "So this program is not right yet. Let me work out exactly why it fails and what the real rule is, and then fix it.\n")


def messages_b(task, code, k, n, detail):
    user2 = (f"I ran your program on the training examples. It reproduces {k} of {n} of them.\n{detail}\n\n"
             "The program is not correct yet. Think carefully about what the rule really is, using this feedback, then give a corrected complete Python program "
             f"(imports plus the function `transform(grid)`) inside one fenced {FENCE}python code block, and write nothing after the code block.")
    return [{"role": "user", "content": build_prompt(task)}, {"role": "assistant", "content": f"{FENCE}python\n" + code.strip() + f"\n{FENCE}"}, {"role": "user", "content": user2}]


class Replica:
    def __init__(self, port):
        self.port = port
        self.client = AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="x", timeout=86400, max_retries=0)
        self.active = 0


class Solver:
    def __init__(self, a):
        self.a = a
        self.t0 = a.t_start
        self.deadline = self.t0 + a.total_s
        self.main_deadline = self.deadline - a.reserve_s
        self.tok = AutoTokenizer.from_pretrained(a.model_dir)
        self.replicas = [Replica(int(p)) for p in a.ports.split(",")]
        self.tasks = json.load(open(a.challenges))
        if a.tasks:
            keep = set(json.load(open(a.tasks)))
            self.tasks = {k: v for k, v in self.tasks.items() if k in keep}
        self.sol = json.load(open(a.solutions)) if a.solutions else None
        self.submission = {tid: [{"attempt_1": t["input"], "attempt_2": t["input"]} for t in task["test"]] for tid, task in self.tasks.items()}
        self.done_tasks = set()
        if os.path.exists(a.log):
            for line in open(a.log):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("event") == "done":
                    self.done_tasks.add(r["task"])
            if self.done_tasks and os.path.exists(a.out):
                try:
                    prev = json.load(open(a.out))
                    for tid in self.done_tasks:
                        if tid in prev and tid in self.submission:
                            self.submission[tid] = prev[tid]
                except Exception:
                    pass
        self.pool = ThreadPoolExecutor(a.exec_workers)
        self.log = open(a.log, "a")
        self.sub_lock = asyncio.Lock()
        self.inflight_caps = {}
        self.inflight_start = {}
        self.stat_tokens, self.stat_pred, self.stat_secs, self.stat_n = 0.0, 0.0, 0.0, 0
        self.last_cap, self.n_started = None, 0
        self.not_started = 0
        self.gen_hist = deque(maxlen=400)
        self.thr = a.thr_prior
        self.cap_sum, self.cap_n = 0.0, 0

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

    def head(self, task, ti=0):
        return self.head_from_messages([{"role": "user", "content": build_prompt(task, ti)}])

    def head_from_messages(self, msgs):
        kw = {"reasoning_effort": self.a.effort} if self.a.effort else {}
        h = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        return h if h.rstrip().endswith("<think>") else h + "<think>\n"

    def gen_tokens_total(self):
        tot = 0.0
        for r in self.replicas:
            try:
                txt = urllib.request.urlopen(f"http://127.0.0.1:{r.port}/metrics", timeout=5).read().decode()
            except Exception:
                return None
            for line in txt.splitlines():
                if line.startswith("vllm:generation_tokens_total"):
                    tot += float(line.rpartition(" ")[2])
        return tot

    async def monitor(self):
        """Throughput of the whole server group over the last 20 minutes (tokens decoded per wall second, prefill stalls included)."""
        while True:
            await asyncio.sleep(30)
            tot = await asyncio.get_running_loop().run_in_executor(None, self.gen_tokens_total)
            if tot is None:
                continue
            now = time.time()
            self.gen_hist.append((now, tot, sum(r.active for r in self.replicas)))
            while self.gen_hist and now - self.gen_hist[0][0] > 1200:
                self.gen_hist.popleft()
            full = [h for h in self.gen_hist if h[2] >= 0.8 * self.target_streams]
            if len(self.gen_hist) >= 20 and len(full) >= 0.7 * len(self.gen_hist):
                dt = self.gen_hist[-1][0] - self.gen_hist[0][0]
                self.thr = max(50.0, (self.gen_hist[-1][1] - self.gen_hist[0][1]) / dt)

    def sec_per_cost(self):
        """Stream-seconds per predicted decoded token (the cost table at the cap the task was given), from the tasks finished so far: decode, finalization and every wait are inside the
        seconds, and a table that is too high or too low is absorbed. The prior counts as `prior-tasks` tasks of 40K tokens."""
        base = 40e3 * self.a.prior_tasks
        prior_r = max(self.target_streams, 1) / self.a.thr_prior
        return (self.stat_secs + base * prior_r) / (self.stat_pred + base)

    def thr_est(self):
        """Decoded tokens per second of the whole group, from the tasks finished so far (for the log)."""
        base = 40e3 * self.a.prior_tasks
        n = max(self.target_streams, 1)
        return (self.stat_tokens + base) / (self.stat_secs / n + base / self.a.thr_prior)

    def note_done(self, rec):
        toks = rec.get("ntok", 0) + rec.get("forced_tokens", 0)
        secs = rec.get("seconds", 0.0)
        if toks > 0 and secs > 0 and rec.get("cap"):
            self.stat_tokens += toks
            self.stat_pred += cost_of_cap(rec["cap"])
            self.stat_secs += secs
            self.stat_n += 1

    def choose_cap(self):
        """Largest cap whose expected stream-seconds, added to what the tasks in flight still need, fit in the time left. The seconds per token come from finished tasks, so the waves of
        decoding and finalization that come from tasks starting together average out; the cap moves by at most --cap-step between tasks and the first wave is spread."""
        a = self.a
        now = time.time()
        left = self.main_deadline - now
        r = self.sec_per_cost()
        fb_extra = a.fb_n1 * 1.2e3 if a.fb != "off" else 0.0
        avail = max(self.target_streams, 1) * left * a.safety
        inflight = 0.0
        for tid, cap_i in self.inflight_caps.items():
            d_i = r * cost_of_cap(cap_i)
            inflight += max(0.15 * d_i, d_i - (now - self.inflight_start.get(tid, now)))
        c = a.max_cap
        while c > a.min_cap:
            if inflight + (self.not_started + 1) * r * (cost_of_cap(c) + fb_extra) <= avail:
                break
            c -= 512
        if self.last_cap is not None:
            c = max(self.last_cap - a.cap_step_down, min(self.last_cap + a.cap_step, c))
        self.last_cap = c
        if self.n_started < self.target_streams and a.first_wave_lo < 1.0:
            c = int(c * (a.first_wave_lo + (1.0 - a.first_wave_lo) * self.n_started / max(self.target_streams - 1, 1)))
        self.n_started += 1
        # a trace has to fit in the time that is left at a pessimistic per-stream speed
        return int(max(a.min_cap, min(c, left * a.stream_rate)))

    async def stream_text(self, rep, prompt, max_tokens, seed=None):
        extra = {"top_k": 20}
        if seed is not None:
            extra["seed"] = seed
        text, finish, ntok = [], None, 0
        stream = await rep.client.completions.create(model="qwen", prompt=prompt, max_tokens=max_tokens, temperature=1.0, top_p=0.95, stream=True,
                                                     stream_options={"include_usage": True}, extra_body=extra)
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
        return "".join(text), finish, ntok

    async def forced(self, rep, prompt_head, reasoning, n=None):
        a = self.a
        p = prompt_head + reasoning.rstrip() + CLOSE
        for attempt in range(4):
            try:
                r = await rep.client.completions.create(model="qwen", prompt=p, max_tokens=a.forced_max_tokens, temperature=0.7, top_p=0.95, n=n or a.n_forced)
                return [FENCE + "python\n" + c.text for c in r.choices], r.usage.completion_tokens
            except Exception as e:
                self.emit({"event": "forced_error", "err": repr(e)[:200], "attempt": attempt})
                await asyncio.sleep(30 * (attempt + 1))
                if time.time() > self.deadline - 300:
                    break
        return [], 0

    def run_program(self, code, task):
        req = {"code": code, "train": task["train"], "test_inputs": [t["input"] for t in task["test"]]}
        try:
            p = subprocess.run([sys.executable, "-c", EXEC_DEMO], input=json.dumps(req), capture_output=True, text=True, timeout=90)
            return json.loads(p.stdout.strip().splitlines()[-1])
        except Exception as e:
            return {"n_train": len(task["train"]), "n_pass": 0, "demo": [], "preds": [None] * len(task["test"]), "error": "exec: " + repr(e)[:120]}

    async def execute(self, codes, task):
        loop = asyncio.get_running_loop()
        return await asyncio.gather(*(loop.run_in_executor(self.pool, self.run_program, c, task) for c in codes if c))

    @staticmethod
    def verified(r):
        return r["n_pass"] == r["n_train"] and r.get("preds") and all(p is not None for p in r["preds"])

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

    async def finish_task(self, tid, task, results, rec):
        picks = self.vote(results, task)
        for j, ranked in enumerate(picks):
            if ranked:
                self.submission[tid][j] = {"attempt_1": ranked[0], "attempt_2": ranked[1] if len(ranked) > 1 else ranked[0]}
        await self.write_submission()
        rec["n_verified"] = sum(1 for r in results if self.verified(r))
        self.note_done(rec)
        if self.sol is not None:
            gold = self.sol[tid]
            rec["correct_top1"] = [self.submission[tid][j]["attempt_1"] == gold[j] for j in range(len(task["test"]))]
            rec["correct_top2"] = [self.submission[tid][j]["attempt_1"] == gold[j] or self.submission[tid][j]["attempt_2"] == gold[j] for j in range(len(task["test"]))]
        self.emit(rec)

    async def solve(self, tid, rep):
        a = self.a
        task = self.tasks[tid]
        t_begin = self.now()
        if time.time() > self.main_deadline - a.min_trace_s:
            self.emit({"event": "skipped", "task": tid})
            return
        self.not_started -= 1
        prompt = self.head(task)
        plen = len(self.tok(prompt, add_special_tokens=False)["input_ids"])
        # prompt, reasoning and the forced program have to fit in one server context
        cap = max(2048, min(self.choose_cap(), a.max_len - plen - a.forced_max_tokens - 64))
        self.inflight_caps[tid] = cap
        self.inflight_start[tid] = time.time()
        rep.active += 1
        self.cap_sum += cap
        self.cap_n += 1
        self.emit({"event": "start", "task": tid, "cap": cap, "plen": plen, "replica": rep.port, "thr": round(self.thr_est(), 1), "left_s": round(self.main_deadline - time.time()), "not_started": self.not_started})
        try:
            fb = a.fb != "off" and cap > a.fb_ckpt + 4096
            seg1 = a.fb_ckpt if fb else cap
            text, finish, ntok = "", None, 0
            for attempt in range(4):
                try:
                    text, finish, ntok = await self.stream_text(rep, prompt, seg1)
                    break
                except Exception as e:
                    self.emit({"event": "trace_error", "task": tid, "err": repr(e)[:200], "attempt": attempt})
                    await asyncio.sleep(45 * (attempt + 1))
                    if time.time() > self.main_deadline - a.min_trace_s:
                        return
            else:
                return
            rec = {"event": "done", "task": tid, "cap": cap, "replica": rep.port, "fb": a.fb if fb else "off"}
            own = extract_code(text.split("</think>")[-1]) if "</think>" in text and finish == "stop" else None
            reasoning_tokens, forced_tokens = ntok, 0
            if own:
                results = await self.execute([own], task)
                rec.update({"finish": "stop", "ntok": reasoning_tokens, "forced_tokens": 0, "n_programs": 1, "seconds": round(self.now() - t_begin, 1)})
                await self.finish_task(tid, task, results, rec)
                return
            reasoning = text.split("</think>")[0] if "</think>" in text else text
            if fb and finish == "length":
                # checkpoint: a few programs from the prefix; a program that passes every demo ends the task, otherwise its failure is fed back
                comps, ft = await self.forced(rep, prompt, reasoning, n=a.fb_n1)
                forced_tokens += ft
                codes = [c for c in (extract_code(x) for x in comps) if c]
                res1 = await self.execute(codes, task)
                if any(self.verified(r) for r in res1):
                    rec.update({"finish": "verified_at_checkpoint", "ntok": reasoning_tokens, "forced_tokens": forced_tokens, "n_programs": len(codes), "seconds": round(self.now() - t_begin, 1)})
                    await self.finish_task(tid, task, res1, rec)
                    return
                if codes:
                    best = min(range(len(codes)), key=lambda i: (-res1[i]["n_pass"], closeness(task, res1[i]), i))
                    k, n = res1[best]["n_pass"], res1[best]["n_train"]
                    det = detail_text(task, res1[best])
                    if a.fb == "A":
                        prompt2 = prompt + reasoning + inject_a(codes[best], k, n, det)
                    else:
                        prompt2 = self.head_from_messages(messages_b(task, codes[best], k, n, det))
                    seg2 = max(1024, cap - seg1)
                    t2, finish, n2 = await self.stream_text(rep, prompt2, seg2)
                    reasoning_tokens += n2
                    own2 = extract_code(t2.split("</think>")[-1]) if "</think>" in t2 and finish == "stop" else None
                    if own2:
                        results = await self.execute([own2], task)
                        rec.update({"finish": "stop", "ntok": reasoning_tokens, "forced_tokens": forced_tokens, "n_programs": 1, "seconds": round(self.now() - t_begin, 1)})
                        await self.finish_task(tid, task, results, rec)
                        return
                    prompt, reasoning = prompt2, (t2.split("</think>")[0] if "</think>" in t2 else t2)
                    text = t2
            comps, ft = await self.forced(rep, prompt, reasoning)
            forced_tokens += ft
            codes = [c for c in (extract_code(x) for x in comps) if c]
            results = await self.execute(codes, task)
            rec.update({"finish": finish, "ntok": reasoning_tokens, "forced_tokens": forced_tokens, "n_programs": len(codes), "seconds": round(self.now() - t_begin, 1)})
            await self.finish_task(tid, task, results, rec)
        finally:
            rep.active -= 1
            self.inflight_caps.pop(tid, None)
            self.inflight_start.pop(tid, None)

    async def run(self):
        a = self.a
        await self.write_submission()
        order = [t for t in self.tasks if t not in self.done_tasks]
        random.Random(a.seed).shuffle(order)
        self.not_started = len(order)
        self.target_streams = a.per_replica * len(self.replicas)
        queue = asyncio.Queue()
        for t in order:
            queue.put_nowait(t)
        self.emit({"event": "run_start", "n_tasks": len(order), "resumed": len(self.done_tasks), "max_cap": a.max_cap, "streams": self.target_streams, "total_s": a.total_s, "fb": a.fb})
        mon = asyncio.create_task(self.monitor())

        async def worker(rep):
            while True:
                try:
                    tid = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                await self.solve(tid, rep)

        await asyncio.gather(*(worker(r) for r in self.replicas for _ in range(a.per_replica)))
        mon.cancel()
        await self.write_submission()
        self.emit({"event": "end", "mean_cap": round(self.cap_sum / max(self.cap_n, 1))})


def make_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--solutions", default="")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", default="submission.json")
    ap.add_argument("--log", default="solver_log.jsonl")
    ap.add_argument("--ports", default="8000")
    ap.add_argument("--per-replica", type=int, default=24)
    ap.add_argument("--max-cap", type=int, default=49152)
    ap.add_argument("--min-cap", type=int, default=8192)
    ap.add_argument("--thr-prior", type=float, default=330.0, help="tokens per second for the whole server group until it has been measured")
    ap.add_argument("--safety", type=float, default=0.92)
    ap.add_argument("--stream-rate", type=float, default=9.0, help="pessimistic tokens per second of one stream; a trace never gets more than this x the time left")
    ap.add_argument("--n-forced", type=int, default=8)
    ap.add_argument("--forced-max-tokens", type=int, default=4000)
    ap.add_argument("--max-len", type=int, default=65536, help="context length of the servers")
    ap.add_argument("--fb", default="off", choices=["off", "A", "B"])
    ap.add_argument("--fb-ckpt", type=int, default=16384)
    ap.add_argument("--fb-n1", type=int, default=4)
    ap.add_argument("--exec-workers", type=int, default=16)
    ap.add_argument("--t-start", type=float, default=time.time())
    ap.add_argument("--total-s", type=float, default=11.5 * 3600)
    ap.add_argument("--reserve-s", type=float, default=1500)
    ap.add_argument("--min-trace-s", type=float, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--effort", default="", help="reasoning effort of the chat template (xhigh when empty; medium and low exist)")
    ap.add_argument("--cap-step", type=int, default=1536, help="the cap of a task is at most this many tokens above the one before")
    ap.add_argument("--cap-step-down", type=int, default=4096, help="the cap of a task is at most this many tokens below the one before")
    ap.add_argument("--first-wave-lo", type=float, default=0.85, help="the caps of the first wave of tasks are spread over [this x the cap, the cap] so the streams do not finish together")
    ap.add_argument("--prior-tasks", type=float, default=6.0, help="weight of the throughput prior, in tasks of 40K decoded tokens")
    ap.add_argument("--prompt-summary", action="store_true", help="add code-computed facts about the grids (colors, objects, changes) to the prompt")
    ap.add_argument("--prompt-extra", default="", help="text added to the instructions of the prompt (after the header, before the examples)")
    ap.add_argument("--cost-points", default="", help="JSON list of [cap, mean decoded tokens per task] pairs measured for this effort; replaces the built-in table")
    return ap


def main():
    a = make_parser().parse_args()
    if a.cost_points:
        COST_POINTS[:] = [tuple(p) for p in json.loads(a.cost_points)]
    global PROMPT_EXTRA, PROMPT_SUMMARY
    if a.prompt_extra:
        PROMPT_EXTRA = a.prompt_extra + "\n\n"
    PROMPT_SUMMARY = a.prompt_summary
    asyncio.run(Solver(a).run())


if __name__ == "__main__":
    main()
