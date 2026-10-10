#!/usr/bin/env python
"""Tool-integrated reasoning on ARC-AGI-2: the model reasons in turns and runs Python in a persistent per-task session, then writes a final program.

Why: a long reasoning trace spends most of its tokens simulating grids by hand. A tool call lets the model test a hypothesis on the demo pairs exactly and read the result back
in a few hundred prompt tokens, which prefix caching makes cheap. The transcript is built as text in the chat template's own format (assistant turn with <think> reasoning, then a
<tool_call> block; the result comes back as a user turn holding <tool_response>), so every turn extends the previous token sequence and the prefix cache keeps hitting.

Per task and rollout: turns until the model answers without a tool call, the token budget is used up, or the turn limit is reached. The final answer is the last ```python block
(a function transform); when the budget ends first, the transcript is closed with a fixed sentence and --n-forced programs are sampled. Programs run in a sandbox on the demo pairs
and on every test input; demo-passing programs weigh 2, partial passes 0.25 x fraction (the voting of solver2.py). One JSON line per turn and per finished rollout is appended to --log.
"""
import argparse, ast, asyncio, gzip, json, os, random, re, subprocess, sys, tempfile, time
from collections import defaultdict

from openai import AsyncOpenAI
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
FENCE = "`" * 3

def make_tools(autocheck=False):
    helpers = ("`show(grid)` prints a grid as rows of digits and `summarize()` prints shapes, colors, objects and changes of every pair. ")
    if autocheck:
        helpers = ("`show(grid)` prints a grid as rows of digits, `summarize()` prints shapes, colors, objects and changes of every pair, and `check(f)` runs a function on every training pair and the test "
                   "input and reports which pairs pass and which cells differ. Whenever a call defines or redefines `transform`, the session runs `check(transform)` by itself and appends the report to the output. ")
    return [{"type": "function", "function": {
        "name": "execute_python",
        "description": ("Run Python code in a persistent session (variables and functions defined in one call stay available in the next). numpy is imported as np. "
                        "`train` is a list of {'input': grid, 'output': grid} (grids are lists of lists of ints) and `test_input` is the test grid. " + helpers +
                        "Whatever the code prints, and the value of its last expression, comes back as text."),
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to run."}}, "required": ["code"]}}}]


TOOLS = make_tools(False)

HEADER = (
    "You are participating in a puzzle solving competition. You are an expert programmer and puzzle solver.\n\n"
    "Below is a list of input and output grid pairs that share one transformation rule. Your goal is to find the rule and write a Python function `transform(grid)` that maps any "
    "input grid to its output grid. `grid` is a list of lists of ints (0-9); return a list of lists of ints. You may import numpy. The function must reproduce every training "
    "output exactly and must generalize to the test input.\n\n"
    "Grids are shown as rows of digits 0-9, one row per line, no separators. Each digit is a color.\n\n"
    "You have a Python tool, `execute_python`, with a persistent session. The training pairs are preloaded as `train` and the test input as `test_input`. Use the tool to check "
    "your hypotheses on the data instead of simulating grids by hand: measure shapes, colors, objects and symmetries, and run candidate `transform` functions on every training "
    "pair and compare with the expected output. When your function reproduces every training output, run it on `test_input` and check that the result looks plausible.\n\n"
    "When you are done, reply with the complete final code (imports plus the function `transform`) inside one fenced " + FENCE + "python code block and write nothing after the code block.\n\n")

FORCE_TOOL = "\n\nLet me test my current idea with code.\n</think>\n\n<tool_call>\n<function=execute_python>\n<parameter=code>\n"
CLOSE_FORCED = "\n\nI have run out of budget. I will now write my best final code.\n</think>\n\n" + FENCE + "python\n"

WORKER = r'''
import ast, contextlib, io, json, resource, signal, sys, time, traceback
try:
    resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))
except (ValueError, OSError):
    pass
proto_in, proto_out = sys.stdin, sys.__stdout__
sys.stdin = io.StringIO("")
init = json.loads(proto_in.readline())
G = {"__name__": "__repl__"}
exec("import numpy as np, itertools, collections, math, copy, functools, re", G)
G["train"] = init["train"]
G["test_input"] = init["test_input"]
G["test_inputs"] = init["test_inputs"]
AUTOCHECK = bool(init.get("autocheck"))
LAST = [None]
exec("def show(g):\n    print('\\n'.join(''.join(str(int(c)) for c in r) for r in g))", G)


class Timeout(Exception):
    pass


def on_alarm(*_):
    raise Timeout()


signal.signal(signal.SIGALRM, on_alarm)

exec("""
def summarize(pairs=None, test=None):
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

    pairs = pairs if pairs is not None else train
    for i, ex in enumerate(pairs):
        a, b = np.array(ex["input"]), np.array(ex["output"])
        print("Example", i)
        print("  " + desc(a, "input"))
        print("  " + desc(b, "output"))
        if a.shape == b.shape:
            d = a != b
            n = int(d.sum())
            tr = Counter((int(x), int(y)) for x, y in zip(a[d], b[d]))
            if n:
                ys, xs = np.where(d)
                print("  same shape; " + str(n) + " cells change (rows " + str(ys.min()) + "-" + str(ys.max()) + ", cols " + str(xs.min()) + "-" + str(xs.max()) + "); color changes (from, to): " + ", ".join(str(k) + " x" + str(v) for k, v in tr.most_common(8)))
            else:
                print("  same shape; no cell changes")
        else:
            note = []
            if b.shape[0] % a.shape[0] == 0 and b.shape[1] % a.shape[1] == 0:
                fy, fx = b.shape[0] // a.shape[0], b.shape[1] // a.shape[1]
                note.append("output is " + str(fy) + "x" + str(fx) + " times the input size" + (" and equals the input tiled" if (np.tile(a, (fy, fx)) == b).all() else "") + (" and equals the input scaled up" if (np.kron(a, np.ones((fy, fx), int)) == b).all() else ""))
            if b.shape[0] <= a.shape[0] and b.shape[1] <= a.shape[1]:
                hits = [(y, x) for y in range(a.shape[0] - b.shape[0] + 1) for x in range(a.shape[1] - b.shape[1] + 1) if (a[y:y + b.shape[0], x:x + b.shape[1]] == b).all()]
                if hits: note.append("output is a sub-grid of the input at row, col " + str(hits[0]))
            print("  different shape" + ("; " + "; ".join(note) if note else ""))
    tests = [test] if test is not None else test_inputs
    for k, t_ in enumerate(tests):
        print("Test input" + (" %d" % k if len(tests) > 1 else ""))
        print("  " + desc(np.array(t_), "input"))
""", G)

def check_text(f=None, until=None):
    import time as _t
    f = f if f is not None else G.get("transform")
    if not callable(f):
        return "no function `transform` is defined yet"
    np = G["np"]
    end = min(_t.time() + 12, until) if until else _t.time() + 12

    def run(g):
        if _t.time() > end:
            return None, "skipped (the check ran out of time)"
        signal.alarm(4)
        try:
            a = np.asarray(f([list(map(int, r)) for r in g]))
            if a.dtype == object or a.ndim != 2 or a.size == 0:
                return None, "the result is not a non-empty 2D list of lists"
            if not np.issubdtype(a.dtype, np.integer):
                if np.issubdtype(a.dtype, np.floating) and np.all(a == np.round(a)):
                    a = a.astype(int)
                else:
                    return None, "the result has non-integer values"
            if a.min() < 0 or a.max() > 9:
                return None, "the result has values outside 0-9"
            return a.astype(int), None
        except Timeout:
            return None, "timed out after 4 s"
        except BaseException as e:
            return None, (type(e).__name__ + ": " + str(e))[:120]
        finally:
            signal.alarm(0)

    lines, ok = [], 0
    for i, ex in enumerate(G["train"]):
        o, err = run(ex["input"])
        exp = np.array(ex["output"])
        if err:
            lines.append("train %d: error: %s" % (i, err))
        elif o.shape != exp.shape:
            lines.append("train %d: wrong shape %dx%d, expected %dx%d" % (i, o.shape[0], o.shape[1], exp.shape[0], exp.shape[1]))
        else:
            d = np.argwhere(o != exp)
            if len(d) == 0:
                ok += 1
                lines.append("train %d: pass" % i)
            else:
                lines.append("train %d: %d of %d cells differ, e.g. %s" % (i, len(d), exp.size, "; ".join("(%d,%d) got %d expected %d" % (y, x, o[y, x], exp[y, x]) for y, x in d[:4])))
    out = "[check of `transform`: %d of %d training pairs pass]\n" % (ok, len(G["train"])) + "\n".join(lines)
    for k, t in enumerate(G["test_inputs"]):
        o, err = run(t)
        if err:
            out += "\ntest input %d: error: %s" % (k, err)
        else:
            out += "\ntest input %d: output %dx%d" % (k, o.shape[0], o.shape[1])
            if k == 0 and ok == len(G["train"]) and o.shape[0] <= 30:
                out += "\n" + "\n".join("".join(str(c) for c in r) for r in o)
    return out


def check(f=None):
    """Run f (default: the session's `transform`) on every training pair and the test inputs and print what differs."""
    print(check_text(f))


G["check"] = check

for line in proto_in:
    req = json.loads(line)
    t_req = time.time()
    buf = io.StringIO()
    signal.alarm(int(req.get("timeout", 20)))
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            tree = ast.parse(req["code"], "<cell>")
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            exec(compile(tree, "<cell>", "exec"), G)
            if last is not None:
                val = eval(compile(ast.Expression(last.value), "<cell>", "eval"), G)
                if val is not None:
                    print(repr(val))
    except Timeout:
        buf.write("\nTimeoutError: the code ran longer than %s seconds" % req.get("timeout", 20))
    except BaseException as e:
        et, ev, tb = sys.exc_info()
        frames = [f for f in traceback.extract_tb(tb) if f.filename == "<cell>"]
        lines = ["Traceback (most recent call last):"] + ["  line %d, in %s" % (f.lineno, f.name) for f in frames[-4:]] + [x.rstrip() for x in traceback.format_exception_only(et, ev)]
        buf.write("\n" + "\n".join(lines))
    finally:
        signal.alarm(0)
    chk = ""
    if AUTOCHECK:
        f_ = G.get("transform")
        code_ = getattr(f_, "__code__", None)
        if callable(f_) and code_ is not None and code_ is not LAST[0]:
            LAST[0] = code_
            try:
                chk = "\n" + check_text(f_, t_req + int(req.get("timeout", 20)) + 6)
            except BaseException:
                chk = ""
    out = buf.getvalue()
    if chk and "[check of `transform`" in out:
        chk = ""
    lim = int(req.get("limit", 2500))
    room = max(600, lim - len(chk))
    if len(out) > room:
        out = out[:room] + "\n...[output truncated, %d characters in total]" % len(out)
    out += chk
    proto_out.write(json.dumps({"out": out}) + "\n")
    proto_out.flush()
'''


def grid_text(g):
    return "\n".join("".join(str(c) for c in r) for r in g)


def build_user(task, ti=0, all_tests=False):
    """The prompt. With all_tests every test input is listed (and the header says one function must serve all of them); otherwise only test input ti."""
    multi = all_tests and len(task["test"]) > 1
    parts = [HEADER + (f"This puzzle has {len(task['test'])} test inputs, listed after the training examples. One function `transform` must be correct for all of them; in the session they are the list `test_inputs`.\n\n" if multi else ""), "--Training Examples--\n"]
    for i, ex in enumerate(task["train"]):
        parts.append(f"--Example {i}--\nINPUT:\n{grid_text(ex['input'])}\nOUTPUT:\n{grid_text(ex['output'])}\n")
    if multi:
        for j, t in enumerate(task["test"]):
            parts.append(f"--Test Input {j}--\n{grid_text(t['input'])}\n")
    else:
        parts.append(f"--Test Input--\n{grid_text(task['test'][ti]['input'])}\n")
    return "\n".join(parts)


class Repl:
    """One persistent Python process per rollout, driven over pipes with a timeout per call."""

    def __init__(self, worker_path, task, ti, autocheck=False):
        self.worker_path, self.task, self.ti, self.proc, self.started, self.autocheck = worker_path, task, ti, None, False, autocheck

    async def start(self):
        self.proc = await asyncio.create_subprocess_exec(sys.executable, self.worker_path, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        self.started = True
        init = {"train": self.task["train"], "test_input": self.task["test"][self.ti]["input"], "test_inputs": [t["input"] for t in self.task["test"]], "autocheck": self.autocheck}
        self.proc.stdin.write((json.dumps(init) + "\n").encode())
        await self.proc.stdin.drain()

    async def run(self, code, timeout=20, limit=2500):
        note = ""
        if self.proc is None or self.proc.returncode is not None:
            restarted = self.started
            await self.start()
            if restarted:
                note = "(the session had ended and was restarted; earlier variables are gone)\n"
        try:
            self.proc.stdin.write((json.dumps({"code": code, "timeout": timeout, "limit": limit}) + "\n").encode())
            await self.proc.stdin.drain()
            line = await asyncio.wait_for(self.proc.stdout.readline(), timeout + 10)
            if not line:
                raise RuntimeError("session closed")
            return note + json.loads(line)["out"]
        except Exception as e:
            try:
                self.proc.kill()
            except ProcessLookupError:
                pass
            self.proc = None
            return note + f"(the session crashed or hung: {type(e).__name__}; it will be restarted on the next call and earlier variables are gone)"

    async def close(self):
        if self.proc and self.proc.returncode is None:
            try:
                self.proc.kill()
            except ProcessLookupError:
                pass
            await self.proc.wait()


CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_calls(text):
    out = []
    for m in CALL_RE.finditer(text):
        params = {k: v for k, v in PARAM_RE.findall(m.group(2))}
        out.append((m.group(1), params))
    return out


def extract_code(text):
    blocks = re.findall(FENCE + r"(?:python|py)?\n(.*?)" + FENCE, text or "", re.S)
    return blocks[-1] if blocks else None


def tkey(g):
    return tuple(map(tuple, g)) if g else None


def run_program(code, task):
    req = {"code": code, "train": task["train"], "test_inputs": [t["input"] for t in task["test"]]}
    try:
        p = subprocess.run([sys.executable, os.path.join(HERE, "exec_demo.py")], input=json.dumps(req), capture_output=True, text=True, timeout=90)
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"n_train": len(task["train"]), "n_pass": 0, "demo": [], "preds": [None] * len(task["test"]), "error": "exec: " + repr(e)[:120]}


def vote(results, n_test):
    out = []
    for j in range(n_test):
        w = defaultdict(float)
        for r in results:
            ps = r.get("preds") or []
            p = ps[j] if j < len(ps) else None
            if p is None:
                continue
            w[tkey(p)] += 2.0 if r["n_pass"] == r["n_train"] else 0.25 * r["n_pass"] / max(r["n_train"], 1)
        out.append([g for g, _ in sorted(w.items(), key=lambda kv: -kv[1])])
    return out


class Agent:
    def __init__(self, a):
        self.a = a
        self.tok = AutoTokenizer.from_pretrained(a.model_dir)
        self.clients = [AsyncOpenAI(base_url=f"http://127.0.0.1:{p}/v1", api_key="x", timeout=86400, max_retries=0) for p in a.ports.split(",")]
        self.tasks = json.load(open(a.challenges))
        if a.tasks:
            keep = json.load(open(a.tasks)) if os.path.exists(a.tasks) else a.tasks.split(",")
            self.tasks = {k: v for k, v in self.tasks.items() if k in set(keep)}
        self.sol = json.load(open(a.solutions)) if a.solutions else None
        self.worker_path = os.path.join(tempfile.gettempdir(), "arc_repl_worker.py")
        open(self.worker_path, "w").write(WORKER)
        os.makedirs(a.traces, exist_ok=True)
        self.log_f = open(a.log, "a")
        self.done = set()
        if os.path.exists(a.log):
            for line in open(a.log):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("event") == "done":
                    self.done.add((r["task"], r["rollout"]))
        self.t0 = time.time()

    def emit(self, rec):
        rec["t"] = round(time.time() - self.t0, 1)
        self.log_f.write(json.dumps(rec) + "\n")
        self.log_f.flush()

    def head(self, task):
        kw = {"reasoning_effort": self.a.effort} if self.a.effort else {}
        h = self.tok.apply_chat_template([{"role": "user", "content": build_user(task, all_tests=self.a.all_tests)}], tools=make_tools(self.a.auto_check), tokenize=False, add_generation_prompt=True, **kw)
        return h if h.rstrip().endswith("<think>") else h + "<think>\n"

    async def complete(self, client, prompt, max_tokens, temperature=1.0, n=1, top_p=0.95):
        r = await client.completions.create(model="qwen", prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, extra_body={"top_k": 20})
        return r

    async def rollout(self, tid, k, client, sem):
        a = self.a
        task = self.tasks[tid]
        async with sem:
            t_begin = time.time()
            transcript = self.head(task)
            repl = Repl(self.worker_path, task, 0, autocheck=a.auto_check)
            if a.auto_summary:
                out = await repl.run("summarize()", timeout=a.call_timeout, limit=6000)
                transcript += ("Let me start by summarizing the grids with code.\n</think>\n\n<tool_call>\n<function=execute_python>\n<parameter=code>\nsummarize()\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
                               "<|im_start|>user\n<tool_response>\n" + out + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n")
            gen_total, turns, calls, final_text, finish, status = 0, 0, 0, "", None, "budget"
            try:
                while True:
                    remaining = a.budget - gen_total
                    if remaining < 1024 or turns >= a.max_turns:
                        break
                    turns += 1
                    think_cap = a.think_cap if a.think_cap else a.turn_cap
                    resp = await self.complete(client, transcript, min(remaining, think_cap))
                    ch = resp.choices[0]
                    text, finish, ntok = ch.text or "", ch.finish_reason, resp.usage.completion_tokens
                    gen_total += ntok
                    forced = False
                    # the cap limits thinking only: once </think> is out the turn (answer or tool call) may run on
                    conts = 0
                    while finish == "length" and "</think>" in text and gen_total < a.budget and conts < 4:
                        r3 = await self.complete(client, transcript + text, min(a.budget - gen_total, 4096))
                        text += r3.choices[0].text or ""
                        finish = r3.choices[0].finish_reason
                        gen_total += r3.usage.completion_tokens
                        ntok += r3.usage.completion_tokens
                        conts += 1
                    if a.think_cap and finish == "length" and "</think>" not in text and remaining - ntok > 1500:
                        # still thinking at the cap: end the thought and make the model write a test of its current idea
                        r2 = await self.complete(client, transcript + text + FORCE_TOOL, 1500)
                        text = text + FORCE_TOOL + (r2.choices[0].text or "")
                        if "</tool_call>" not in text:
                            text += "\n</parameter>\n</function>\n</tool_call>"
                        finish = "stop"
                        gen_total += r2.usage.completion_tokens
                        ntok += r2.usage.completion_tokens
                        forced = True
                    after = text.split("</think>")[-1] if "</think>" in text else text
                    cs = parse_calls(after) if finish == "stop" else []
                    self.emit({"event": "turn", "task": tid, "rollout": k, "turn": turns, "ntok": ntok, "finish": finish, "calls": len(cs), "gen_total": gen_total, "forced_tool": forced, "conts": conts})
                    if finish != "stop":
                        transcript += text
                        status = "turn_cut"
                        break
                    if not cs:
                        transcript += text + "<|im_end|>\n"
                        final_text, status = after, "answered"
                        break
                    transcript += text + "<|im_end|>\n<|im_start|>user"
                    for name, params in cs:
                        calls += 1
                        if name == "execute_python" and "code" in params:
                            out = await repl.run(params["code"], timeout=a.call_timeout, limit=a.out_limit)
                        else:
                            out = f"Error: unknown tool or missing argument (tool {name}, arguments {sorted(params)}). Use execute_python with a `code` argument."
                        transcript += "\n<tool_response>\n" + out + "\n</tool_response>"
                    transcript += "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            finally:
                await repl.close()
            codes, forced_tokens = [], 0
            own = extract_code(final_text) if status == "answered" else None
            if own:
                codes = [own]
            else:
                # budget used up, a cut turn, or an answer without code: close the transcript where it is and sample programs
                closed = transcript.rstrip("\n") + CLOSE_FORCED if status != "turn_cut" else transcript.rstrip() + CLOSE_FORCED
                if status == "answered":
                    closed = transcript[: transcript.rindex("<|im_start|>assistant")] + "<|im_start|>assistant\n<think>\n" + "\n\nI will now write my final code.\n</think>\n\n" + FENCE + "python\n"
                for attempt in range(3):
                    try:
                        r = await self.complete(client, closed, 4000, temperature=0.7, n=a.n_forced)
                        codes = [c for c in (extract_code(FENCE + "python\n" + x.text) for x in r.choices) if c]
                        forced_tokens = r.usage.completion_tokens
                        break
                    except Exception as e:
                        self.emit({"event": "forced_error", "task": tid, "rollout": k, "err": repr(e)[:160]})
                        await asyncio.sleep(10)
            loop = asyncio.get_running_loop()
            results = await asyncio.gather(*(loop.run_in_executor(self.pool, run_program, c, task) for c in codes))
            ranked = vote(results, len(task["test"]))
            rec = {"event": "done", "task": tid, "rollout": k, "status": status, "turns": turns, "calls": calls, "gen_tokens": gen_total, "forced_tokens": forced_tokens, "n_programs": len(codes),
                   "n_verified": sum(1 for r in results if r["n_pass"] == r["n_train"] and r.get("preds") and all(p is not None for p in r["preds"])), "seconds": round(time.time() - t_begin, 1),
                   "answers": [[[list(r) for r in g] for g in rk[:2]] for rk in ranked]}
            if self.sol is not None:
                gold = self.sol[tid]
                rec["correct_top1"] = [bool(rk) and [list(r) for r in rk[0]] == gold[j] for j, rk in enumerate(ranked)]
                rec["correct_top2"] = [any([list(r) for r in g] == gold[j] for g in rk[:2]) for j, rk in enumerate(ranked)]
            self.emit(rec)
            with gzip.open(os.path.join(self.a.traces, f"{tid}_{k}.txt.gz"), "wt") as f:
                f.write(transcript)

    async def run(self):
        from concurrent.futures import ThreadPoolExecutor
        a = self.a
        self.pool = ThreadPoolExecutor(a.exec_workers)
        order = [(t, k) for t in self.tasks for k in range(a.rollouts) if (t, k) not in self.done]
        random.Random(a.seed).shuffle(order)
        self.emit({"event": "run_start", "n_rollouts": len(order), "budget": a.budget, "turn_cap": a.turn_cap, "effort": a.effort or "xhigh"})
        sems = [asyncio.Semaphore(a.per_replica) for _ in self.clients]
        await asyncio.gather(*(self.rollout(t, k, self.clients[i % len(self.clients)], sems[i % len(self.clients)]) for i, (t, k) in enumerate(order)))
        self.emit({"event": "end"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--solutions", default="")
    ap.add_argument("--tasks", default="", help="JSON list file or comma list of task ids")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--ports", default="8001,8002,8003,8004")
    ap.add_argument("--per-replica", type=int, default=16)
    ap.add_argument("--budget", type=int, default=40000, help="generated tokens per rollout, all turns together")
    ap.add_argument("--turn-cap", type=int, default=16384)
    ap.add_argument("--think-cap", type=int, default=0, help="when a thinking turn reaches this many tokens without a tool call, end the thought and force a tool call (0 = off)")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--call-timeout", type=int, default=20)
    ap.add_argument("--out-limit", type=int, default=2500)
    ap.add_argument("--n-forced", type=int, default=4)
    ap.add_argument("--rollouts", type=int, default=1)
    ap.add_argument("--effort", default="")
    ap.add_argument("--auto-summary", action="store_true", help="start every rollout with a harness-made summarize() call and its output, as if the model had called it")
    ap.add_argument("--all-tests", action="store_true", help="list every test input in the prompt (default: the first one only)")
    ap.add_argument("--auto-check", action="store_true", help="after every call that defines or redefines `transform`, run it on the training pairs and the test input and append the report to the output")
    ap.add_argument("--exec-workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    asyncio.run(Agent(a).run())


if __name__ == "__main__":
    main()
