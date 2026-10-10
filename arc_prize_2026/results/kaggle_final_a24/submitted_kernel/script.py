# Shared helpers for Kaggle L4x4 scripts: wheelhouse install, multi-server vLLM management, dataset discovery, metrics, benchmark streams.
import asyncio, glob, gzip, json, os, random, signal, subprocess, sys, time, urllib.request

T0 = time.time()
WORK = "/kaggle/working"


def sh(cmd, tail=2500, quiet=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if not quiet:
        print(f"[{time.time() - T0:6.0f}s] $ {cmd[:170]}\n{(r.stdout + r.stderr)[-tail:]}", flush=True)
    return r


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def find_dir(name, files=None, depth=4):
    """Directory under /kaggle/input whose basename is `name` (or that holds all of `files`)."""
    base = "/kaggle/input"
    for root, dirs, fs in os.walk(base):
        if root[len(base):].count("/") > depth:
            dirs[:] = []
            continue
        if files:
            if all(f in fs for f in files):
                return root
        elif os.path.basename(root) == name:
            return root
    return None


def install_wheelhouse():
    wh = find_dir("wh")
    whls = [w for w in sorted(glob.glob(wh + "/*.whl")) if "/triton-" not in w and "torchcodec" not in w]
    log(len(whls), "wheels from", wh)
    sh("pip install --no-index --no-deps --find-links " + wh + " " + " ".join(f"'{w}'" for w in whls), tail=300)
    sh("ln -sf /usr/local/cuda-12.8/compat/libcuda.so.1 /usr/local/lib/libcuda.so", quiet=True)
    sh("python -c \"import vllm, torch; print(vllm.__version__, torch.__version__, torch.cuda.device_count())\" 2>&1 | tail -3")


SERVERS = []


def stop_servers():
    global SERVERS
    for p in SERVERS:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except Exception:
                pass
    for p in SERVERS:
        try:
            p.wait(90)
        except Exception:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass
    SERVERS = []
    time.sleep(8)


def start_servers(name, model, groups, tp, port0=8000, extra=(), max_len=49152, util=0.90, max_seqs=24, kv="fp8", max_batched=4096):
    """One vLLM server per device group (list of GPU index lists); returns the ports. `model` is one path or one path per group."""
    global SERVERS
    stop_servers()
    ports = []
    models = list(model) if isinstance(model, (list, tuple)) else [model] * len(groups)
    for i, devs in enumerate(groups):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, devs)), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_USE_FLASHINFER_SAMPLER="0",
                   VLLM_CACHE_ROOT=f"/tmp/vllm_cache_{name}_{i}", TOKENIZERS_PARALLELISM="false")
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", models[i], "--served-model-name", "qwen", "--tensor-parallel-size", str(tp),
               "--max-model-len", str(max_len), "--gpu-memory-utilization", str(util), "--kv-cache-dtype", kv, "--max-num-seqs", str(max_seqs),
               "--max-num-batched-tokens", str(max_batched), "--reasoning-parser", "qwen3", "--port", str(port0 + i), *extra]
        log("CMD", " ".join(cmd[2:]), "devices", devs)
        SERVERS.append(subprocess.Popen(cmd, stdout=open(f"{WORK}/vllm_{name}_{i}.log", "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True))
        ports.append(port0 + i)
    return ports


def wait_ready(name, ports, timeout=2400):
    t = time.time()
    last = 0
    while time.time() - t < timeout:
        for i, p in enumerate(SERVERS):
            if p.poll() is not None:
                log("server", i, "exited", p.returncode)
                print(sh(f"grep -E 'Error|error|Traceback' {WORK}/vllm_{name}_{i}.log | tail -n 12", quiet=True).stdout[-2500:], flush=True)
                return False
        ok = 0
        for port in ports:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3).read()
                ok += 1
            except Exception:
                pass
        if ok == len(ports):
            log(f"READY after {time.time() - t:.0f}s")
            for i in range(len(ports)):
                sh(f"grep -E 'KV cache size|Maximum concurrency|Using .*(backend|Backend)' {WORK}/vllm_{name}_{i}.log | sed -E 's/\\(Worker_TP[0-9] pid=[0-9]+\\) //' | cut -c1-200 | sort -u | head -n 6", tail=1500)
            return True
        if time.time() - last > 180:
            last = time.time()
            log(f"waiting {time.time() - t:.0f}s ({ok}/{len(ports)} up)")
        time.sleep(5)
    return False


def scrape(ports):
    d = {}
    for port in ports:
        txt = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode()
        for line in txt.splitlines():
            if not line or line[0] == "#":
                continue
            k, _, v = line.rpartition(" ")
            if k.startswith("vllm:") and any(s in k for s in ("generation_tokens_total", "prompt_tokens_total", "num_requests_running", "num_requests_waiting", "kv_cache_usage",
                                                              "num_preemptions", "spec_decode", "prefix_cache")):
                try:
                    d[k] = d.get(k, 0.0) + float(v)
                except ValueError:
                    pass
    return d


def pick(d, sub):
    return sum(v for k, v in d.items() if sub in k)


def gpu_util():
    r = sh("nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits", quiet=True).stdout.strip().splitlines()
    try:
        u = [float(x.split(",")[0]) for x in r]
        w = [float(x.split(",")[1]) for x in r]
        return round(sum(u) / len(u)), round(sum(w) / len(w))
    except Exception:
        return None, None

CFG = json.loads('{"layout": "tp4_fp8", "layouts": {"tp4_fp8": {"model": "fp8", "groups": [[0, 1, 2, 3]], "tp": 4, "max_len": 98304, "max_seqs": 48, "per_replica": 24, "thr_prior": 300}, "tp2x2_int4": {"model": "int4", "groups": [[0, 1], [2, 3]], "tp": 2, "max_len": 65536, "max_seqs": 32, "per_replica": 12, "thr_prior": 400}}, "total_s_rerun": 41400, "hard_end_rerun": 42000, "reserve_s": 1500, "max_cap": 55000, "min_cap": 8192, "n_forced": 8, "fb": "off", "fb_ckpt": 16384, "fb_n1": 4, "mtp_tokens": 2, "prefix_caching": true, "safety": 0.92, "min_trace_s": 600, "max_restarts": 4, "commit_mode": "replica", "smoke_tasks": 6, "smoke_cap": 16384, "smoke_total_s": 2400, "replica_tasks": 120, "replica_s_per_task": 163.5, "replica_seed": 11, "replica_overhead_s": 2160, "effort": "", "cost_points": [], "prompt_summary": true}')

SOLVER_SRC = '#!/usr/bin/env python\n"""ARC-AGI-2 solver v2 for one or more vLLM replicas: one reasoning trace per task, budget-forced program finalization, demo-verified voting.\n\nPer task: stream one reasoning trace from a Qwen3.8 server through the raw completions API. A trace that ends on its own yields its own program; a trace that\nreaches its token cap is closed with a fixed sentence and --n-forced programs are sampled from the truncated reasoning. Programs run in a sandbox on the demo pairs and on\nevery test input; programs that reproduce all demos weigh 2, partial passes 0.25 x fraction, and the two heaviest distinct grids become attempt_1 and attempt_2.\n\nDifferences from v1: several replicas (each task stays on one replica so a prefix cache can serve its forced step), a cap controller that picks the token cap of each new\ntrace from the measured throughput and the work left (so a slow or fast machine still finishes inside the budget), resume from the log, and an optional execution-feedback\nstage (--fb A|B): at a checkpoint the best program is run, and if no program passes every demo the first failing demo is described to the model (A: inside the thinking\nblock, B: as a fresh chat turn) before the trace continues. submission.json is rewritten atomically after every task and starts from a fallback (the test input).\n"""\nimport argparse, asyncio, json, os, random, re, subprocess, sys, time, urllib.request\nfrom collections import defaultdict, deque\nfrom concurrent.futures import ThreadPoolExecutor\n\nfrom openai import AsyncOpenAI\nfrom transformers import AutoTokenizer\n\nHEADER_CODE = (\n    "You are participating in a puzzle solving competition. You are an expert programmer and puzzle solver.\\n\\n"\n    "Below is a list of input and output grid pairs that share one transformation rule. Your goal is to find the rule "\n    "and write a Python function `transform(grid)` that maps any input grid to its output grid. "\n    "`grid` is a list of lists of ints (0-9); return a list of lists of ints. You may import numpy. "\n    "The function must reproduce every training output exactly and must generalize to the test input.\\n\\n"\n    "Grids are shown as rows of digits 0-9, one row per line, no separators. Each digit is a color.\\n"\n    "After your reasoning, give only the complete Python code (imports plus the function) inside one fenced ```python code block, "\n    "and write nothing after the code block.\\n\\n")\nCLOSE = "\\n\\nI have run out of thinking time. I will now write my best final code.\\n</think>\\n\\n```python\\n"\nFENCE = "`" * 3\n\nEXEC_DEMO = r\'\'\'\nimport json, resource, signal, sys\ntry:\n    resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))\nexcept (ValueError, OSError):\n    pass\nreq = json.load(sys.stdin)\n\n\nclass Timeout(Exception):\n    pass\n\n\ndef on_alarm(*_):\n    raise Timeout()\n\n\nsignal.signal(signal.SIGALRM, on_alarm)\n\n\ndef clean(out):\n    import numpy as np\n    a = np.asarray(out)\n    if a.dtype == object:\n        return None, "ragged or non-numeric nested list"\n    if a.ndim != 2:\n        return None, f"expected a 2D grid but got {a.ndim} dimensions"\n    if a.size == 0:\n        return None, "empty grid"\n    if a.shape[0] > 30 or a.shape[1] > 30:\n        return None, f"grid of shape {a.shape[0]}x{a.shape[1]} is larger than 30x30"\n    if not np.issubdtype(a.dtype, np.integer):\n        if np.issubdtype(a.dtype, np.floating) and np.all(a == np.round(a)):\n            a = a.astype(int)\n        else:\n            return None, "non-integer values"\n    if a.min() < 0 or a.max() > 9:\n        return None, "values outside 0-9"\n    return a.astype(int).tolist(), None\n\n\ndef run(f, g, limit=6):\n    signal.alarm(limit)\n    try:\n        return clean(f([list(map(int, r)) for r in g]))\n    except Timeout:\n        return None, f"timed out after {limit} s"\n    except BaseException as e:\n        return None, (type(e).__name__ + ": " + str(e))[:160]\n    finally:\n        signal.alarm(0)\n\n\nres = {"n_train": len(req["train"]), "n_pass": 0, "demo": [], "preds": [], "error": None}\ntry:\n    ns = {"__name__": "candidate"}\n    signal.alarm(10)\n    exec(req["code"], ns)\n    signal.alarm(0)\n    f = ns["transform"]\nexcept BaseException as e:\n    signal.alarm(0)\n    res["error"] = "compile: " + (type(e).__name__ + ": " + str(e))[:160]\n    print(json.dumps(res))\n    sys.exit(0)\nfor ex in req["train"]:\n    out, err = run(f, ex["input"])\n    res["demo"].append({"out": out, "err": err})\n    if out is not None and out == ex["output"]:\n        res["n_pass"] += 1\nres["preds"] = [run(f, g)[0] for g in req["test_inputs"]]\nprint(json.dumps(res))\n\'\'\'\n\n# mean decoded tokens per task (reasoning plus forced programs) at each cap, measured on 120 public evaluation tasks (two runs agree within 0.2K)\nCOST_POINTS = [(8192, 17.0e3), (16384, 24.7e3), (32768, 42.0e3), (49152, 58.0e3), (63000, 70.0e3)]\n\n\ndef cost_of_cap(c):\n    pts = COST_POINTS\n    if c <= pts[0][0]:\n        return pts[0][1] * c / pts[0][0] if c > 0 else 0.0\n    for (c0, v0), (c1, v1) in zip(pts, pts[1:]):\n        if c <= c1:\n            return v0 + (v1 - v0) * (c - c0) / (c1 - c0)\n    return pts[-1][1] + (c - pts[-1][0]) * 0.9\n\n\ndef extract_code(text):\n    blocks = re.findall(FENCE + r"(?:python|py)?\\n(.*?)" + FENCE, text or "", re.S)\n    return blocks[-1] if blocks else None\n\n\ndef grid_text(g):\n    return "\\n".join("".join(str(c) for c in r) for r in g)\n\n\nPROMPT_EXTRA = ""\nPROMPT_SUMMARY = False\n\n\ndef summarize_text(train, test_inputs):\n    out = []\n    def _print(*a):\n        out.append(" ".join(str(x) for x in a))\n\n    import numpy as np\n    from collections import Counter\n\n    def comps(a, bg, diag=True):\n        h, w = a.shape\n        seen = np.zeros((h, w), bool)\n        out = []\n        for i in range(h):\n            for j in range(w):\n                if a[i, j] != bg and not seen[i, j]:\n                    c = a[i, j]\n                    st = [(i, j)]\n                    seen[i, j] = True\n                    cells = []\n                    while st:\n                        y, x = st.pop()\n                        cells.append((y, x))\n                        for dy in (-1, 0, 1):\n                            for dx in (-1, 0, 1):\n                                if (dy or dx) and (diag or not (dy and dx)):\n                                    yy, xx = y + dy, x + dx\n                                    if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and a[yy, xx] == c:\n                                        seen[yy, xx] = True\n                                        st.append((yy, xx))\n                    ys = [c_[0] for c_ in cells]\n                    xs = [c_[1] for c_ in cells]\n                    out.append((int(c), len(cells), (min(ys), min(xs), max(ys), max(xs))))\n        return out\n\n    def desc(a, name):\n        bg = Counter(a.ravel().tolist()).most_common(1)[0][0]\n        cc = Counter(a.ravel().tolist())\n        cs = comps(a, bg)\n        t = name + ": " + str(a.shape[0]) + "x" + str(a.shape[1]) + ", colors " + ", ".join(str(k) + ":" + str(v) for k, v in sorted(cc.items())) + ", background guess " + str(bg)\n        t += "; " + str(len(cs)) + " same-color objects (8-connected)"\n        if cs:\n            big = sorted(cs, key=lambda o: -o[1])[:6]\n            t += ", largest: " + "; ".join("color " + str(c) + " size " + str(n) + " rows " + str(b[0]) + "-" + str(b[2]) + " cols " + str(b[1]) + "-" + str(b[3]) for c, n, b in big)\n        sym = []\n        if (a == a[::-1]).all(): sym.append("up-down symmetric")\n        if (a == a[:, ::-1]).all(): sym.append("left-right symmetric")\n        if a.shape[0] == a.shape[1] and (a == a.T).all(): sym.append("transpose symmetric")\n        if sym: t += "; " + ", ".join(sym)\n        return t\n\n    pairs = train\n    for i, ex in enumerate(pairs):\n        a, b = np.array(ex["input"]), np.array(ex["output"])\n        _print("Example", i)\n        _print("  " + desc(a, "input"))\n        _print("  " + desc(b, "output"))\n        if a.shape == b.shape:\n            d = a != b\n            n = int(d.sum())\n            tr = Counter((int(x), int(y)) for x, y in zip(a[d], b[d]))\n            if n:\n                ys, xs = np.where(d)\n                _print("  same shape; " + str(n) + " cells change (rows " + str(ys.min()) + "-" + str(ys.max()) + ", cols " + str(xs.min()) + "-" + str(xs.max()) + "); color changes (from, to): " + ", ".join(str(k) + " x" + str(v) for k, v in tr.most_common(8)))\n            else:\n                _print("  same shape; no cell changes")\n        else:\n            note = []\n            if b.shape[0] % a.shape[0] == 0 and b.shape[1] % a.shape[1] == 0:\n                fy, fx = b.shape[0] // a.shape[0], b.shape[1] // a.shape[1]\n                note.append("output is " + str(fy) + "x" + str(fx) + " times the input size" + (" and equals the input tiled" if (np.tile(a, (fy, fx)) == b).all() else "") + (" and equals the input scaled up" if (np.kron(a, np.ones((fy, fx), int)) == b).all() else ""))\n            if b.shape[0] <= a.shape[0] and b.shape[1] <= a.shape[1]:\n                hits = [(y, x) for y in range(a.shape[0] - b.shape[0] + 1) for x in range(a.shape[1] - b.shape[1] + 1) if (a[y:y + b.shape[0], x:x + b.shape[1]] == b).all()]\n                if hits: note.append("output is a sub-grid of the input at row, col " + str(hits[0]))\n            _print("  different shape" + ("; " + "; ".join(note) if note else ""))\n    tests = test_inputs\n    for k, t_ in enumerate(tests):\n        _print("Test input" + (" %d" % k if len(tests) > 1 else ""))\n        _print("  " + desc(np.array(t_), "input"))\n    return "\\n".join(out)\n\n\ndef build_prompt(task, ti=0):\n    parts = [HEADER_CODE + PROMPT_EXTRA, "--Training Examples--\\n"]\n    for i, ex in enumerate(task["train"]):\n        parts.append(f"--Example {i}--\\nINPUT:\\n{grid_text(ex[\'input\'])}\\nOUTPUT:\\n{grid_text(ex[\'output\'])}\\n")\n    parts.append(f"--Test Input--\\n{grid_text(task[\'test\'][ti][\'input\'])}\\n")\n    if PROMPT_SUMMARY:\n        parts.append("--Facts about the grids, computed by code (colors with their cell counts; objects are connected groups of cells of one color, largest first, with their row and column ranges)--\\n"\n                     + summarize_text(task["train"], [task["test"][ti]["input"]]) + "\\n")\n    return "\\n".join(parts)\n\n\ndef tkey(g):\n    return tuple(map(tuple, g)) if g else None\n\n\ndef wrong_cells(out, exp):\n    if out is None:\n        return 10 ** 4\n    if (len(out), len(out[0])) != (len(exp), len(exp[0])):\n        return 10 ** 3\n    return sum(1 for r in range(len(exp)) for c in range(len(exp[0])) if out[r][c] != exp[r][c])\n\n\ndef closeness(task, res):\n    if res.get("error"):\n        return 10 ** 6\n    return sum(wrong_cells(d["out"], task["train"][i]["output"]) for i, d in enumerate(res["demo"]) if d["out"] is None or d["out"] != task["train"][i]["output"])\n\n\ndef detail_text(task, res):\n    if res.get("error"):\n        return f"The program does not run at all: {res[\'error\']}."\n    train = task["train"]\n    fails = [i for i, d in enumerate(res["demo"]) if d["out"] is None or d["out"] != train[i]["output"]]\n    i = min(fails, key=lambda i: len(train[i]["output"]) * len(train[i]["output"][0]))\n    d, exp = res["demo"][i], train[i]["output"]\n    if d["out"] is None:\n        return f"On training example {i} the function failed: {d[\'err\']}."\n    out = d["out"]\n    if (len(out), len(out[0])) != (len(exp), len(exp[0])):\n        return (f"On training example {i} the output has shape {len(out)}x{len(out[0])} but the expected shape is {len(exp)}x{len(exp[0])}.\\n"\n                f"Program output:\\n{grid_text(out)}\\nExpected output:\\n{grid_text(exp)}")\n    cells = [(r, c, out[r][c], exp[r][c]) for r in range(len(exp)) for c in range(len(exp[0])) if out[r][c] != exp[r][c]]\n    shown = ", ".join(f"({r},{c}) {o}->{e}" for r, c, o, e in cells[:12]) + (" ..." if len(cells) > 12 else "")\n    return (f"On training example {i} the output has the right shape ({len(out)}x{len(out[0])}) but {len(cells)} cells are wrong.\\n"\n            f"Program output:\\n{grid_text(out)}\\nExpected output:\\n{grid_text(exp)}\\nWrong cells (row, col) program->expected: {shown}")\n\n\ndef inject_a(code, k, n, detail):\n    return (f"\\n\\nLet me stop here and test my current best program on the training examples.\\n\\n{FENCE}python\\n" + code.strip() + f"\\n{FENCE}\\n\\n"\n            f"I ran it. It reproduces {k} of {n} training examples.\\n{detail}\\n\\n"\n            "So this program is not right yet. Let me work out exactly why it fails and what the real rule is, and then fix it.\\n")\n\n\ndef messages_b(task, code, k, n, detail):\n    user2 = (f"I ran your program on the training examples. It reproduces {k} of {n} of them.\\n{detail}\\n\\n"\n             "The program is not correct yet. Think carefully about what the rule really is, using this feedback, then give a corrected complete Python program "\n             f"(imports plus the function `transform(grid)`) inside one fenced {FENCE}python code block, and write nothing after the code block.")\n    return [{"role": "user", "content": build_prompt(task)}, {"role": "assistant", "content": f"{FENCE}python\\n" + code.strip() + f"\\n{FENCE}"}, {"role": "user", "content": user2}]\n\n\nclass Replica:\n    def __init__(self, port):\n        self.port = port\n        self.client = AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="x", timeout=86400, max_retries=0)\n        self.active = 0\n\n\nclass Solver:\n    def __init__(self, a):\n        self.a = a\n        self.t0 = a.t_start\n        self.deadline = self.t0 + a.total_s\n        self.main_deadline = self.deadline - a.reserve_s\n        self.tok = AutoTokenizer.from_pretrained(a.model_dir)\n        self.replicas = [Replica(int(p)) for p in a.ports.split(",")]\n        self.tasks = json.load(open(a.challenges))\n        if a.tasks:\n            keep = set(json.load(open(a.tasks)))\n            self.tasks = {k: v for k, v in self.tasks.items() if k in keep}\n        self.sol = json.load(open(a.solutions)) if a.solutions else None\n        self.submission = {tid: [{"attempt_1": t["input"], "attempt_2": t["input"]} for t in task["test"]] for tid, task in self.tasks.items()}\n        self.done_tasks = set()\n        if os.path.exists(a.log):\n            for line in open(a.log):\n                try:\n                    r = json.loads(line)\n                except ValueError:\n                    continue\n                if r.get("event") == "done":\n                    self.done_tasks.add(r["task"])\n            if self.done_tasks and os.path.exists(a.out):\n                try:\n                    prev = json.load(open(a.out))\n                    for tid in self.done_tasks:\n                        if tid in prev and tid in self.submission:\n                            self.submission[tid] = prev[tid]\n                except Exception:\n                    pass\n        self.pool = ThreadPoolExecutor(a.exec_workers)\n        self.log = open(a.log, "a")\n        self.sub_lock = asyncio.Lock()\n        self.inflight_caps = {}\n        self.inflight_start = {}\n        self.stat_tokens, self.stat_pred, self.stat_secs, self.stat_n = 0.0, 0.0, 0.0, 0\n        self.last_cap, self.n_started = None, 0\n        self.not_started = 0\n        self.gen_hist = deque(maxlen=400)\n        self.thr = a.thr_prior\n        self.cap_sum, self.cap_n = 0.0, 0\n\n    def now(self):\n        return time.time() - self.t0\n\n    def emit(self, rec):\n        rec["t"] = round(self.now(), 1)\n        self.log.write(json.dumps(rec) + "\\n")\n        self.log.flush()\n\n    async def write_submission(self):\n        async with self.sub_lock:\n            tmp = self.a.out + ".tmp"\n            with open(tmp, "w") as f:\n                json.dump(self.submission, f)\n            os.replace(tmp, self.a.out)\n\n    def head(self, task, ti=0):\n        return self.head_from_messages([{"role": "user", "content": build_prompt(task, ti)}])\n\n    def head_from_messages(self, msgs):\n        kw = {"reasoning_effort": self.a.effort} if self.a.effort else {}\n        h = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)\n        return h if h.rstrip().endswith("<think>") else h + "<think>\\n"\n\n    def gen_tokens_total(self):\n        tot = 0.0\n        for r in self.replicas:\n            try:\n                txt = urllib.request.urlopen(f"http://127.0.0.1:{r.port}/metrics", timeout=5).read().decode()\n            except Exception:\n                return None\n            for line in txt.splitlines():\n                if line.startswith("vllm:generation_tokens_total"):\n                    tot += float(line.rpartition(" ")[2])\n        return tot\n\n    async def monitor(self):\n        """Throughput of the whole server group over the last 20 minutes (tokens decoded per wall second, prefill stalls included)."""\n        while True:\n            await asyncio.sleep(30)\n            tot = await asyncio.get_running_loop().run_in_executor(None, self.gen_tokens_total)\n            if tot is None:\n                continue\n            now = time.time()\n            self.gen_hist.append((now, tot, sum(r.active for r in self.replicas)))\n            while self.gen_hist and now - self.gen_hist[0][0] > 1200:\n                self.gen_hist.popleft()\n            full = [h for h in self.gen_hist if h[2] >= 0.8 * self.target_streams]\n            if len(self.gen_hist) >= 20 and len(full) >= 0.7 * len(self.gen_hist):\n                dt = self.gen_hist[-1][0] - self.gen_hist[0][0]\n                self.thr = max(50.0, (self.gen_hist[-1][1] - self.gen_hist[0][1]) / dt)\n\n    def sec_per_cost(self):\n        """Stream-seconds per predicted decoded token (the cost table at the cap the task was given), from the tasks finished so far: decode, finalization and every wait are inside the\n        seconds, and a table that is too high or too low is absorbed. The prior counts as `prior-tasks` tasks of 40K tokens."""\n        base = 40e3 * self.a.prior_tasks\n        prior_r = max(self.target_streams, 1) / self.a.thr_prior\n        return (self.stat_secs + base * prior_r) / (self.stat_pred + base)\n\n    def thr_est(self):\n        """Decoded tokens per second of the whole group, from the tasks finished so far (for the log)."""\n        base = 40e3 * self.a.prior_tasks\n        n = max(self.target_streams, 1)\n        return (self.stat_tokens + base) / (self.stat_secs / n + base / self.a.thr_prior)\n\n    def note_done(self, rec):\n        toks = rec.get("ntok", 0) + rec.get("forced_tokens", 0)\n        secs = rec.get("seconds", 0.0)\n        if toks > 0 and secs > 0 and rec.get("cap"):\n            self.stat_tokens += toks\n            self.stat_pred += cost_of_cap(rec["cap"])\n            self.stat_secs += secs\n            self.stat_n += 1\n\n    def choose_cap(self):\n        """Largest cap whose expected stream-seconds, added to what the tasks in flight still need, fit in the time left. The seconds per token come from finished tasks, so the waves of\n        decoding and finalization that come from tasks starting together average out; the cap moves by at most --cap-step between tasks and the first wave is spread."""\n        a = self.a\n        now = time.time()\n        left = self.main_deadline - now\n        r = self.sec_per_cost()\n        fb_extra = a.fb_n1 * 1.2e3 if a.fb != "off" else 0.0\n        avail = max(self.target_streams, 1) * left * a.safety\n        inflight = 0.0\n        for tid, cap_i in self.inflight_caps.items():\n            d_i = r * cost_of_cap(cap_i)\n            inflight += max(0.15 * d_i, d_i - (now - self.inflight_start.get(tid, now)))\n        c = a.max_cap\n        while c > a.min_cap:\n            if inflight + (self.not_started + 1) * r * (cost_of_cap(c) + fb_extra) <= avail:\n                break\n            c -= 512\n        if self.last_cap is not None:\n            c = max(self.last_cap - a.cap_step_down, min(self.last_cap + a.cap_step, c))\n        self.last_cap = c\n        if self.n_started < self.target_streams and a.first_wave_lo < 1.0:\n            c = int(c * (a.first_wave_lo + (1.0 - a.first_wave_lo) * self.n_started / max(self.target_streams - 1, 1)))\n        self.n_started += 1\n        # a trace has to fit in the time that is left at a pessimistic per-stream speed\n        return int(max(a.min_cap, min(c, left * a.stream_rate)))\n\n    async def stream_text(self, rep, prompt, max_tokens, seed=None):\n        extra = {"top_k": 20}\n        if seed is not None:\n            extra["seed"] = seed\n        text, finish, ntok = [], None, 0\n        stream = await rep.client.completions.create(model="qwen", prompt=prompt, max_tokens=max_tokens, temperature=1.0, top_p=0.95, stream=True,\n                                                     stream_options={"include_usage": True}, extra_body=extra)\n        try:\n            async for ch in stream:\n                if ch.choices:\n                    text.append(ch.choices[0].text or "")\n                    if ch.choices[0].finish_reason:\n                        finish = ch.choices[0].finish_reason\n                if getattr(ch, "usage", None):\n                    ntok = ch.usage.completion_tokens\n                if time.time() > self.main_deadline:\n                    finish = "deadline"\n                    break\n        finally:\n            await stream.close()\n        return "".join(text), finish, ntok\n\n    async def forced(self, rep, prompt_head, reasoning, n=None):\n        a = self.a\n        p = prompt_head + reasoning.rstrip() + CLOSE\n        for attempt in range(4):\n            try:\n                r = await rep.client.completions.create(model="qwen", prompt=p, max_tokens=a.forced_max_tokens, temperature=0.7, top_p=0.95, n=n or a.n_forced)\n                return [FENCE + "python\\n" + c.text for c in r.choices], r.usage.completion_tokens\n            except Exception as e:\n                self.emit({"event": "forced_error", "err": repr(e)[:200], "attempt": attempt})\n                await asyncio.sleep(30 * (attempt + 1))\n                if time.time() > self.deadline - 300:\n                    break\n        return [], 0\n\n    def run_program(self, code, task):\n        req = {"code": code, "train": task["train"], "test_inputs": [t["input"] for t in task["test"]]}\n        try:\n            p = subprocess.run([sys.executable, "-c", EXEC_DEMO], input=json.dumps(req), capture_output=True, text=True, timeout=90)\n            return json.loads(p.stdout.strip().splitlines()[-1])\n        except Exception as e:\n            return {"n_train": len(task["train"]), "n_pass": 0, "demo": [], "preds": [None] * len(task["test"]), "error": "exec: " + repr(e)[:120]}\n\n    async def execute(self, codes, task):\n        loop = asyncio.get_running_loop()\n        return await asyncio.gather(*(loop.run_in_executor(self.pool, self.run_program, c, task) for c in codes if c))\n\n    @staticmethod\n    def verified(r):\n        return r["n_pass"] == r["n_train"] and r.get("preds") and all(p is not None for p in r["preds"])\n\n    def vote(self, results, task):\n        out = []\n        for j in range(len(task["test"])):\n            w = defaultdict(float)\n            for r in results:\n                ps = r.get("preds") or []\n                p = ps[j] if j < len(ps) else None\n                if p is None:\n                    continue\n                w[tkey(p)] += 2.0 if r["n_pass"] == r["n_train"] else 0.25 * r["n_pass"] / max(r["n_train"], 1)\n            ranked = [g for g, _ in sorted(w.items(), key=lambda kv: -kv[1])]\n            out.append([[list(r) for r in g] for g in ranked[:2]])\n        return out\n\n    async def finish_task(self, tid, task, results, rec):\n        picks = self.vote(results, task)\n        for j, ranked in enumerate(picks):\n            if ranked:\n                self.submission[tid][j] = {"attempt_1": ranked[0], "attempt_2": ranked[1] if len(ranked) > 1 else ranked[0]}\n        await self.write_submission()\n        rec["n_verified"] = sum(1 for r in results if self.verified(r))\n        self.note_done(rec)\n        if self.sol is not None:\n            gold = self.sol[tid]\n            rec["correct_top1"] = [self.submission[tid][j]["attempt_1"] == gold[j] for j in range(len(task["test"]))]\n            rec["correct_top2"] = [self.submission[tid][j]["attempt_1"] == gold[j] or self.submission[tid][j]["attempt_2"] == gold[j] for j in range(len(task["test"]))]\n        self.emit(rec)\n\n    async def solve(self, tid, rep):\n        a = self.a\n        task = self.tasks[tid]\n        t_begin = self.now()\n        if time.time() > self.main_deadline - a.min_trace_s:\n            self.emit({"event": "skipped", "task": tid})\n            return\n        self.not_started -= 1\n        prompt = self.head(task)\n        plen = len(self.tok(prompt, add_special_tokens=False)["input_ids"])\n        # prompt, reasoning and the forced program have to fit in one server context\n        cap = max(2048, min(self.choose_cap(), a.max_len - plen - a.forced_max_tokens - 64))\n        self.inflight_caps[tid] = cap\n        self.inflight_start[tid] = time.time()\n        rep.active += 1\n        self.cap_sum += cap\n        self.cap_n += 1\n        self.emit({"event": "start", "task": tid, "cap": cap, "plen": plen, "replica": rep.port, "thr": round(self.thr_est(), 1), "left_s": round(self.main_deadline - time.time()), "not_started": self.not_started})\n        try:\n            fb = a.fb != "off" and cap > a.fb_ckpt + 4096\n            seg1 = a.fb_ckpt if fb else cap\n            text, finish, ntok = "", None, 0\n            for attempt in range(4):\n                try:\n                    text, finish, ntok = await self.stream_text(rep, prompt, seg1)\n                    break\n                except Exception as e:\n                    self.emit({"event": "trace_error", "task": tid, "err": repr(e)[:200], "attempt": attempt})\n                    await asyncio.sleep(45 * (attempt + 1))\n                    if time.time() > self.main_deadline - a.min_trace_s:\n                        return\n            else:\n                return\n            rec = {"event": "done", "task": tid, "cap": cap, "replica": rep.port, "fb": a.fb if fb else "off"}\n            own = extract_code(text.split("</think>")[-1]) if "</think>" in text and finish == "stop" else None\n            reasoning_tokens, forced_tokens = ntok, 0\n            if own:\n                results = await self.execute([own], task)\n                rec.update({"finish": "stop", "ntok": reasoning_tokens, "forced_tokens": 0, "n_programs": 1, "seconds": round(self.now() - t_begin, 1)})\n                await self.finish_task(tid, task, results, rec)\n                return\n            reasoning = text.split("</think>")[0] if "</think>" in text else text\n            if fb and finish == "length":\n                # checkpoint: a few programs from the prefix; a program that passes every demo ends the task, otherwise its failure is fed back\n                comps, ft = await self.forced(rep, prompt, reasoning, n=a.fb_n1)\n                forced_tokens += ft\n                codes = [c for c in (extract_code(x) for x in comps) if c]\n                res1 = await self.execute(codes, task)\n                if any(self.verified(r) for r in res1):\n                    rec.update({"finish": "verified_at_checkpoint", "ntok": reasoning_tokens, "forced_tokens": forced_tokens, "n_programs": len(codes), "seconds": round(self.now() - t_begin, 1)})\n                    await self.finish_task(tid, task, res1, rec)\n                    return\n                if codes:\n                    best = min(range(len(codes)), key=lambda i: (-res1[i]["n_pass"], closeness(task, res1[i]), i))\n                    k, n = res1[best]["n_pass"], res1[best]["n_train"]\n                    det = detail_text(task, res1[best])\n                    if a.fb == "A":\n                        prompt2 = prompt + reasoning + inject_a(codes[best], k, n, det)\n                    else:\n                        prompt2 = self.head_from_messages(messages_b(task, codes[best], k, n, det))\n                    seg2 = max(1024, cap - seg1)\n                    t2, finish, n2 = await self.stream_text(rep, prompt2, seg2)\n                    reasoning_tokens += n2\n                    own2 = extract_code(t2.split("</think>")[-1]) if "</think>" in t2 and finish == "stop" else None\n                    if own2:\n                        results = await self.execute([own2], task)\n                        rec.update({"finish": "stop", "ntok": reasoning_tokens, "forced_tokens": forced_tokens, "n_programs": 1, "seconds": round(self.now() - t_begin, 1)})\n                        await self.finish_task(tid, task, results, rec)\n                        return\n                    prompt, reasoning = prompt2, (t2.split("</think>")[0] if "</think>" in t2 else t2)\n                    text = t2\n            comps, ft = await self.forced(rep, prompt, reasoning)\n            forced_tokens += ft\n            codes = [c for c in (extract_code(x) for x in comps) if c]\n            results = await self.execute(codes, task)\n            rec.update({"finish": finish, "ntok": reasoning_tokens, "forced_tokens": forced_tokens, "n_programs": len(codes), "seconds": round(self.now() - t_begin, 1)})\n            await self.finish_task(tid, task, results, rec)\n        finally:\n            rep.active -= 1\n            self.inflight_caps.pop(tid, None)\n            self.inflight_start.pop(tid, None)\n\n    async def run(self):\n        a = self.a\n        await self.write_submission()\n        order = [t for t in self.tasks if t not in self.done_tasks]\n        random.Random(a.seed).shuffle(order)\n        self.not_started = len(order)\n        self.target_streams = a.per_replica * len(self.replicas)\n        queue = asyncio.Queue()\n        for t in order:\n            queue.put_nowait(t)\n        self.emit({"event": "run_start", "n_tasks": len(order), "resumed": len(self.done_tasks), "max_cap": a.max_cap, "streams": self.target_streams, "total_s": a.total_s, "fb": a.fb})\n        mon = asyncio.create_task(self.monitor())\n\n        async def worker(rep, delay):\n            # streams open one after another so that the tasks do not reach their caps, and need the cache for their forced programs, at the same time\n            if queue.empty():\n                return\n            await asyncio.sleep(delay)\n            while True:\n                try:\n                    tid = queue.get_nowait()\n                except asyncio.QueueEmpty:\n                    return\n                await self.solve(tid, rep)\n\n        slots = [(r, k * len(self.replicas) + i) for k in range(a.per_replica) for i, r in enumerate(self.replicas)]\n        # after a restart the log holds finished tasks: the ramp is cut to 20 minutes so a restart does not idle the streams for an hour\n        stagger = a.stagger_s if not self.done_tasks else min(a.stagger_s, 1200.0 / max(len(slots), 1))\n        await asyncio.gather(*(worker(r, j * stagger) for r, j in slots))\n        mon.cancel()\n        await self.write_submission()\n        self.emit({"event": "end", "mean_cap": round(self.cap_sum / max(self.cap_n, 1))})\n\n\ndef make_parser():\n    ap = argparse.ArgumentParser()\n    ap.add_argument("--challenges", required=True)\n    ap.add_argument("--solutions", default="")\n    ap.add_argument("--tasks", default="")\n    ap.add_argument("--model-dir", required=True)\n    ap.add_argument("--out", default="submission.json")\n    ap.add_argument("--log", default="solver_log.jsonl")\n    ap.add_argument("--ports", default="8000")\n    ap.add_argument("--per-replica", type=int, default=24)\n    ap.add_argument("--max-cap", type=int, default=49152)\n    ap.add_argument("--min-cap", type=int, default=8192)\n    ap.add_argument("--thr-prior", type=float, default=330.0, help="tokens per second for the whole server group until it has been measured")\n    ap.add_argument("--safety", type=float, default=0.92)\n    ap.add_argument("--stream-rate", type=float, default=9.0, help="pessimistic tokens per second of one stream; a trace never gets more than this x the time left")\n    ap.add_argument("--n-forced", type=int, default=8)\n    ap.add_argument("--forced-max-tokens", type=int, default=4000)\n    ap.add_argument("--max-len", type=int, default=65536, help="context length of the servers")\n    ap.add_argument("--fb", default="off", choices=["off", "A", "B"])\n    ap.add_argument("--fb-ckpt", type=int, default=16384)\n    ap.add_argument("--fb-n1", type=int, default=4)\n    ap.add_argument("--exec-workers", type=int, default=16)\n    ap.add_argument("--t-start", type=float, default=time.time())\n    ap.add_argument("--total-s", type=float, default=11.5 * 3600)\n    ap.add_argument("--reserve-s", type=float, default=1500)\n    ap.add_argument("--min-trace-s", type=float, default=600)\n    ap.add_argument("--seed", type=int, default=0)\n    ap.add_argument("--effort", default="", help="reasoning effort of the chat template (xhigh when empty; medium and low exist)")\n    ap.add_argument("--cap-step", type=int, default=1536, help="the cap of a task is at most this many tokens above the one before")\n    ap.add_argument("--cap-step-down", type=int, default=4096, help="the cap of a task is at most this many tokens below the one before")\n    ap.add_argument("--stagger-s", type=float, default=150.0, help="stream j starts its first task j x this many seconds after the first (0 = all at once)")\n    ap.add_argument("--first-wave-lo", type=float, default=1.0, help="the caps of the first wave of tasks are spread over [this x the cap, the cap] so the streams do not finish together")\n    ap.add_argument("--prior-tasks", type=float, default=6.0, help="weight of the throughput prior, in tasks of 40K decoded tokens")\n    ap.add_argument("--prompt-summary", action="store_true", help="add code-computed facts about the grids (colors, objects, changes) to the prompt")\n    ap.add_argument("--prompt-extra", default="", help="text added to the instructions of the prompt (after the header, before the examples)")\n    ap.add_argument("--cost-points", default="", help="JSON list of [cap, mean decoded tokens per task] pairs measured for this effort; replaces the built-in table")\n    return ap\n\n\ndef main():\n    a = make_parser().parse_args()\n    if a.cost_points:\n        COST_POINTS[:] = [tuple(p) for p in json.loads(a.cost_points)]\n    global PROMPT_EXTRA, PROMPT_SUMMARY\n    if a.prompt_extra:\n        PROMPT_EXTRA = a.prompt_extra + "\\n\\n"\n    PROMPT_SUMMARY = a.prompt_summary\n    asyncio.run(Solver(a).run())\n\n\nif __name__ == "__main__":\n    main()\n'


# ---- final driver: serve Qwen3.8-27B on Kaggle L4x4 and run solver2 over the hidden test tasks (or a smoke / replica run when not a competition rerun) ----
import hashlib

rerun = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN")) or bool(CFG.get("force_rerun"))
install_wheelhouse()
open(f"{WORK}/solver2.py", "w").write(SOLVER_SRC)

MODELS = {"fp8": find_dir("qwen3-8-27b-fp8-hf-snapshot"), "int4": find_dir("qwen3-8-27b-awq")}
LAY = CFG["layouts"][CFG["layout"]]
MODEL = MODELS[LAY["model"]]
COMP = find_dir("arc-prize-2026-arc-agi-2")
log("rerun", rerun, "layout", CFG["layout"], "model", MODEL, "competition dir", COMP)

override = CFG.get("rerun_total_s_override", 0)  # lets an interactive run rehearse the rerun branch (hidden test file, no solutions) on a short clock
if rerun:
    chal = f"{COMP}/arc-agi_test_challenges.json"
    sol, tasks_file = "", ""
    total_s = override or CFG["total_s_rerun"]
    hard_end_s = (override + 600) if override else CFG["hard_end_rerun"]
else:
    chal = f"{COMP}/arc-agi_evaluation_challenges.json"
    sol = f"{COMP}/arc-agi_evaluation_solutions.json"
    ids = sorted(json.load(open(chal)))
    random.Random(CFG["replica_seed"]).shuffle(ids)
    mode = CFG["commit_mode"]
    n_tasks = CFG["smoke_tasks"] if mode == "smoke" else CFG["replica_tasks"]
    tasks_file = f"{WORK}/tasks_subset.json"
    json.dump(ids[:n_tasks], open(tasks_file, "w"))
    # the replica gives each task the time one hidden-set task gets, (41,400 s - 660 s startup - 1,500 s reserve) / 240 = 163.5 s, plus the same fixed overhead
    total_s = CFG["smoke_total_s"] if mode == "smoke" else CFG["replica_overhead_s"] + n_tasks * CFG["replica_s_per_task"]
    hard_end_s = total_s
log("tasks file", tasks_file, "total_s", total_s, "chal", chal)

solver_args = ["--challenges", chal, "--model-dir", MODEL, "--out", f"{WORK}/submission.json", "--log", f"{WORK}/solver_log.jsonl", "--ports", ",".join(str(8000 + i) for i in range(len(LAY["groups"]))),
               "--per-replica", str(LAY["per_replica"]), "--max-cap", str(CFG["max_cap"] if rerun or CFG["commit_mode"] != "smoke" else CFG["smoke_cap"]), "--min-cap", str(CFG["min_cap"]),
               "--thr-prior", str(LAY["thr_prior"]), "--total-s", str(total_s), "--reserve-s", str(CFG["reserve_s"] if rerun or CFG["commit_mode"] != "smoke" else 240),
               "--n-forced", str(CFG["n_forced"]), "--fb", CFG["fb"], "--fb-ckpt", str(CFG["fb_ckpt"]), "--fb-n1", str(CFG["fb_n1"]), "--exec-workers", "16", "--safety", str(CFG["safety"]),
               "--min-trace-s", str(CFG["min_trace_s"]), "--effort", CFG["effort"], "--max-len", str(LAY["max_len"])]
if CFG.get("cost_points"):
    solver_args += ["--cost-points", json.dumps(CFG["cost_points"])]
if CFG.get("prompt_summary"):
    solver_args += ["--prompt-summary"]
if CFG.get("prompt_extra"):
    solver_args += ["--prompt-extra", CFG["prompt_extra"]]
if sol:
    solver_args += ["--solutions", sol]
if tasks_file:
    solver_args += ["--tasks", tasks_file]

HARD_END = T0 + hard_end_s
spec = ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": CFG["mtp_tokens"]})] if CFG["mtp_tokens"] else []
pc = ["--enable-prefix-caching", "--mamba-cache-mode", "align"] if CFG["prefix_caching"] else []
env_solver = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")


def solver_done():
    try:
        return any(json.loads(l).get("event") == "end" for l in open(f"{WORK}/solver_log.jsonl"))
    except Exception:
        return False


def progress():
    try:
        recs = [json.loads(l) for l in open(f"{WORK}/solver_log.jsonl")]
    except Exception:
        return
    done = [r for r in recs if r.get("event") == "done"]
    caps = [r["cap"] for r in done]
    msg = f"progress: {len(done)} tasks done, mean cap {sum(caps) / max(len(caps), 1):.0f}"
    if done and "correct_top2" in done[0]:
        t1 = sum(sum(r["correct_top1"]) / len(r["correct_top1"]) for r in done) / len(done)
        t2 = sum(sum(r["correct_top2"]) / len(r["correct_top2"]) for r in done) / len(done)
        msg += f", task-level top1 {100 * t1:.1f} top2 {100 * t2:.1f}"
    starts = [r for r in recs if r.get("event") == "start"]
    if starts:
        msg += f", last controller state: thr {starts[-1]['thr']} tok/s, left {starts[-1]['left_s']} s, not started {starts[-1]['not_started']}"
    log(msg)
    try:
        import glob as _g
        lines = []
        for f in sorted(_g.glob(f"{WORK}/vllm_run*_0.log")):
            lines += [l for l in open(f, errors="replace") if "Avg prompt throughput" in l]
        if lines:
            log("vllm:", lines[-1].split("loggers.py")[-1].strip()[:260])
        import urllib.request as _u
        txt = _u.urlopen("http://127.0.0.1:8000/metrics", timeout=5).read().decode()
        want = ("vllm:num_preemptions_total", "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total", "vllm:generation_tokens_total", "vllm:prompt_tokens_total")
        vals = {k.split(":")[1]: float(l.rpartition(" ")[2]) for l in txt.splitlines() for k in want if l.startswith(k)}
        log("vllm counters:", {k: round(v) for k, v in vals.items()})
    except Exception as e:
        log("vllm telemetry failed:", repr(e)[:100])


def servers_alive():
    return bool(SERVERS) and all(p.poll() is None for p in SERVERS)


attempts, ports = 0, []
while time.time() < HARD_END - 1200 and not solver_done() and attempts < CFG["max_restarts"]:
    attempts += 1
    if not servers_alive():
        ports = start_servers(f"run{attempts}", MODEL, LAY["groups"], LAY["tp"], extra=[*spec, *pc, *LAY.get("extra", [])], max_len=LAY["max_len"], max_seqs=LAY["max_seqs"],
                              max_batched=LAY.get("max_batched", 4096))
        if not wait_ready(f"run{attempts}", ports, timeout=1500):
            log("servers did not come up; retrying")
            stop_servers()
            continue
    log("starting solver, attempt", attempts)
    sp = subprocess.Popen([sys.executable, f"{WORK}/solver2.py", *solver_args, "--t-start", str(T0)], stdout=open(f"{WORK}/solver.out", "a"), stderr=subprocess.STDOUT, env=env_solver)
    last_progress = time.time()
    while True:
        time.sleep(20)
        if sp.poll() is not None:
            log("solver exited with", sp.returncode)
            break
        if any(p.poll() is not None for p in SERVERS):
            log("a vLLM server died; restarting everything")
            sp.kill()
            break
        if time.time() > HARD_END:
            log("hard end reached")
            sp.kill()
            break
        if time.time() - last_progress > 600:
            last_progress = time.time()
            progress()
    if solver_done() or time.time() > HARD_END - 1200:
        break
    if not servers_alive():
        stop_servers()
stop_servers()
progress()

# the solver keeps submission.json valid at every step; make sure a file exists even if it never started
sub_path = f"{WORK}/submission.json"
if not os.path.exists(sub_path):
    tasks = json.load(open(chal))
    json.dump({tid: [{"attempt_1": t["input"], "attempt_2": t["input"]} for t in task["test"]] for tid, task in tasks.items()}, open(sub_path, "w"))
sub = json.load(open(sub_path))
log("submission.json holds", len(sub), "tasks;", sum(len(v) for v in sub.values()), "test outputs")
log("ALL DONE")
