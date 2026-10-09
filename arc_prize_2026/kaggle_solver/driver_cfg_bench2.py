
# ---- driver: compare Qwen3.8-27B serving layouts on Kaggle L4x4 (decode throughput on real ARC reasoning prefixes, plus a quantization-shift probe) ----
from openai import AsyncOpenAI, OpenAI
from transformers import AutoTokenizer

install_wheelhouse()
FP8 = find_dir("qwen3-8-27b-fp8-hf-snapshot")
AWQ = find_dir("qwen3-8-27b-awq")
BENCH = find_dir("arc-bench-traces")
EVAL = glob.glob("/kaggle/input/**/arc-agi_evaluation_challenges.json", recursive=True)[0]
CHAL = json.load(open(EVAL))
META = json.load(open(BENCH + "/meta.json"))
TOK = AutoTokenizer.from_pretrained(FP8)
log("FP8", FP8, "AWQ", AWQ, "BENCH", BENCH)

HEADER_CODE = (
    "You are participating in a puzzle solving competition. You are an expert programmer and puzzle solver.\n\n"
    "Below is a list of input and output grid pairs that share one transformation rule. Your goal is to find the rule "
    "and write a Python function \x60transform(grid)\x60 that maps any input grid to its output grid. "
    "\x60grid\x60 is a list of lists of ints (0-9); return a list of lists of ints. You may import numpy. "
    "The function must reproduce every training output exactly and must generalize to the test input.\n\n"
    "Grids are shown as rows of digits 0-9, one row per line, no separators. Each digit is a color.\n"
    "After your reasoning, give only the complete Python code (imports plus the function) inside one fenced \x60\x60\x60python code block, "
    "and write nothing after the code block.\n\n")


def gt(g):
    return "\n".join("".join(str(c) for c in r) for r in g)


def build_prompt(task, ti):
    parts = [HEADER_CODE, "--Training Examples--\n"]
    for i, ex in enumerate(task["train"]):
        parts.append(f"--Example {i}--\nINPUT:\n{gt(ex['input'])}\nOUTPUT:\n{gt(ex['output'])}\n")
    parts.append(f"--Test Input--\n{gt(task['test'][ti]['input'])}\n")
    return "\n".join(parts)


def head_for(key):
    tid, ti = key.rsplit("_", 1)
    h = TOK.apply_chat_template([{"role": "user", "content": build_prompt(CHAL[tid], int(ti))}], tokenize=False, add_generation_prompt=True)
    return h if h.rstrip().endswith("<think>") else h + "<think>\n"


def read_trace(key):
    p = glob.glob(f"{BENCH}/{key}__0.txt*")[0]
    if os.path.isdir(p):
        p = sorted(glob.glob(p + "/*"))[0]
    data = open(p, "rb").read()
    try:
        return gzip.decompress(data).decode()
    except Exception:
        return data.decode()


TRACE_IDS = {}


def trace_ids(key):
    if key not in TRACE_IDS:
        TRACE_IDS[key] = TOK(read_trace(key).strip(), add_special_tokens=False)["input_ids"]
    return TRACE_IDS[key]


def prefix_prompt(key, n_tokens):
    return head_for(key) + TOK.decode(trace_ids(key)[:n_tokens])


async def bench(name, ports, n_streams, lo=2000, hi=40000, duration=700, max_tokens=16000, seed=3):
    rnd = random.Random(seed)
    keys = [m["key"] for m in META]
    rnd.shuffle(keys)
    lens = [int(lo + (hi - lo) * i / max(n_streams - 1, 1)) for i in range(n_streams)]
    prompts = [(keys[i % len(keys)], L, prefix_prompt(keys[i % len(keys)], L)) for i, L in enumerate(lens)]
    clients = [AsyncOpenAI(base_url=f"http://127.0.0.1:{p}/v1", api_key="x", timeout=7200, max_retries=0) for p in ports]
    out = open(f"{WORK}/cfg_bench.jsonl", "a")

    async def one(i, p):
        try:
            await clients[i % len(ports)].completions.create(model="qwen", prompt=p, max_tokens=max_tokens, temperature=1.0, top_p=0.95, extra_body={"top_k": 20})
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log("request error", repr(e)[:160])

    tasks = [asyncio.create_task(one(i, x[2])) for i, x in enumerate(prompts)]
    t0 = time.time()
    prev, tprev = scrape(ports), t0
    while time.time() - t0 < duration and not all(t.done() for t in tasks):
        await asyncio.sleep(30)
        cur, tc = scrape(ports), time.time()
        dt = tc - tprev
        u, w = gpu_util()
        rec = {"name": name, "n_streams": n_streams, "t": round(tc - t0), "run": pick(cur, "num_requests_running"), "wait": pick(cur, "num_requests_waiting"),
               "kv": round(pick(cur, "kv_cache_usage") / len(ports), 3), "gen_tps": round((pick(cur, "generation_tokens_total") - pick(prev, "generation_tokens_total")) / dt, 1),
               "prompt_tps": round((pick(cur, "prompt_tokens_total") - pick(prev, "prompt_tokens_total")) / dt, 1), "preempt": pick(cur, "num_preemptions"),
               "accepted": pick(cur, "num_accepted_tokens_total"), "drafts": pick(cur, "num_drafts_total"), "gpu_util": u, "watts": w}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(json.dumps(rec), flush=True)
        prev, tprev = cur, tc
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    for c in clients:
        await c.close()


def micro(port):
    c = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="x")
    p = prefix_prompt(META[0]["key"], 6000)
    t0 = time.time()
    first = usage = None
    for ch in c.completions.create(model="qwen", prompt=p, max_tokens=300, temperature=0.7, stream=True, stream_options={"include_usage": True}):
        if first is None:
            first = time.time()
        if getattr(ch, "usage", None):
            usage = ch.usage
    t1 = time.time()
    log(f"single stream (port {port}): ttft {first - t0:.1f}s on a 6K prefix; {usage.completion_tokens} tokens in {t1 - first:.1f}s = {usage.completion_tokens / (t1 - first):.1f} tok/s")
    t0 = time.time()
    c.completions.create(model="qwen", prompt=prefix_prompt(META[1]["key"], 30000), max_tokens=1, temperature=0)
    dt = time.time() - t0
    log(f"prefill (port {port}): about 30K tokens in {dt:.1f}s = {30000 / dt:.0f} tok/s")


def nll_probe(port, name, keys, n_ctx=2500, n_eval=1500):
    """Teacher-forced mean negative log-likelihood (nats per token, last n_eval tokens of an n_ctx-token excerpt of saved FP8-generated reasoning, after the task prompt)
    under the served model. The same keys are scored for every model, so the paired difference estimates the shift that quantization adds."""
    c = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="x", timeout=900)
    per = {}
    for key in keys:
        try:
            ids = trace_ids(key)[:n_ctx]
            text = head_for(key) + TOK.decode(ids)
            pids = TOK(text, add_special_tokens=False)["input_ids"]
            r = c.completions.create(model="qwen", prompt=text, max_tokens=1, temperature=0, extra_body={"prompt_logprobs": 1})
            ch = r.choices[0]
            pl = getattr(ch, "prompt_logprobs", None) or (ch.model_extra or {}).get("prompt_logprobs")
            if not pl or len(pl) != len(pids):
                log("nll probe: unexpected prompt_logprobs length", None if not pl else len(pl), len(pids))
                continue
            per[key] = sum(-pl[pos][str(pids[pos])]["logprob"] for pos in range(len(pids) - n_eval, len(pids))) / n_eval
        except Exception as e:
            log("nll probe excerpt failed", key, repr(e)[:200])
    rec = {"name": name, "probe": "nll", "n_eval": n_eval, "per_key": {k: round(v, 4) for k, v in per.items()}, "mean_nll": round(sum(per.values()) / max(len(per), 1), 5)}
    open(f"{WORK}/cfg_bench.jsonl", "a").write(json.dumps(rec) + "\n")
    log("NLL probe", name, rec["mean_nll"], "over", len(per), "excerpts")


MTP = lambda n: ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": n})]
PC = ["--enable-prefix-caching", "--mamba-cache-mode", "align"]
CONFIGS = [
    # name, model dir, GPU groups, tp, extra args, max_len, max_seqs, streams, bench seconds
    ("int4_tp2x2_mtp2_pc", AWQ, [[0, 1], [2, 3]], 2, MTP(2) + PC, 49152, 16, 20, 720),
    ("int4_tp2x2_mtp3_pc", AWQ, [[0, 1], [2, 3]], 2, MTP(3) + PC, 49152, 16, 20, 480),
]
for name, model, groups, tp, extra, max_len, max_seqs, streams, secs in CONFIGS:
    log("=" * 20, name)
    try:
        ports = start_servers(name, model, groups, tp, extra=extra, max_len=max_len, max_seqs=max_seqs)
        if not wait_ready(name, ports):
            stop_servers()
            continue
        try:
            micro(ports[0])
        except Exception as e:
            log("micro failed", repr(e)[:300])
        asyncio.run(bench(name, ports, streams, duration=secs))
    except Exception as e:
        log("config failed", name, repr(e)[:400])
    stop_servers()

# quantization-shift probe: teacher-forced NLL of the same saved reasoning under INT4 (GPUs 0,1) and FP8 (GPUs 2,3), no speculative decoding, 1K-token prefill chunks so
# the prompt-logprob buffers fit next to the weights (the first attempt died with the server on 4K chunks)
try:
    keys = [m["key"] for m in META if len(TOK(head_for(m["key"]), add_special_tokens=False)["input_ids"]) <= 5000][:24]
    log("nll keys", len(keys))
    ports = start_servers("nll", [AWQ, FP8], [[0, 1], [2, 3]], 2, extra=[], max_len=10240, util=0.80, max_seqs=4, max_batched=1024)
    if wait_ready("nll", ports, timeout=1500):
        for port, nm in zip(ports, ["int4", "fp8"]):
            nll_probe(port, nm, keys)
    stop_servers()
except Exception as e:
    log("nll setup failed", repr(e)[:300])
try:
    import numpy as np
    recs = {r["name"]: r for r in map(json.loads, open(f"{WORK}/cfg_bench.jsonl")) if r.get("probe") == "nll"}
    common = sorted(set(recs["int4"]["per_key"]) & set(recs["fp8"]["per_key"]))
    a, b = np.array([recs["int4"]["per_key"][k] for k in common]), np.array([recs["fp8"]["per_key"][k] for k in common])
    d = a - b
    log("NLL int4 minus fp8, paired over", len(d), "excerpts: mean", round(float(d.mean()), 5), "sd", round(float(d.std(ddof=1)), 5), "sem", round(float(d.std(ddof=1) / len(d) ** 0.5), 5),
        "mean nll fp8", round(float(b.mean()), 4), "int4", round(float(a.mean()), 4))
except Exception as e:
    log("paired nll failed", repr(e)[:200])
log("ALL DONE")
