
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


def long_ids(key, offset, need):
    """Token ids of this trace followed by the next traces, so any length up to `need` exists; `offset` makes streams that share a trace differ."""
    keys = [m["key"] for m in META]
    i = keys.index(key)
    ids = []
    while len(ids) < offset + need:
        ids += trace_ids(keys[i % len(keys)])
        i += 1
    return ids[offset:offset + need]


def prefix_prompt(key, n_tokens, offset=0):
    return head_for(key) + TOK.decode(long_ids(key, offset, n_tokens))


async def bench(name, ports, n_streams, lo=2000, hi=40000, duration=700, max_tokens=16000, seed=3):
    rnd = random.Random(seed)
    keys = [m["key"] for m in META]
    rnd.shuffle(keys)
    lens = [int(lo + (hi - lo) * i / max(n_streams - 1, 1)) for i in range(n_streams)]
    prompts = [(keys[i % len(keys)], L, prefix_prompt(keys[i % len(keys)], L, offset=1500 * (i // len(keys)))) for i, L in enumerate(lens)]
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
# TP4 FP8 as deployed (MTP2, prefix caching); the question is how aggregate decode throughput scales with the number of concurrent streams the KV cache can hold
ports = start_servers("tp4", FP8, [[0, 1, 2, 3]], 4, extra=MTP(2) + PC, max_len=65536, max_seqs=64)
if wait_ready("tp4", ports):
    try:
        micro(ports[0])
    except Exception as e:
        log("micro failed", repr(e)[:300])
    for n in (24, 32, 40):
        log("=" * 10, "streams", n)
        asyncio.run(bench(f"tp4_fp8_mtp2_pc_{n}streams", ports, n, hi=40000, duration=540))
stop_servers()
log("ALL DONE")
