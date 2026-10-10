
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
    solver_args += ["--prompt-summary", "2" if str(CFG["prompt_summary"]) == "2" else "1"]
if CFG.get("prompt_image"):
    solver_args += ["--prompt-image"]
if CFG.get("wait_continue"):
    solver_args += ["--wait-continue", str(CFG["wait_continue"])]
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
