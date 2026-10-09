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
