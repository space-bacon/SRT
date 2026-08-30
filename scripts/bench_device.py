#!/usr/bin/env python3
"""Same task, same code, two devices. Run it here and on a GPU box, then diff.

Comparing "our speed" to "a GPU" is only meaningful if both sides run the same
work under the same measurement. This script defines that work as the two things
the atlas actually does, so the comparison is against real workload rather than a
synthetic matmul:

  encode  N texts -> hidden states at a fixed relative depth, mean-pooled
  decode  greedy continuation, tokens per second

Reports device, dtype, load seconds, peak memory, and throughput. Writes JSON
keyed by device so results from different machines can sit side by side.

Measurement discipline, because throughput claims are easy to inflate:
  - a warmup pass is discarded, so kernel compilation is not counted as work
  - device is synchronized before every timer stop
  - tokens are counted from the attention mask, not estimated from characters
  - refuses to run when other heavy jobs are live, since contention silently
    halves the number and nothing in the output would reveal it

    python scripts/bench_device.py --model Qwen/Qwen2.5-0.5B --texts 1000
    # then on a CUDA box, same flags, and compare the two JSON files
"""
import argparse
import json
import os
import platform
import subprocess
import time

import torch

DEV = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--texts", type=int, default=1000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq", type=int, default=48)
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--layer-frac", type=float, default=0.6)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--force", action="store_true", help="run even if the box is busy")
    p.add_argument("--out", default="artifacts/nla/atlas/bench_device.json")
    return p.parse_args()


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()
    elif DEV == "mps":
        torch.mps.synchronize()


JOBS = ("atlas", "hivemind", "sunstone_server", "train_", "encode")


def busy_check(force):
    """A contended box reports half its real throughput and looks fine doing it.

    Match job keywords directly. Prefiltering on "python" silently matches nothing
    on macOS, where the interpreter is .../Python.app/Contents/MacOS/Python and
    pgrep is case-sensitive, so the guard passes vacuously.
    """
    heavy, me = [], os.getpid()
    for kw in JOBS:
        try:
            r = subprocess.run(["pgrep", "-fl", kw], capture_output=True, text=True)
        except Exception:
            continue
        for line in r.stdout.splitlines():
            pid = line.split(None, 1)[0]
            if pid.isdigit() and int(pid) != me and "bench_device" not in line:
                heavy.append(line)
    heavy = sorted(set(heavy))
    if heavy and not force:
        print("REFUSING TO BENCHMARK: heavy jobs are live and would corrupt the numbers.")
        for l in heavy[:5]:
            print("   ", l[:100])
        print("\nWait for them to finish, or pass --force to measure anyway.")
        raise SystemExit(1)
    if heavy:
        print(f"WARNING: {len(heavy)} heavy job(s) live, numbers are contended.\n")


def device_info():
    d = {"device": DEV, "torch": torch.__version__, "platform": platform.platform(),
         "machine": platform.machine()}
    if DEV == "cuda":
        d["name"] = torch.cuda.get_device_name(0)
        d["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    else:
        try:
            d["name"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                       capture_output=True, text=True).stdout.strip()
            d["unified_mem_gb"] = round(int(subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True,
                text=True).stdout) / 1e9)
        except Exception:
            pass
    return d


def peak_gb():
    if DEV == "cuda":
        return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    if DEV == "mps":
        return round(torch.mps.current_allocated_memory() / 1e9, 2)
    return None


def main():
    a = parse_args()
    busy_check(a.force)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    info = device_info()
    print(f"device {DEV}  {info.get('name', '?')}")
    print(f"model  {a.model}  dtype {a.dtype}\n")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mod = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=getattr(torch, a.dtype)).to(DEV).eval()
    sync()
    load_s = time.time() - t0
    L = mod.config.num_hidden_layers
    layer = max(1, int(round(a.layer_frac * L)))

    texts = [f"The quick brown fox number {i} jumps over a lazy dog near the river bank "
             f"while considering what to do next." for i in range(a.texts)]

    # warmup, discarded: first pass pays for kernel compilation
    with torch.no_grad():
        b = tok(texts[:a.batch], padding=True, truncation=True, max_length=a.seq,
                return_tensors="pt").to(DEV)
        mod(**b, output_hidden_states=True)
    sync()

    ntok = 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), a.batch):
            b = tok(texts[i:i + a.batch], padding=True, truncation=True,
                    max_length=a.seq, return_tensors="pt").to(DEV)
            h = mod(**b, output_hidden_states=True).hidden_states[layer]
            m = b["attention_mask"].unsqueeze(-1).float()
            (h * m).sum(1) / m.sum(1).clamp(min=1)
            ntok += int(b["attention_mask"].sum())
    sync()
    enc_s = time.time() - t0

    prompt = "The best way to learn a new language is"
    b = tok([prompt], return_tensors="pt").to(DEV)
    with torch.no_grad():
        mod.generate(**b, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
    sync()
    t0 = time.time()
    with torch.no_grad():
        g = mod.generate(**b, max_new_tokens=a.gen_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    sync()
    gen_s = time.time() - t0
    new_tok = int(g.shape[1] - b["input_ids"].shape[1])

    res = {"device_info": info, "model": a.model, "dtype": a.dtype,
           "hidden_layers": L, "tap_layer": layer,
           "load_seconds": round(load_s, 2), "peak_gb": peak_gb(),
           "encode": {"n_texts": a.texts, "batch": a.batch, "max_seq": a.seq,
                      "seconds": round(enc_s, 2),
                      "texts_per_s": round(a.texts / enc_s, 1),
                      "tokens_per_s": round(ntok / enc_s, 1)},
           "decode": {"new_tokens": new_tok, "seconds": round(gen_s, 2),
                      "tokens_per_s": round(new_tok / gen_s, 2)}}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    prev = json.load(open(a.out)) if os.path.isfile(a.out) else {}
    key = f"{DEV}:{info.get('name', '?')[:40]}:{a.model.split('/')[-1]}"
    prev[key] = res
    json.dump(prev, open(a.out, "w"), indent=1)

    print(f"load        {res['load_seconds']}s   peak {res['peak_gb']} GB")
    print(f"encode      {res['encode']['texts_per_s']} texts/s   "
          f"{res['encode']['tokens_per_s']} tok/s")
    print(f"decode      {res['decode']['tokens_per_s']} tok/s   "
          f"({new_tok} tokens in {res['decode']['seconds']}s)")
    if len(prev) > 1:
        print(f"\ncomparison across {len(prev)} runs in {a.out}:")
        for k, v in prev.items():
            print(f"  {k:52s} enc {v['encode']['tokens_per_s']:>9.1f} tok/s   "
                  f"dec {v['decode']['tokens_per_s']:>7.2f} tok/s")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
