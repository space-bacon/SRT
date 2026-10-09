"""Aggregate decode throughput of one vLLM server under N concurrent real ARC prompts. Appends one JSON line per level to --out."""
import argparse, asyncio, json, time, re, urllib.request
from openai import AsyncOpenAI
import run_llm_arc as R


def metric(name):
    txt = urllib.request.urlopen("http://127.0.0.1:8001/metrics").read().decode()
    tot = 0.0
    for line in txt.splitlines():
        if line.startswith(name) and not line.startswith("#"):
            tot += float(line.rsplit(" ", 1)[1])
    return tot


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--levels", default="2,4,8,12")
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--min-prompt", type=int, default=6000)
    a = ap.parse_args()
    data = json.load(open(a.challenges))
    prompts = []
    for tid, t in data.items():
        p = R.build_prompt(t, 0)
        if R.est_tokens(p) >= a.min_prompt:
            prompts.append((R.est_tokens(p), p))
    prompts.sort(reverse=True)
    prompts = [p for _, p in prompts[:24]]
    client = AsyncOpenAI(base_url="http://127.0.0.1:8001/v1", api_key="x", timeout=3600, max_retries=0)
    for n in [int(x) for x in a.levels.split(",")]:
        g0, ac0, dr0 = metric("vllm:generation_tokens_total"), metric("vllm:spec_decode_num_accepted_tokens_total"), metric("vllm:spec_decode_num_draft_tokens_total")
        t0 = time.time()

        async def one(i):
            r = await client.chat.completions.create(model="qwen", messages=[{"role": "user", "content": prompts[i % len(prompts)]}],
                                                      temperature=1.0, top_p=0.95, max_tokens=a.max_tokens, extra_body={"top_k": 20})
            return r.usage.completion_tokens, r.usage.prompt_tokens
        res = await asyncio.gather(*(one(i) for i in range(n)))
        dt = time.time() - t0
        g1, ac1, dr1 = metric("vllm:generation_tokens_total"), metric("vllm:spec_decode_num_accepted_tokens_total"), metric("vllm:spec_decode_num_draft_tokens_total")
        rec = {"tag": a.tag, "concurrency": n, "seconds": round(dt, 1), "completion_tokens": sum(r[0] for r in res),
               "agg_tok_per_s": round(sum(r[0] for r in res) / dt, 1), "per_seq_tok_per_s": round(sum(r[0] for r in res) / dt / n, 1),
               "mean_prompt_tokens": round(sum(r[1] for r in res) / n), "accept_rate": round((ac1 - ac0) / (dr1 - dr0), 3) if dr1 > dr0 else None}
        print(json.dumps(rec), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")

asyncio.run(main())
