"""Live consensus worker: sample K replies, run them, return the pick and the receipts.

This runs on a throwaway GPU box and nowhere else. It generates code with a
language model in response to a stranger's prompt and then executes that code,
so it is a remote code execution primitive by design. The Lab gateway reaches it
over an SSH tunnel only; the port binds to loopback; every child runs under
`srt_select.sandbox` plus whatever `SRT_SELECT_WRAP` adds (no network, nobody
uid). The box holds no secrets and is destroyed when the demo is over.

    SRT_SELECT_WRAP="unshare -n -- setpriv --reuid=65534 --regid=65534 --clear-groups" \
    python scripts/consensus_worker.py --model Qwen/Qwen2.5-Coder-3B-Instruct

Sampling matches the banked `chat` arm: temperature 1.0, top_p 0.9, the model's
own chat template, K = 8. The number a visitor sees is therefore the number the
paper reports, not a tuned demo setting.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from srt_select.consensus import choose  # noqa: E402
from srt_select.recover import code_of  # noqa: E402

app = FastAPI(title="consensus-worker")
LLM = None
TOK = None
SP = None
MODEL = ""
K = 8

# One generation at a time on one card; a short queue behind it.
GEN_LOCK = threading.Lock()
WAITING = threading.Semaphore(4)
MAX_PROMPT_CHARS = 4000


class Ask(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)


@app.get("/healthz")
def healthz():
    return {"ok": LLM is not None, "model": MODEL, "k": K,
            "wrap": os.environ.get("SRT_SELECT_WRAP", "")}


@app.post("/select")
def select(ask: Ask):
    if LLM is None:
        raise HTTPException(503, "model not loaded")
    if not WAITING.acquire(blocking=False):
        raise HTTPException(503, "busy with other visitors")
    try:
        with GEN_LOCK:
            t0 = time.time()
            text = TOK.apply_chat_template(
                [{"role": "user", "content": ask.prompt}],
                tokenize=False, add_generation_prompt=True)
            out = LLM.generate([text], SP, use_tqdm=False)[0]
            replies = [o.text for o in out.outputs]
            t_gen = time.time() - t0
            t1 = time.time()
            pick = choose(ask.prompt, replies)
            t_sel = time.time() - t1
    finally:
        WAITING.release()

    idx = pick.index if pick.index is not None else 0
    return {
        "model": MODEL,
        "k": len(replies),
        "replies": replies,
        "codes": [code_of(r) for r in replies],
        "groups": pick.groups or [-1] * len(replies),
        "pick": idx,
        "resolved": pick.index is not None,
        "reason": pick.reason,
        "entry": pick.entry,
        "kinds": pick.kinds,
        "ran": pick.ran,
        "agreed": pick.cluster_size,
        "clusters": pick.clusters,
        "seconds": {"generate": round(t_gen, 2), "select": round(t_sel, 2)},
    }


def main() -> None:
    global LLM, TOK, SP, MODEL, K
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()

    from vllm import LLM as _LLM, SamplingParams
    import uvicorn

    MODEL, K = os.path.basename(a.model.rstrip("/")), a.k
    LLM = _LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_frac,
               max_model_len=a.max_len, enable_prefix_caching=True)
    TOK = LLM.get_tokenizer()
    SP = SamplingParams(n=a.k, temperature=a.temp, top_p=a.top_p, max_tokens=a.max_new)
    # One warm request so the first visitor does not pay the graph capture.
    LLM.generate([TOK.apply_chat_template([{"role": "user", "content": "def f(x): return x"}],
                                          tokenize=False, add_generation_prompt=True)],
                 SamplingParams(n=1, max_tokens=8), use_tqdm=False)
    print(f"worker ready: {MODEL} K={K} wrap={os.environ.get('SRT_SELECT_WRAP', '')!r}",
          flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
