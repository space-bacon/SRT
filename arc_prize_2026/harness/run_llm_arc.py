#!/usr/bin/env python
"""Sample K reasoning traces per ARC test input from vLLM replicas.

One JSON line per finished request is appended to --out (resume skips finished pairs); reasoning text goes to
traces/<key>__<sample>.txt.gz via temp file and rename.
"""
import argparse, asyncio, base64, gzip, io, json, os, random, time
from openai import AsyncOpenAI

HEADER = (
    "You are participating in a puzzle solving competition. You are an expert at solving puzzles.\n\n"
    "Below is a list of input and output pairs with a pattern. Your goal is to identify the pattern or "
    "transformation in the training examples that maps the input to the output, then apply that "
    "transformation to the test input to give a final output.\n\n"
    "Grids are rows of digits 0-9, one row per line, no separators. Each digit is a color.\n"
    "After your reasoning, give only the final output grid in the same format inside one fenced code block, "
    "and write nothing after the code block.\n\n")


HEADER_CODE = (
    "You are participating in a puzzle solving competition. You are an expert programmer and puzzle solver.\n\n"
    "Below is a list of input and output grid pairs that share one transformation rule. Your goal is to find the rule "
    "and write a Python function `transform(grid)` that maps any input grid to its output grid. "
    "`grid` is a list of lists of ints (0-9); return a list of lists of ints. You may import numpy. "
    "The function must reproduce every training output exactly and must generalize to the test input.\n\n"
    "Grids are shown as rows of digits 0-9, one row per line, no separators. Each digit is a color.\n"
    "After your reasoning, give only the complete Python code (imports plus the function) inside one fenced ```python code block, "
    "and write nothing after the code block.\n\n")


def grid_text(g):
    return "\n".join("".join(str(c) for c in r) for r in g)


def build_prompt(task, ti, mode="direct"):
    parts = [HEADER_CODE if mode == "code" else HEADER, "--Training Examples--\n"]
    for i, ex in enumerate(task["train"]):
        parts.append(f"--Example {i}--\nINPUT:\n{grid_text(ex['input'])}\nOUTPUT:\n{grid_text(ex['output'])}\n")
    parts.append(f"--Test Input--\n{grid_text(task['test'][ti]['input'])}\n")
    return "\n".join(parts)


PALETTE = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]


def render_png(g):
    from PIL import Image, ImageDraw
    h, w = len(g), len(g[0])
    cell = max(10, min(32, 640 // max(h, w)))
    im = Image.new("RGB", (w * cell + 1, h * cell + 1), "#555555")
    d = ImageDraw.Draw(im)
    for r in range(h):
        for c in range(w):
            d.rectangle([c * cell + 1, r * cell + 1, (c + 1) * cell - 1, (r + 1) * cell - 1], fill=PALETTE[g[r][c]])
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_content(task, ti):
    """Same text as build_prompt with a rendered image after every grid."""
    def img(g):
        return {"type": "image_url", "image_url": {"url": render_png(g)}}
    parts = [{"type": "text", "text": HEADER + "Each grid is given as text and again as an image of the same grid.\n\n--Training Examples--\n"}]
    for i, ex in enumerate(task["train"]):
        parts.append({"type": "text", "text": f"--Example {i}--\nINPUT:\n{grid_text(ex['input'])}\n(image of the input)"})
        parts.append(img(ex["input"]))
        parts.append({"type": "text", "text": f"OUTPUT:\n{grid_text(ex['output'])}\n(image of the output)"})
        parts.append(img(ex["output"]))
    parts.append({"type": "text", "text": f"--Test Input--\n{grid_text(task['test'][ti]['input'])}\n(image of the test input)"})
    parts.append(img(task["test"][ti]["input"]))
    return parts


def est_tokens(prompt):
    # digits and newlines are one token each; words about 4 chars per token
    grid_chars = sum(1 for ch in prompt if ch.isdigit() or ch == "\n")
    return int(grid_chars * 1.02 + (len(prompt) - grid_chars) / 3.5) + 64


def atomic_gz(path, text):
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt") as f:
        f.write(text)
    os.replace(tmp, path)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--ports", default="8001,8002,8003,8004")
    ap.add_argument("--per-replica", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=110000)
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--effort", default="")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--mode", default="direct", choices=["direct", "code"])
    ap.add_argument("--sample-offset", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.traces, exist_ok=True)
    data = json.load(open(args.challenges))
    keep = set(json.load(open(args.tasks))) if args.tasks else None
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") == "ok":
                done.add((r["key"], r["sample"]))

    work = []
    for tid, task in data.items():
        if keep is not None and tid not in keep:
            continue
        for ti in range(len(task["test"])):
            prompt = build_prompt(task, ti, args.mode)
            content = build_content(task, ti) if args.images else prompt
            for s in range(args.sample_offset, args.sample_offset + args.k):
                if (f"{tid}_{ti}", s) not in done:
                    work.append((s, f"{tid}_{ti}", prompt, content))
    random.Random(args.seed).shuffle(work)
    work.sort(key=lambda w: w[0])
    ports = [int(p) for p in args.ports.split(",")]
    clients = [AsyncOpenAI(base_url=f"http://127.0.0.1:{p}/v1", api_key="x", timeout=172800, max_retries=0) for p in ports]
    sems = [asyncio.Semaphore(args.per_replica) for _ in ports]
    inflight = [0] * len(ports)
    stats = {"done": 0, "tokens": 0, "err": 0}
    t_start = time.time()
    out_f = open(args.out, "a")

    def emit(rec):
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()

    async def one(s, key, prompt, content):
        r = min(range(len(ports)), key=lambda i: inflight[i])
        inflight[r] += 1
        try:
            async with sems[r]:
                mt = min(args.max_tokens, args.ctx - est_tokens(prompt) - (8000 if args.images else 512))
                if mt < 4096:
                    emit({"key": key, "sample": s, "status": "prompt_too_long", "prompt_est": est_tokens(prompt)})
                    return
                extra = {"top_k": 20, "seed": args.seed * 1000003 + s * 7919 + hash(key) % 100000}
                if args.effort:
                    extra["chat_template_kwargs"] = {"reasoning_effort": args.effort}
                t0 = time.time()
                for attempt in range(3):
                    try:
                        resp = await clients[r].chat.completions.create(
                            model="qwen", messages=[{"role": "user", "content": content}],
                            temperature=args.temperature, top_p=0.95, max_tokens=mt, extra_body=extra)
                        break
                    except Exception as e:
                        stats["err"] += 1
                        emit({"key": key, "sample": s, "status": "error", "attempt": attempt, "err": repr(e)[:300]})
                        await asyncio.sleep(10 * (attempt + 1))
                else:
                    return
                msg = resp.choices[0].message
                extra_f = getattr(msg, "model_extra", None) or {}
                reasoning = extra_f.get("reasoning_content") or extra_f.get("reasoning") or ""
                atomic_gz(os.path.join(args.traces, f"{key}__{s}.txt.gz"), reasoning)
                u = resp.usage
                emit({"key": key, "sample": s, "status": "ok", "replica": r, "t0": t0, "t1": time.time(),
                      "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                      "finish": resp.choices[0].finish_reason, "content": msg.content or "",
                      "n_reasoning_chars": len(reasoning), "max_tokens": mt})
                stats["done"] += 1
                stats["tokens"] += u.completion_tokens
        finally:
            inflight[r] -= 1

    async def heartbeat():
        while True:
            await asyncio.sleep(60)
            el = time.time() - t_start
            emit({"status": "heartbeat", "t": time.time(), "done": stats["done"], "err": stats["err"],
                  "tokens": stats["tokens"], "tok_per_s": round(stats["tokens"] / max(el, 1), 1),
                  "inflight": list(inflight), "todo": len(work) - stats["done"]})

    hb = asyncio.create_task(heartbeat())
    emit({"status": "run_start", "t": time.time(), "n_work": len(work), "k": args.k, "ports": ports})
    await asyncio.gather(*(one(*w) for w in work))
    hb.cancel()
    emit({"status": "run_end", "t": time.time(), "done": stats["done"], "tokens": stats["tokens"]})


if __name__ == "__main__":
    asyncio.run(main())
