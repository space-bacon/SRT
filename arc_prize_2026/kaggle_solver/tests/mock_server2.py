import asyncio, json, sys
from aiohttp import web

IDENT = "def transform(grid):\n    return grid\n"
GEN = {"n": 0}


async def metrics(request):
    return web.Response(text=f"vllm:generation_tokens_total{{model_name=\"qwen\"}} {GEN['n']}\n")


async def completions(request):
    body = await request.json()
    prompt = body.get("prompt", "")
    n = body.get("n", 1)
    max_tokens = body.get("max_tokens", 100)
    if body.get("stream"):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        words = ["thinking "] * min(max_tokens, 30)
        for w in words:
            chunk = {"id": "x", "object": "text_completion", "created": 0, "model": "qwen", "choices": [{"index": 0, "text": w, "finish_reason": None}]}
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
            await asyncio.sleep(0.003)
        GEN["n"] += len(words)
        finish = "length"
        last = {"id": "x", "object": "text_completion", "created": 0, "model": "qwen", "choices": [{"index": 0, "text": "", "finish_reason": finish}]}
        await resp.write(f"data: {json.dumps(last)}\n\n".encode())
        usage = {"id": "x", "object": "text_completion", "created": 0, "model": "qwen", "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": len(words), "total_tokens": 10 + len(words)}}
        await resp.write(f"data: {json.dumps(usage)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp
    GEN["n"] += 20 * n
    choices = [{"index": i, "text": IDENT + "```\n", "finish_reason": "stop"} for i in range(n)]
    return web.json_response({"id": "x", "object": "text_completion", "created": 0, "model": "qwen", "choices": choices,
                              "usage": {"prompt_tokens": 10, "completion_tokens": 20 * n, "total_tokens": 10 + 20 * n}})


app = web.Application()
app.router.add_post("/v1/completions", completions)
app.router.add_get("/metrics", metrics)
web.run_app(app, port=int(sys.argv[1]), print=None)
