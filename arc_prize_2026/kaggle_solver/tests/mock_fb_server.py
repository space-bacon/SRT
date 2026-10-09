import asyncio, json, sys
from aiohttp import web

IDENT = "import numpy as np\n\ndef transform(grid):\n    return grid\n```\n"


async def completions(request):
    body = await request.json()
    n = body.get("n", 1)
    prompt = body.get("prompt", "")
    forced = "I have run out of thinking time" in prompt[-300:]
    await asyncio.sleep(0.05)
    if forced:
        choices = [{"index": i, "text": IDENT, "finish_reason": "stop"} for i in range(n)]
        usage = {"prompt_tokens": 100, "completion_tokens": 30 * n, "total_tokens": 100 + 30 * n}
    else:
        choices = [{"index": 0, "text": "still thinking " * 20, "finish_reason": "length"}]
        usage = {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}
    return web.json_response({"id": "x", "object": "text_completion", "created": 0, "model": "qwen", "choices": choices, "usage": usage})


app = web.Application()
app.router.add_post("/v1/completions", completions)
web.run_app(app, port=int(sys.argv[1]), print=None)
