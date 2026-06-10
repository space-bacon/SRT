"""Minimal OpenAI-compatible server for the SRT adapter.

Serves `SRTAdapter` (frozen backbone + semiotic adapter) behind a
`/v1/chat/completions` endpoint so external agent harnesses
(ResearchHarness, etc.) can drive it like any OpenAI model.

The SRT forward runs the backbone decoder layers manually to inject the
FiLM correction, so it is NOT compatible with vLLM/SGLang. Generation here
is a plain autoregressive loop with NO KV cache (each step re-runs the full
forward over the growing sequence). This is correct but O(T^2); fine for
benchmarking, not for production throughput.

A/B switch:
    --inject     (default) SRT injection ON  -> "the SRT"
    --no-inject            SRT injection OFF -> bare frozen backbone

Run:
    python scripts/serve_srt_openai.py \
        --backbone Qwen/Qwen2.5-7B-Instruct \
        --adapter-ckpt checkpoints/best_adapter_step30k.pt \
        --port 8080 --inject

Then point an OpenAI client at http://localhost:8080/v1 with any api_key.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import uuid
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

from srt.adapter import SRTAdapter
from srt.config import SRTConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("serve_srt")

# Populated in main().
STATE: dict = {}


# ── OpenAI-compatible request/response schemas ───────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str | list | None = None
    # Present on assistant messages that previously emitted tool calls, and on
    # tool-result messages the harness feeds back in.
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int | None = None
    stop: list[str] | str | None = None
    stream: bool = False
    n: int = 1
    # Native OpenAI function-calling fields. The ResearchHarness ReAct agent
    # sends these; without honoring them the model can never invoke a tool.
    tools: list[dict] | None = None
    tool_choice: Any | None = None
    parallel_tool_calls: bool | None = None


# Qwen2.5 emits Hermes-style tool calls:  <tool_call>\n{json}\n</tool_call>
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _prepare_messages_for_template(messages: list[dict]) -> list[dict]:
    """Normalize OpenAI messages for the Qwen chat template.

    The OpenAI wire format carries assistant `tool_calls[].function.arguments`
    as a JSON *string*; the Qwen template expects a dict so it can re-render the
    `<tool_call>` block. Convert those here and drop empty fields the template
    does not understand.
    """
    out: list[dict] = []
    for m in messages:
        msg = {k: v for k, v in m.items() if v is not None}
        if isinstance(msg.get("content"), list):
            # Flatten structured content to text (we only generate/consume text).
            parts = []
            for c in msg["content"]:
                if isinstance(c, dict):
                    parts.append(str(c.get("text", "")))
                else:
                    parts.append(str(c))
            msg["content"] = "".join(parts)
        if msg.get("content") is None and "tool_calls" not in msg:
            msg["content"] = ""
        tcs = msg.get("tool_calls")
        if tcs:
            new_tcs = []
            for tc in tcs:
                fn = dict(tc.get("function", {}))
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        fn["arguments"] = {}
                new_tcs.append({**tc, "function": fn})
            msg["tool_calls"] = new_tcs
        out.append(msg)
    return out


def _parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Extract Hermes `<tool_call>` blocks from generated text.

    Returns (cleaned_text, tool_calls) where tool_calls is OpenAI-format. The
    cleaned text has the tool-call blocks removed (kept as message content).
    """
    tool_calls: list[dict] = []
    for match in _TOOL_CALL_RE.finditer(text):
        raw = match.group(1)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name = obj.get("name", "")
        args = obj.get("arguments", {})
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": name,
                # OpenAI wire format wants arguments as a JSON string.
                "arguments": json.dumps(args, ensure_ascii=False)
                if not isinstance(args, str) else args,
            },
        })
    cleaned = _TOOL_CALL_RE.sub("", text).strip()
    return cleaned, tool_calls



def _load_model(backbone_id: str, adapter_ckpt: str | None, device: str) -> None:
    cfg = SRTConfig(backbone_id=backbone_id)
    logger.info("Building SRTAdapter on backbone %s ...", backbone_id)
    model = SRTAdapter(cfg)

    if adapter_ckpt:
        logger.info("Loading adapter weights from %s", adapter_ckpt)
        sd = torch.load(adapter_ckpt, map_location="cpu", weights_only=False)
        # Checkpoint may be a bare adapter state_dict or wrapped under
        # "trainable"/"state_dict"/"model".
        if isinstance(sd, dict):
            for wrap in ("trainable", "state_dict", "model"):
                if wrap in sd and isinstance(sd[wrap], dict):
                    sd = sd[wrap]
                    break
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # The frozen backbone params live under _embed_tokens / _layers /
        # _final_norm / _lm_head and are loaded from pretrained, so they are
        # expected to be "missing" from the adapter checkpoint. Anything else
        # missing is a real adapter key that did NOT load (e.g. a v3 checkpoint
        # on v4 FiLM code), which silently leaves it at random init.
        backbone_prefixes = ("_embed_tokens", "_layers", "_final_norm", "_lm_head", "backbone")
        adapter_missing = [
            k for k in missing if not k.startswith(backbone_prefixes)
        ]
        if adapter_missing:
            logger.warning(
                "ADAPTER keys MISSING from checkpoint (left at RANDOM INIT): %s",
                adapter_missing,
            )
        if unexpected:
            logger.warning(
                "UNEXPECTED checkpoint keys (NOT loaded — likely version mismatch): %s",
                unexpected,
            )
        if adapter_missing or unexpected:
            logger.warning(
                "Checkpoint/code version MISMATCH detected. Injection may be untrained."
            )
        logger.info(
            "Adapter loaded: %d adapter keys matched, %d backbone keys from pretrained",
            len(sd) - len(unexpected),
            sum(1 for k in missing if k.startswith(backbone_prefixes)),
        )

    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(backbone_id)
    STATE["model"] = model
    STATE["tok"] = tok
    STATE["device"] = device
    STATE["backbone_id"] = backbone_id
    # eos / pad ids for stopping
    eos_ids = set()
    if tok.eos_token_id is not None:
        eos_ids.add(tok.eos_token_id)
    # Qwen chat end-of-turn token
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        eos_ids.add(im_end)
    STATE["eos_ids"] = eos_ids
    logger.info("Ready. eos_ids=%s", eos_ids)


@torch.no_grad()
def _generate(
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop_strings: list[str],
    disable_injectors: bool,
) -> tuple[str, int, str]:
    """KV-cached generation via SRTAdapter.generate(). Returns (text, n_new, finish_reason)."""
    model = STATE["model"]
    tok = STATE["tok"]
    eos_ids = STATE["eos_ids"]
    device = STATE["device"]

    new_ids = model.generate(
        input_ids.to(device),
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_ids,
        temperature=temperature,
        top_p=top_p,
        disable_injectors=disable_injectors,
    )[0]

    n_new = int(new_ids.shape[0])
    last_is_eos = n_new > 0 and int(new_ids[-1].item()) in eos_ids
    finish_reason = "stop" if last_is_eos else "length"
    text = tok.decode(new_ids, skip_special_tokens=True)

    # Trim at the first stop string if present.
    for s in stop_strings:
        idx = text.find(s)
        if idx != -1:
            text = text[:idx]
            finish_reason = "stop"
    return text, n_new, finish_reason


app = FastAPI()


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": STATE.get("served_name", "srt"), "object": "model", "owned_by": "srt"}],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": STATE.get("served_name")}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    tok = STATE["tok"]
    messages = _prepare_messages_for_template([m.model_dump() for m in req.messages])

    template_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if req.tools:
        template_kwargs["tools"] = req.tools
    prompt = tok.apply_chat_template(messages, **template_kwargs)
    input_ids = tok(prompt, return_tensors="pt").input_ids

    stop_strings = []
    if isinstance(req.stop, str):
        stop_strings = [req.stop]
    elif isinstance(req.stop, list):
        stop_strings = req.stop

    max_new = req.max_tokens or STATE["default_max_tokens"]
    t0 = time.time()
    text, n_new, finish_reason = _generate(
        input_ids,
        max_new_tokens=max_new,
        temperature=req.temperature,
        top_p=req.top_p,
        stop_strings=stop_strings,
        disable_injectors=STATE["disable_injectors"],
    )
    dt = time.time() - t0
    prompt_toks = int(input_ids.shape[1])

    # Parse Hermes tool-call blocks (only when tools were offered).
    tool_calls: list[dict] = []
    if req.tools:
        text, tool_calls = _parse_tool_calls(text)
    if tool_calls:
        finish_reason = "tool_calls"

    logger.info(
        "completion: %d prompt + %d new tokens in %.1fs (%.1f tok/s) inject=%s tool_calls=%d",
        prompt_toks, n_new, dt, n_new / dt if dt else 0.0,
        not STATE["disable_injectors"], len(tool_calls),
    )

    message: dict = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": STATE.get("served_name", "srt"),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_toks,
            "completion_tokens": n_new,
            "total_tokens": prompt_toks + n_new,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter-ckpt", default=None)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--served-name", default="srt-qwen2.5-7b")
    ap.add_argument("--default-max-tokens", type=int, default=2048)
    inj = ap.add_mutually_exclusive_group()
    inj.add_argument("--inject", dest="inject", action="store_true",
                     help="SRT injection ON (default)")
    inj.add_argument("--no-inject", dest="inject", action="store_false",
                     help="SRT injection OFF (bare backbone control)")
    ap.set_defaults(inject=True)
    args = ap.parse_args()

    _load_model(args.backbone, args.adapter_ckpt, args.device)
    STATE["disable_injectors"] = not args.inject
    STATE["served_name"] = args.served_name
    STATE["default_max_tokens"] = args.default_max_tokens
    logger.info(
        "Serving '%s' on %s:%d | inject=%s | backbone=%s",
        args.served_name, args.host, args.port, args.inject, args.backbone,
    )

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
