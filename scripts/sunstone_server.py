"""SRT-Sunstone local serving core: gemma-4-31B (4-bit MLX) + linear head, HTTP.

The home-computer serving test for the instrumented-appliance architecture:
one quantized multimodal backbone on Apple Silicon serving chat, captioning,
and cross-modal retrieval from the SAME forward passes, with the shipped
44MB linear head reading layer-47 states. This is the workload a leased
CPU/ARM box would run; the /bench endpoint gives the numbers to compare
against candidate hardware.

Endpoints:
  GET  /healthz            model + head + gallery status, peak memory
  POST /chat               {"prompt": str, "max_tokens": int} -> text + tok/s
  POST /chat_trace         chat + per-token entropy, alternatives, and
                           head-space readouts (nearest gallery captions)
  POST /caption            multipart image (+ optional prompt) -> native VLM text
  POST /retrieve           multipart image -> top-K captions via the head
  POST /search             {"text": str} -> top-K COCO gallery images via the head
  GET  /bench              fixed-prompt decode benchmark -> tok/s

Run:
  .venv/bin/python scripts/sunstone_server.py            # port 8765
  curl -s localhost:8765/healthz | jq
  curl -s -X POST localhost:8765/chat -H 'content-type: application/json' \
       -d '{"prompt": "What is a sunstone?"}' | jq -r .text
  curl -s -X POST localhost:8765/retrieve -F image=@photo.jpg | jq
"""
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
# NOTE: no `from __future__ import annotations` here. PEP 563 string
# annotations break FastAPI's dependency resolution for request models
# and UploadFile params.

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("sunstone")

MODEL_ID = os.environ.get("SUNSTONE_MODEL", "mlx-community/gemma-4-31b-it-4bit")
TAP_LAYER = int(os.environ.get("SUNSTONE_LAYER", "47"))
HEAD_REPO = "RiverRider/srt-sunstone-linear-head"
HEAD_FILE = "sunstone_linear_head.pt"
CALIB_REPO = "RiverRider/srt-nla-gemma4-artifacts"
CALIB_FILE = "procrustes/encoded_L47_n5000.pt"
LOCAL_MU_CACHE = Path("artifacts/local/local_mu_txt.npy")
RECAL_N = 256           # captions used for the 42KB local-mean recalibration
MAX_PROMPT_CHARS = 4000
MAX_TOKENS_CAP = 1024

S = {}                  # server state, filled in lifespan


class ChatReq(BaseModel):
    prompt: str
    max_tokens: int = 256


class TraceReq(BaseModel):
    prompt: str
    max_tokens: int = 256
    budget: int = 8


class SearchReq(BaseModel):
    text: str
    k: int = 5


# --------------------------------------------------------------------- head
def _project(v: np.ndarray, W: np.ndarray, b: np.ndarray,
             mu: np.ndarray) -> np.ndarray:
    """Center, apply the linear head, L2-normalize. v: [d] or [n, d]."""
    z = (v - mu) @ W.T + b
    return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)


def load_head():
    import torch
    from huggingface_hub import hf_hub_download

    d = torch.load(hf_hub_download(HEAD_REPO, HEAD_FILE),
                   map_location="cpu", weights_only=True)
    head = {
        "W_img": d["img"]["weight"].float().numpy(),
        "b_img": d["img"]["bias"].float().numpy(),
        "W_txt": d["txt"]["weight"].float().numpy(),
        "b_txt": d["txt"]["bias"].float().numpy(),
        "mu_img": d["mu_img"].float().numpy(),
        "mu_txt": d["mu_txt"].float().numpy(),
        "meta": d["meta"],
    }
    log.info("head loaded: d=%d proj=%d (train i2t r@1=%.3f)",
             head["meta"]["d"], head["meta"]["proj_dim"],
             head["meta"]["i2t"]["r@1"])
    return head


def load_gallery(head):
    """Caption + image galleries from the HF bf16 calibration cache,
    projected once into head space."""
    import torch
    from huggingface_hub import hf_hub_download

    c = torch.load(hf_hub_download(CALIB_REPO, CALIB_FILE),
                   map_location="cpu", weights_only=True)
    captions = [t for caps in c["captions5"] for t in caps]      # 5000 strings
    cap_states = c["cap5"].float().numpy()                        # match rows
    files = [Path(f).name for f in c["files"]]
    gal = {
        "captions": captions,
        "Z_txt": _project(cap_states, head["W_txt"], head["b_txt"],
                          head["mu_txt"]),
        "files": files,
        "Z_img": _project(c["img"].float().numpy(), head["W_img"],
                          head["b_img"], head["mu_img"]),
        "texts_for_recal": captions[:RECAL_N],
    }
    log.info("gallery: %d captions, %d images", len(captions), len(files))
    return gal


# --------------------------------------------------------------------- model
def encode_text_local(texts: list[str], max_seq_len: int = 64) -> np.ndarray:
    """Last-token L47 states from the local MLX model, BOS-prefixed
    (gemma-4 is BOS-sensitive: bare re-encode drops replay 0.9986 -> 0.615)."""
    import mlx.core as mx

    tok = getattr(S["processor"], "tokenizer", S["processor"])
    lm = getattr(S["model"], "language_model", None) or S["model"]
    bos = getattr(tok, "bos_token_id", None) or 2
    out = []
    for t in texts:
        ids = tok.encode(t)[:max_seq_len]
        if ids[0] != bos:
            ids = [bos] + ids[: max_seq_len - 1]
        res = lm(mx.array([ids]), capture_layer_ids=[TAP_LAYER])
        h = res.hidden_states[0]
        mx.eval(h)
        out.append(np.array(h[0, -1, :].astype(mx.float32)))
    return np.stack(out)


def encode_image_local(img) -> np.ndarray:
    """Mean L47 state over image-token positions (Sunstone convention),
    from the local MLX multimodal forward."""
    import mlx.core as mx
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs

    prompt = apply_chat_template(S["processor"], S["config"],
                                 "Describe this image.", num_images=1)
    inputs = prepare_inputs(S["processor"], images=[img], prompts=[prompt])
    input_ids = inputs["input_ids"]
    kwargs = {k: v for k, v in inputs.items()
              if k not in ("input_ids", "pixel_values", "attention_mask")}
    res = S["model"](input_ids, inputs.get("pixel_values"),
                     capture_layer_ids=[TAP_LAYER], **kwargs)
    h = res.hidden_states[0]
    mx.eval(h)
    h = np.array(h[0].astype(mx.float32))                       # [T, d]
    ids = np.array(input_ids[0])
    mask = ids == S["image_token_id"]
    if not mask.any():
        raise ValueError("no image tokens in the processed prompt")
    return h[mask].mean(0)


def local_mu_txt(head, gal) -> np.ndarray:
    """The 42KB recalibration: local text mean over RECAL_N captions.
    Validated procedure (docs/CROSSMODAL_LINEAR_HEAD.md rule 4): swapping
    the shipped mu_txt for the runtime's own mean took cross-runtime
    head-space agreement 0.984 -> 1.000 on this machine."""
    if LOCAL_MU_CACHE.exists():
        mu = np.load(LOCAL_MU_CACHE)
        if mu.shape == head["mu_txt"].shape:
            log.info("local mu_txt loaded from %s", LOCAL_MU_CACHE)
            return mu
    log.info("computing local mu_txt over %d captions (one-time)...", RECAL_N)
    t0 = time.time()
    states = encode_text_local(gal["texts_for_recal"])
    mu = states.mean(0)
    LOCAL_MU_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(LOCAL_MU_CACHE, mu)
    log.info("local mu_txt computed in %.0fs, cached", time.time() - t0)
    return mu


def generate_text(prompt: str, max_tokens: int, image=None) -> dict:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    tmpl = apply_chat_template(S["processor"], S["config"], prompt,
                               num_images=1 if image is not None else 0)
    t0 = time.time()
    r = generate(S["model"], S["processor"], tmpl,
                 image=[image] if image is not None else None,
                 max_tokens=max_tokens, verbose=False)
    wall = time.time() - t0
    text = getattr(r, "text", r if isinstance(r, str) else str(r))
    return {
        "text": text,
        "wall_s": round(wall, 2),
        "prompt_tokens": getattr(r, "prompt_tokens", None),
        "generation_tokens": getattr(r, "generation_tokens", None),
        "prompt_tps": round(getattr(r, "prompt_tps", 0.0), 1),
        "generation_tps": round(getattr(r, "generation_tps", 0.0), 1),
        "peak_memory_gb": round(getattr(r, "peak_memory", 0.0), 2),
    }


def chat_trace(prompt: str, max_tokens: int, budget: int = 8) -> dict:
    """Chat with a per-token trace: entropy, top alternatives, and a
    head-space readout (nearest gallery caption) for every generated token.

    Same trick as the showcase demo's dense path: generate normally, then
    ONE teacher-forced forward over prompt+output with the L47 tap. The
    logits of that forward reproduce the decode distributions (entropy at
    position i is the model's uncertainty when it produced token i+1), and
    the captured states give each token's place in head space. Readouts are
    observational: 'the caption this state sits nearest', not ground truth.
    """
    gen = generate_text(prompt, max_tokens)
    t0 = time.time()
    tokens = _trace_tokens(prompt, gen["text"], budget)
    return {**gen, "tokens": tokens, "trace_s": round(time.time() - t0, 2)}


def _trace_tokens(prompt: str, gen_text: str, budget: int) -> list[dict]:
    """Teacher-forced trace core: per-token entropy, alternatives, and
    head-space readouts for already-generated text."""
    import mlx.core as mx
    from mlx_vlm.prompt_utils import apply_chat_template

    if not gen_text.strip():
        return []

    tok = getattr(S["processor"], "tokenizer", S["processor"])
    tmpl = apply_chat_template(S["processor"], S["config"], prompt, num_images=0)
    p_ids = tok(tmpl, add_special_tokens=False).input_ids
    full_ids = tok(tmpl + gen_text, add_special_tokens=False).input_ids
    if len(full_ids) <= len(p_ids):
        return []

    lm = getattr(S["model"], "language_model", None) or S["model"]
    res = lm(mx.array([full_ids]), capture_layer_ids=[TAP_LAYER])
    logits = res.logits[0].astype(mx.float32)              # [T, V]
    h = res.hidden_states[0][0].astype(mx.float32)         # [T, d]
    mx.eval(logits, h)

    head, gal = S["head"], S["gallery"]
    plen = len(p_ids)
    n_gen = len(full_ids) - plen

    # Head-space projection of every generated token's own state (one matmul).
    states = np.array(h[plen:])                            # [n_gen, d]
    Z = _project(states, head["W_txt"], head["b_txt"], S["mu_txt_local"])
    sims = Z @ gal["Z_txt"].T                              # [n_gen, n_captions]

    tokens = []
    ents = []
    for j in range(n_gen):
        i = plen - 1 + j                                   # logits that produced token i+1
        pr = mx.softmax(logits[i])
        ent = float(-mx.sum(pr * mx.log(pr + 1e-12)))
        top_ids = mx.argpartition(-pr, kth=2)[:3]
        mx.eval(top_ids)
        alts = sorted(
            ((tok.decode([int(t)]), round(float(pr[int(t)]), 3)) for t in top_ids),
            key=lambda x: -x[1])
        best = int(np.argmax(sims[j]))
        tokens.append({
            "t": tok.decode([full_ids[plen + j]]),
            "ent": round(ent, 3),
            "alts": alts,
            "reads": [[gal["captions"][best], round(float(sims[j][best]), 3)]],
        })
        ents.append(ent)

    # Expand readouts (top-3 captions) at the `budget` highest-entropy tokens,
    # mirroring the showcase demo's adaptive-density verbalization scheduler.
    for j in np.argsort(ents)[::-1][: max(0, budget)]:
        top3 = np.argsort(-sims[j])[:3]
        tokens[j]["reads"] = [[gal["captions"][i], round(float(sims[j][i]), 3)]
                              for i in top3]
        tokens[j]["hot"] = True
    return tokens


def stream_trace(prompt: str, max_tokens: int, budget: int):
    """SSE generator: live tokens with entropy as they are generated, then
    a terminal frame carrying the full enriched trace (alternatives +
    head-space readouts) and generation stats."""
    import json

    from mlx_vlm import stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    tmpl = apply_chat_template(S["processor"], S["config"], prompt, num_images=0)
    t0 = time.time()
    acc = []
    stats = {}
    for r in stream_generate(S["model"], S["processor"], tmpl,
                             max_tokens=max_tokens):
        chunk = r.text or ""
        if chunk:
            acc.append(chunk)
        ent = None
        lp = getattr(r, "logprobs", None)
        if lp is not None:
            try:
                import mlx.core as mx
                if isinstance(lp, mx.array):
                    lp = np.array(lp.astype(mx.float32))
                a = np.asarray(lp, dtype=np.float32).reshape(-1)
                if a.size > 10:
                    ent = round(float(-np.sum(np.exp(a) * a)), 3)
            except Exception:  # noqa: BLE001 - entropy is best-effort live
                ent = None
        stats = {
            "generation_tps": round(getattr(r, "generation_tps", 0.0), 1),
            "peak_memory_gb": round(getattr(r, "peak_memory", 0.0), 2),
        }
        if chunk:
            yield ("data: " + json.dumps({"type": "token", "t": chunk,
                                          "ent": ent}) + "\n\n")
    text = "".join(acc)
    t1 = time.time()
    tokens = _trace_tokens(prompt, text, budget)
    final = {"type": "final", "text": text, "tokens": tokens,
             "wall_s": round(t1 - t0, 2), "trace_s": round(time.time() - t1, 2),
             **stats}
    yield "data: " + json.dumps(final) + "\n\n"
    yield "event: end\ndata: {}\n\n"


# --------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app):
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    t0 = time.time()
    log.info("loading %s ...", MODEL_ID)
    S["model"], S["processor"] = load(MODEL_ID)
    S["config"] = load_config(MODEL_ID)
    img_tok = getattr(S["model"].config, "image_token_id", None)
    S["image_token_id"] = img_tok if img_tok is not None else 258880
    S["head"] = load_head()
    S["gallery"] = load_gallery(S["head"])
    S["mu_txt_local"] = local_mu_txt(S["head"], S["gallery"])
    S["loaded_s"] = round(time.time() - t0, 1)
    log.info("ready in %.1fs", S["loaded_s"])
    yield


def build_app():
    app = FastAPI(title="SRT-Sunstone local serving core", lifespan=lifespan)

    def _check_prompt(p: str):
        if not p.strip() or len(p) > MAX_PROMPT_CHARS:
            raise HTTPException(400, f"prompt must be 1..{MAX_PROMPT_CHARS} chars")

    def _read_image(data: bytes):
        from PIL import Image
        try:
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"not a decodable image: {e}") from e

    @app.get("/healthz")
    def healthz():
        import mlx.core as mx
        return {
            "model": MODEL_ID, "tap_layer": TAP_LAYER,
            "head": {k: v for k, v in S["head"]["meta"].items()
                     if k in ("kind", "n_train", "d", "proj_dim")},
            "gallery_captions": len(S["gallery"]["captions"]),
            "recalibrated": bool(S.get("mu_txt_local") is not None),
            "load_s": S["loaded_s"],
            "peak_memory_gb": round(mx.get_peak_memory() / 2**30, 2),
        }

    @app.post("/chat")
    def chat(req: ChatReq):
        _check_prompt(req.prompt)
        return generate_text(req.prompt, min(req.max_tokens, MAX_TOKENS_CAP))

    @app.post("/chat_trace")
    def chat_trace_route(req: TraceReq):
        _check_prompt(req.prompt)
        return chat_trace(req.prompt, min(req.max_tokens, MAX_TOKENS_CAP),
                          budget=max(0, min(req.budget, 32)))

    @app.get("/stream_trace")
    def stream_trace_route(prompt: str, max_tokens: int = 256, budget: int = 8):
        from fastapi.responses import StreamingResponse

        _check_prompt(prompt)
        return StreamingResponse(
            stream_trace(prompt, min(max_tokens, MAX_TOKENS_CAP),
                         max(0, min(budget, 32))),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/caption")
    def caption(image: UploadFile = File(...),
                prompt: str = Form("Describe this image.")):
        _check_prompt(prompt)
        img = _read_image(image.file.read())
        return generate_text(prompt, 200, image=img)

    @app.post("/retrieve")
    def retrieve(image: UploadFile = File(...), k: int = Form(5)):
        img = _read_image(image.file.read())
        t0 = time.time()
        v = encode_image_local(img)
        z = _project(v, S["head"]["W_img"], S["head"]["b_img"],
                     S["head"]["mu_img"])
        sims = S["gallery"]["Z_txt"] @ z
        top = np.argsort(-sims)[: max(1, min(k, 50))]
        return {
            "encode_s": round(time.time() - t0, 2),
            "results": [{"caption": S["gallery"]["captions"][i],
                         "score": round(float(sims[i]), 4)} for i in top],
        }

    @app.post("/search")
    def search(req: SearchReq):
        _check_prompt(req.text)
        t0 = time.time()
        v = encode_text_local([req.text])[0]
        z = _project(v, S["head"]["W_txt"], S["head"]["b_txt"],
                     S["mu_txt_local"])
        sims = S["gallery"]["Z_img"] @ z
        top = np.argsort(-sims)[: max(1, min(req.k, 50))]
        return {
            "encode_s": round(time.time() - t0, 2),
            "results": [{"file": S["gallery"]["files"][i],
                         "score": round(float(sims[i]), 4)} for i in top],
        }

    @app.get("/bench")
    def bench():
        r = generate_text(
            "List five uses for a linear map in one line each.", 128)
        t0 = time.time()
        encode_text_local(S["gallery"]["texts_for_recal"][:8])
        r["text_encode_s_per_item"] = round((time.time() - t0) / 8, 3)
        r.pop("text", None)
        return r

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_app(), host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8765")))
