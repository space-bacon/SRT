"""Verify gemma-4 LM image tokens form a row-major (rows, cols) grid with
rows*cols = n_tokens and cols/rows ~ aspect ratio. Mirror test: h-flip must
reverse columns, v-flip must reverse rows.
"""
import sys
import torch
from PIL import Image, ImageOps

BACKBONE = "google/gemma-4-31B-it"
LAYER = 47

from transformers import AutoModelForImageTextToText, AutoProcessor
proc = AutoProcessor.from_pretrained(BACKBONE)
model = AutoModelForImageTextToText.from_pretrained(
    BACKBONE, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
img_tok = model.config.image_token_id

def encode(img):
    messages = [[{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Describe this image."},
    ]}]]
    enc = proc.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[LAYER][0]
    mask = enc["input_ids"][0] == img_tok
    return h[mask].float().cpu()

def infer_dims(n, w, h):
    best = None
    for r in range(1, n + 1):
        if n % r:
            continue
        c = n // r
        err = abs(c / r - w / h)
        if best is None or err < best[0]:
            best = (err, r, c)
    return best[1], best[2]

def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]), dim=-1
    ).mean().item()

for path in sys.argv[1:] or ["train2017/000000000009.jpg"]:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    t0, th, tv = encode(img), encode(ImageOps.mirror(img)), encode(ImageOps.flip(img))
    n = t0.shape[0]
    r, c = infer_dims(n, w, h)
    g0 = t0.reshape(r, c, -1)
    gh = th.reshape(r, c, -1)
    gv = tv.reshape(r, c, -1)
    s_h, s_h0 = cos(g0, gh.flip(1)), cos(g0, gh)
    s_v, s_v0 = cos(g0, gv.flip(0)), cos(g0, gv)
    print(f"{path} size={w}x{h} n={n} grid=({r},{c}) "
          f"h-flip {s_h:.4f} vs ctrl {s_h0:.4f} gain {s_h-s_h0:+.4f} | "
          f"v-flip {s_v:.4f} vs ctrl {s_v0:.4f} gain {s_v-s_v0:+.4f}",
          flush=True)
