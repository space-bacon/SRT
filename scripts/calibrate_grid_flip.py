"""Locate the 16x16 grid inside gemma-4's COCO image tokens (n=260).

Encodes one real COCO image and its horizontal + vertical mirrors. Under
the correct (offset, row-major/transposed) layout, h-flip reverses grid
columns and v-flip reverses rows. Scores every candidate by cosine of
grid[r,c] vs flipped[r, 15-c] (resp. flipped[15-r, c]).
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

path = sys.argv[1] if len(sys.argv) > 1 else "val2017/000000000139.jpg"
img = Image.open(path).convert("RGB")
t0 = encode(img)
th = encode(ImageOps.mirror(img))
tv = encode(ImageOps.flip(img))
n = t0.shape[0]
print("n_img_tokens:", n)

def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]), dim=-1
    ).mean().item()

best = None
for off in range(0, n - 256 + 1):
    for tr in (False, True):
        g0 = t0[off:off+256].reshape(16, 16, -1)
        gh = th[off:off+256].reshape(16, 16, -1)
        gv = tv[off:off+256].reshape(16, 16, -1)
        if tr:
            g0, gh, gv = (g.transpose(0, 1) for g in (g0, gh, gv))
        s_h = cos(g0, gh.flip(1))       # h-mirror reverses columns
        s_v = cos(g0, gv.flip(0))       # v-mirror reverses rows
        s_h0 = cos(g0, gh)              # control: no flip
        s_v0 = cos(g0, gv)
        score = (s_h - s_h0) + (s_v - s_v0)
        if best is None or score > best[0]:
            best = (score, off, tr, s_h, s_h0, s_v, s_v0)
        if off < 5 or score > 0.02:
            print(f"off={off} tr={tr}: flip_gain={score:+.4f} "
                  f"(h {s_h:.4f} vs {s_h0:.4f}, v {s_v:.4f} vs {s_v0:.4f})")
print(f"BEST: offset={best[1]} transpose={best[2]} flip_gain={best[0]:+.4f}")
