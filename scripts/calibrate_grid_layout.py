"""Determine the 16x16 grid layout inside gemma-4's 260 image tokens.

Encodes two synthetic images (left-black/right-white, top-black/bottom-white)
and, for candidate (offset, transpose) layouts, measures how well the
reshaped grid separates the two halves. Prints the winning layout.
"""
import torch
from PIL import Image

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
    return h[mask].float().cpu()          # [260, d]

W = H = 448
lr = Image.new("RGB", (W, H)); lr.paste((255,255,255), (W//2, 0, W, H))
tb = Image.new("RGB", (W, H)); tb.paste((255,255,255), (0, H//2, W, H))
toks_lr, toks_tb = encode(lr), encode(tb)
print("n_img_tokens:", toks_lr.shape[0])

def sep(grid, axis):
    # grid [16,16,d]; axis=1 -> left/right halves, axis=0 -> top/bottom
    if axis == 1:
        a, b = grid[:, :8].mean((0,1)), grid[:, 8:].mean((0,1))
    else:
        a, b = grid[:8].mean((0,1)), grid[8:].mean((0,1))
    return (a - b).norm().item()

best = None
for off in range(0, 5):
    if off + 256 > toks_lr.shape[0]:
        break
    for tr in (False, True):
        g_lr = toks_lr[off:off+256].reshape(16, 16, -1)
        g_tb = toks_tb[off:off+256].reshape(16, 16, -1)
        if tr:
            g_lr, g_tb = g_lr.transpose(0,1), g_tb.transpose(0,1)
        # correct layout: lr image separates on axis1, tb on axis0,
        # and cross-terms are small
        s = sep(g_lr,1) + sep(g_tb,0) - sep(g_lr,0) - sep(g_tb,1)
        print(f"off={off} transpose={tr}: score={s:8.2f} "
              f"[lr/ax1={sep(g_lr,1):.1f} lr/ax0={sep(g_lr,0):.1f} "
              f"tb/ax0={sep(g_tb,0):.1f} tb/ax1={sep(g_tb,1):.1f}]")
        if best is None or s > best[0]:
            best = (s, off, tr)
print(f"BEST layout: offset={best[1]} transpose={best[2]} score={best[0]:.2f}")
