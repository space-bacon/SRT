"""The map — a model's understanding, laid out flat and walked across.

A Qwen3.8-27B read 123,287 photographs into a 1024-dimensional room and then
stopped running. This is that room projected to two dimensions, so it can be
looked at, and a frozen Qwen3-0.6B that has never seen a photograph says what
is at any point you choose.

Nothing on the map was labelled by a person. The region names are the reader
saying what is at each centre.
"""
from __future__ import annotations

import json
import struct
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

READER_REPO, READER_FILE = "RiverRider/srt-verbalizer-v1", "browser_gallery_1024.pt"
HEAD_REPO, HEAD_FILE = "RiverRider/srt-browser-head-118k", "head_v3.safetensors"
INDEX_FILE = "gallery_123k_v3.srtidx"
BACKBONE, TAP = "Qwen/Qwen3-0.6B", 28
COCO = "https://s3.amazonaws.com/images.cocodataset.org"

W = H = 760
BG = (19, 18, 51)
DOT = (110, 96, 180)
MARK = (255, 214, 120)
INK = (234, 231, 251)
MUTED = (167, 161, 204)


class Prefix(torch.nn.Module):
    def __init__(self, d_in, d_model, n_tok, hidden=2048):
        super().__init__()
        self.n_tok, self.d_model = n_tok, d_model
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, n_tok * d_model))

    def forward(self, v):
        return self.net(v).view(v.size(0), self.n_tok, self.d_model)


def load_index(path):
    raw = open(path, "rb").read()
    dim, n = struct.unpack_from("<II", raw, 8)
    off = 16
    sc = np.frombuffer(raw, "<f4", n, off); off += 4 * n
    q = np.frombuffer(raw, np.int8, n * dim, off).reshape(n, dim); off += n * dim
    keys = []
    for _ in range(n):
        (ln,) = struct.unpack_from("<I", raw, off); off += 4
        keys.append(raw[off:off + ln].decode()); off += ln
    z = q.astype(np.float32) * sc[:, None]
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
    return z.astype(np.float16), keys


print("loading the room...", flush=True)
_tok = AutoTokenizer.from_pretrained(BACKBONE)
_lm = AutoModelForCausalLM.from_pretrained(BACKBONE, dtype=torch.float32).eval()
_ck = torch.load(hf_hub_download(READER_REPO, READER_FILE), map_location="cpu",
                 weights_only=False)
_pre = Prefix(_ck["d_in"], _ck["d_model"], _ck["n_tok"], _ck.get("hidden", 2048))
_pre.load_state_dict(_ck["prefix"])
_pre.eval()
_h = load_file(hf_hub_download(HEAD_REPO, HEAD_FILE))
_HW = _h["txt.weight"].float().numpy()
_HB = _h["txt.bias"].float().numpy()
_MU = _h["mu_txt"].float().numpy()
_GAL, _KEYS = load_index(hf_hub_download(HEAD_REPO, INDEX_FILE))

_here = Path(__file__).parent
_m = np.load(_here / "space_map.npz")
_XY, _ROWS = _m["xy"].astype(np.float32), _m["rows"]
_REGIONS = json.load(open(_here / "space_map_labels.json"))["regions"]
_MAPPED = _GAL[_ROWS].astype(np.float32)      # the 40k that have a place on the map
print(f"ready: {len(_KEYS):,} photographs, {len(_XY):,} on the map", flush=True)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


# Fit the actual spread of the layout to the canvas rather than assuming it
# fills [-1,1]; UMAP output is never square and the slack shows as dead margin.
_LO = _XY.min(0) - 0.04
_HI = _XY.max(0) + 0.04
_PAD = 26


def to_px(xy):
    """Layout coordinates to pixels, y flipped so up is up."""
    xy = np.asarray(xy, dtype=np.float32)
    f = (xy - _LO) / (_HI - _LO)
    x = _PAD + f[..., 0] * (W - 2 * _PAD)
    y = _PAD + (1 - f[..., 1]) * (H - 2 * _PAD)
    return np.stack([x, y], -1)


def from_px(px, py):
    """Pixels back to layout coordinates, for clicks."""
    fx = (px - _PAD) / (W - 2 * _PAD)
    fy = 1 - (py - _PAD) / (H - 2 * _PAD)
    return (np.array([fx, fy], dtype=np.float32) * (_HI - _LO) + _LO).astype(np.float32)


def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def base_map():
    """Every photograph a point; every region named by the reader."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img, "RGBA")
    for x, y in to_px(_XY):
        d.ellipse([x - 1.1, y - 1.1, x + 1.1, y + 1.1], fill=DOT + (150,))
    f = _font(12)
    for r in sorted(_REGIONS, key=lambda r: -r["n"]):
        cx, cy = to_px(np.array([r["x"], r["y"]]))
        txt = r["text"].rstrip(".")
        if len(txt) > 32:
            txt = txt[:31] + "…"
        w = d.textlength(txt, font=f)
        # Keep the plate inside the frame; edge regions were being clipped.
        bx = min(max(cx - w / 2, 4), W - w - 4)
        by = min(max(cy, 12), H - 12)
        d.rectangle([bx - 5, by - 9, bx + w + 5, by + 9], fill=(28, 26, 71, 215))
        d.text((bx, by - 7), txt, font=f, fill=MUTED)
    return img


_BASE = base_map()


def draw_marker(img, xy, label=None, trail=None):
    im = img.copy()
    d = ImageDraw.Draw(im, "RGBA")
    if trail is not None and len(trail) > 1:
        d.line([tuple(p) for p in to_px(np.asarray(trail))],
               fill=MARK + (120,), width=2)
    x, y = to_px(np.asarray(xy))
    for rad, alpha in ((22, 40), (14, 70), (7, 150)):
        d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=MARK + (alpha,))
    d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=MARK)
    if label:
        f = _font(13)
        w = d.textlength(label, font=f)
        bx = min(max(x - w / 2, 6), W - w - 12)
        d.rectangle([bx - 6, y + 14, bx + w + 6, y + 34], fill=(19, 18, 51, 230))
        d.text((bx, y + 17), label, font=f, fill=INK)
    return im


@lru_cache(maxsize=512)
@torch.no_grad()
def where(text):
    enc = _tok([text], return_tensors="pt", truncation=True, max_length=64)
    hs = _lm(**enc, output_hidden_states=True).hidden_states[TAP].float()
    m = enc["attention_mask"].unsqueeze(-1).float()
    pooled = ((hs * m).sum(1) / m.sum(1)).numpy()[0]
    return tuple(unit((pooled - _MU) @ _HW.T + _HB).tolist())


@torch.no_grad()
def say(v, max_new=36):
    x = torch.tensor((unit(v) - _ck["mu"]) / _ck["sd"],
                     dtype=torch.float32).unsqueeze(0)
    soft = _pre(x)
    out = _lm.generate(inputs_embeds=soft, max_new_tokens=max_new, do_sample=False,
                       pad_token_id=_tok.eos_token_id,
                       attention_mask=torch.ones(soft.shape[:2], dtype=torch.long))
    t = _tok.decode(out[0], skip_special_tokens=True).strip()
    # Trained on captions carrying no stop token, so it restates its last clause.
    # Drop exact repeats and any fragment the budget cut; nothing is invented.
    seen, kept = set(), []
    for s in (p.strip() for p in t.split(".")):
        if s and s.lower() not in seen:
            seen.add(s.lower())
            kept.append(s)
    if len(kept) > 1 and not t.rstrip().endswith("."):
        kept = kept[:-1]
    return ". ".join(kept) + "." if kept else t


def place_of(v, k=64):
    """Where a point sits on the map: the weighted mean of its neighbours."""
    sims = _MAPPED @ unit(v)
    top = np.argpartition(-sims, k)[:k]
    w = np.maximum(sims[top], 0) ** 8          # sharpen, or everything drifts centre
    return (_XY[top] * w[:, None]).sum(0) / (w.sum() + 1e-8)


def vector_at(xy, k=48):
    """What is at a spot: the mean reading of the photographs nearest it."""
    d2 = ((_XY - np.asarray(xy, dtype=np.float32)) ** 2).sum(1)
    top = np.argpartition(d2, k)[:k]
    return _MAPPED[top].mean(0), top


def shots(rows, k=8):
    return [f"{COCO}/{_KEYS[_ROWS[i]]}" for i in rows[:k]]


def go(prompt):
    """Fly to a sentence, then read what is there. Yields frames, so it moves."""
    if not prompt.strip():
        yield _BASE, "Type somewhere to go.", []
        return
    v = np.array(where(prompt.strip()), dtype=np.float32)
    target = place_of(v)
    start = _XY.mean(0)
    trail = []
    for i in range(1, 13):
        t = 1 - (1 - i / 12) ** 3              # ease out, so it lands rather than stops
        p = start + (target - start) * t
        trail.append(p)
        yield draw_marker(_BASE, p, trail=trail), "…", []
    _, near = vector_at(target)
    yield (draw_marker(_BASE, target, label=prompt.strip()[:48], trail=trail),
           say(v), shots(near))


def tap(evt: gr.SelectData):
    """Point at the map and hear what is there."""
    px, py = evt.index
    xy = from_px(px, py)
    v, near = vector_at(xy)
    return draw_marker(_BASE, xy), say(v), shots(near)


def cross(a, b):
    """Walk from one sentence to another, reading the ground on the way."""
    if not (a.strip() and b.strip()):
        yield _BASE, "Give me two places.", []
        return
    va = np.array(where(a.strip()), dtype=np.float32)
    vb = np.array(where(b.strip()), dtype=np.float32)
    lines, trail = [], []
    for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        v = (1 - t) * va + t * vb
        p = place_of(v)
        trail.append(p)
        lines.append(f"{t:.0%} — {say(v)}")
        _, near = vector_at(p)
        yield (draw_marker(_BASE, p, label=f"{t:.0%}", trail=trail),
               "\n".join(lines), shots(near))


CSS = (".said textarea { font-size: 18px !important; line-height: 1.5 !important; }\n"
       "footer { display: none !important; }")

with gr.Blocks(title="A map of what a 27B understood") as demo:
    gr.Markdown(
        "# A map of what a 27B understood\n"
        "Every dot is a photograph, placed by how a **Qwen3.8-27B** read it. That "
        "model ran once, months ago, and is not here: what survives is about a "
        "kilobyte each. A frozen **Qwen3-0.6B** that has never seen a photograph "
        "says what is at any point you choose, and **every region name on the map "
        "was written by it**, not by us.\n\n"
        "**Click anywhere on the map**, or type a sentence and watch where it lands."
    )
    with gr.Row():
        with gr.Column(scale=3):
            canvas = gr.Image(value=_BASE, type="pil", height=H, show_label=False,
                              interactive=False)
        with gr.Column(scale=2):
            said = gr.Textbox(label="what is here", lines=5, elem_classes="said")
            pics = gr.Gallery(label="the nearest photographs", columns=4, height=200)
            with gr.Tab("Go somewhere"):
                prompt = gr.Textbox(label="a sentence",
                                    value="a dog asleep in the afternoon sun")
                fly = gr.Button("Fly there", variant="primary")
            with gr.Tab("Cross the map"):
                p1 = gr.Textbox(label="from", value="a plate of food on a table")
                p2 = gr.Textbox(label="to",
                                value="a man riding skis down a snowy slope")
                walk = gr.Button("Walk it", variant="primary")

    fly.click(go, prompt, [canvas, said, pics])
    prompt.submit(go, prompt, [canvas, said, pics])
    walk.click(cross, [p1, p2], [canvas, said, pics])
    canvas.select(tap, None, [canvas, said, pics])

    gr.Markdown(
        "---\n"
        "**How the map is made.** The room has 1024 dimensions; this is a UMAP "
        "projection of 40,000 of the photographs down to two, so what you see is "
        "neighbourhood structure, not coordinates that mean anything on their own. "
        "Clicking reads the average of the ~48 photographs nearest that spot. A "
        "sentence is placed by where its nearest neighbours already sit.\n\n"
        "**The honest part.** The reader is fitted to the region where photographs "
        "live, so a point far outside it drifts toward a generic scene rather than "
        "failing loudly. On 5,000 held-out photographs a sentence it writes "
        "retrieves the photograph it came from at median rank 18 of 123,287, and "
        "the same reader handed a different photograph's point scores at chance. "
        "[Checkpoints and full evaluation](https://huggingface.co/RiverRider/srt-verbalizer-v1) · "
        "[repository](https://github.com/space-bacon/SRT)"
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(css=CSS, theme=gr.themes.Soft())
