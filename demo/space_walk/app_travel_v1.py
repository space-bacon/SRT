"""Walk the space — a 27B's understanding as terrain you can cross.

Three models' worth of work meet in one 1024-dimensional room. A Qwen3.8-27B
read 123,287 photographs into it, months ago, and is not here. A 2.1 MB linear
head puts your sentences into the same room. A frozen Qwen3-0.6B, which has
never seen a photograph, says what is at any point in it.

That last part is what makes the room a place rather than a lookup table. You
can stand where a photograph is, or where your own sentence is, or halfway
between two photographs where nothing is, and hear what is there.

Nothing large runs. The 27B's reading is a finished file of about a kilobyte
per image; everything on this page is that file plus a 382 MB reader.
"""
from __future__ import annotations

import struct
from functools import lru_cache

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

READER_REPO = "RiverRider/srt-verbalizer-v1"
READER_FILE = "browser_gallery_1024.pt"
HEAD_REPO = "RiverRider/srt-browser-head-118k"
HEAD_FILE = "head_v3.safetensors"
INDEX_FILE = "gallery_123k_v3.srtidx"
BACKBONE = "Qwen/Qwen3-0.6B"
TAP_LAYER = 28
COCO = "https://s3.amazonaws.com/images.cocodataset.org"


class Prefix(torch.nn.Module):
    """One point in the space -> soft tokens the reader can speak from."""

    def __init__(self, d_in: int, d_model: int, n_tok: int, hidden: int = 2048):
        super().__init__()
        self.n_tok, self.d_model = n_tok, d_model
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, n_tok * d_model))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.net(v).view(v.size(0), self.n_tok, self.d_model)


def load_index(path: str):
    """SRTIDX02: int8 rows with a per-row scale, then the keys."""
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
    return z.astype(np.float16), keys       # fp16: 123,287 x 1024 is 252 MB


print("loading the room...", flush=True)
_tok = AutoTokenizer.from_pretrained(BACKBONE)
_lm = AutoModelForCausalLM.from_pretrained(BACKBONE, dtype=torch.float32).eval()
_ck = torch.load(hf_hub_download(READER_REPO, READER_FILE),
                 map_location="cpu", weights_only=False)
_pre = Prefix(_ck["d_in"], _ck["d_model"], _ck["n_tok"], _ck.get("hidden", 2048))
_pre.load_state_dict(_ck["prefix"])
_pre.eval()
_h = load_file(hf_hub_download(HEAD_REPO, HEAD_FILE))
_W = _h["txt.weight"].float().numpy()
_b = _h["txt.bias"].float().numpy()
_MU = _h["mu_txt"].float().numpy()
_GAL, _KEYS = load_index(hf_hub_download(HEAD_REPO, INDEX_FILE))
print(f"ready: {len(_KEYS):,} photographs, {_ck['d_in']}-d room", flush=True)


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


@lru_cache(maxsize=512)
@torch.no_grad()
def where(text: str) -> tuple:
    """Put a sentence in the room. Returns the point as a tuple so it caches."""
    enc = _tok([text], return_tensors="pt", truncation=True, max_length=64)
    hs = _lm(**enc, output_hidden_states=True).hidden_states[TAP_LAYER].float()
    m = enc["attention_mask"].unsqueeze(-1).float()
    pooled = ((hs * m).sum(1) / m.sum(1)).numpy()[0]
    return tuple(unit((pooled - _MU) @ _W.T + _b).tolist())


@torch.no_grad()
def say(v: np.ndarray, max_new: int = 36) -> str:
    """What is at this point. The reader has never seen a photograph."""
    x = torch.tensor((unit(v) - _ck["mu"]) / _ck["sd"],
                     dtype=torch.float32).unsqueeze(0)
    soft = _pre(x)
    out = _lm.generate(inputs_embeds=soft, max_new_tokens=max_new,
                       do_sample=False, pad_token_id=_tok.eos_token_id,
                       attention_mask=torch.ones(soft.shape[:2], dtype=torch.long))
    t = _tok.decode(out[0], skip_special_tokens=True).strip()
    # This reader was trained on captions carrying no stop token, so it runs to
    # the budget and restates its last clause. Keep whole sentences and drop
    # exact repeats; nothing is reordered, rewritten or invented.
    seen, kept = set(), []
    for s in (p.strip() for p in t.split(".")):
        if not s:
            continue
        low = s.lower()
        if low in seen:
            continue
        seen.add(low)
        kept.append(s)
    if not kept:
        return t
    # A trailing fragment is whatever the budget cut off mid-clause.
    if not t.rstrip().endswith(".") and len(kept) > 1:
        kept = kept[:-1]
    return ". ".join(kept) + "."


def nearby(v: np.ndarray, k: int = 8) -> list:
    sims = _GAL @ unit(v).astype(np.float16)
    top = np.argpartition(-sims, k)[:k]
    top = top[np.argsort(-sims[top])]
    return [(f"{COCO}/{_KEYS[i]}", f"{sims[i]:.3f}") for i in top]


def travel(start: str, dest: str, t: float, k: int):
    if not start.strip():
        return "Type somewhere to start.", []
    a = np.array(where(start.strip()), dtype=np.float32)
    if dest.strip():
        b = np.array(where(dest.strip()), dtype=np.float32)
        v = (1 - t) * a + t * b
    else:
        v = a
    return say(v), nearby(v, k)


def journey(start: str, dest: str, k: int):
    """The whole crossing at once, which reads better than any single point."""
    if not (start.strip() and dest.strip()):
        return "Give me two places and I will walk between them."
    a = np.array(where(start.strip()), dtype=np.float32)
    b = np.array(where(dest.strip()), dtype=np.float32)
    lines = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        lines.append(f"**{t:.2f}** — {say((1 - t) * a + t * b)}")
    return "\n\n".join(lines)


def axis_walk(base: str, plus: str, minus: str, strength: float, k: int):
    if not base.strip():
        return "Type a place to stand.", []
    v = np.array(where(base.strip()), dtype=np.float32)
    p = [w.strip() for w in plus.split(",") if w.strip()]
    m = [w.strip() for w in minus.split(",") if w.strip()]
    if p and m:
        ax = np.mean([np.array(where(w)) for w in p], 0) - \
             np.mean([np.array(where(w)) for w in m], 0)
        v = v + strength * unit(ax)
    return say(v), nearby(v, k)


CSS = """
.sentence textarea, .sentence input { font-size: 20px !important; line-height: 1.5 !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="Walk the space", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Walk the space\n"
        "A **Qwen3.8-27B** read 123,287 photographs into a 1024-dimensional room "
        "and then stopped running. A **2.1 MB** linear head puts your sentences "
        "into the same room. A frozen **Qwen3-0.6B**, which has never seen a "
        "photograph, says what is at any point in it.\n\n"
        "So you can stand where a photograph is, where your own words are, or "
        "halfway between two photographs where nothing is at all, and hear what "
        "is there."
    )

    with gr.Tab("Travel"):
        gr.Markdown(
            "Two places, and the ground between them. At `0.00` you are standing "
            "on your first sentence; at `1.00` on the second. **Everything in "
            "between is a scene that does not exist**, and the reader describes "
            "it anyway."
        )
        with gr.Row():
            start = gr.Textbox(label="from", value="a dog asleep in the afternoon sun")
            dest = gr.Textbox(label="to", value="a red bus crossing a bridge in the rain")
        t = gr.Slider(0, 1, value=0.5, step=0.05, label="how far across")
        said = gr.Textbox(label="what is here", lines=3, elem_classes="sentence")
        shots = gr.Gallery(label="the nearest photographs", columns=4, height=260)
        with gr.Row():
            go = gr.Button("Read this point", variant="primary")
            whole = gr.Button("Walk the whole way")
        trip = gr.Markdown()

        for ev in (go.click, t.release, start.submit, dest.submit):
            ev(travel, [start, dest, t, gr.State(8)], [said, shots])
        whole.click(journey, [start, dest, gr.State(8)], trip)

    with gr.Tab("Arithmetic"):
        gr.Markdown(
            "An axis is the difference between two groups of words. Add it to "
            "where you are standing and you move somewhere else. The reader tells "
            "you where you ended up."
        )
        base = gr.Textbox(label="stand here", value="something moving fast down a street")
        with gr.Row():
            plus = gr.Textbox(label="toward (comma separated)", value="a dog, a horse, a bear, a zebra")
            minus = gr.Textbox(label="away from", value="a car, a bus, a train, a motorcycle")
        strength = gr.Slider(-1.5, 1.5, value=1.0, step=0.1, label="how far along the axis")
        asaid = gr.Textbox(label="what is here", lines=3, elem_classes="sentence")
        ashots = gr.Gallery(label="the nearest photographs", columns=4, height=260)
        ago = gr.Button("Read this point", variant="primary")
        for ev in (ago.click, strength.release, base.submit):
            ev(axis_walk, [base, plus, minus, strength, gr.State(8)], [asaid, ashots])

    gr.Markdown(
        "---\n"
        "**What is actually happening.** Your sentence is read once by the 0.6B, "
        "and a 2.1 MB matrix turns that hidden state into a point. Moving is "
        "arithmetic on 1024 numbers. Reading a point back is the same 0.6B, "
        "driven by a 36 MB adapter that was trained to turn a point into a "
        "sentence and has no vision path at all.\n\n"
        "**The honest part.** The reader is fitted to the region of the room "
        "where photographs live, so points far outside it drift toward generic "
        "scenes rather than failing loudly. Measured on 5,000 held-out "
        "photographs, a sentence it writes retrieves the photograph it was "
        "written from at median rank 18 of 123,287, and the same reader handed "
        "another photograph's point scores at chance. "
        "[Checkpoints and the full evaluation](https://huggingface.co/RiverRider/srt-verbalizer-v1) · "
        "[repository](https://github.com/space-bacon/SRT)"
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
