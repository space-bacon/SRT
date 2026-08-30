"""A 0.6B model decodes a 31B model's hidden state, and you can take the state apart.

The previous version of this Space loaded a 4-bit gemma-4-31B on ZeroGPU, ran it
to produce one 5376-dimensional vector, and then did the interesting part, a
0.6B reading that vector, in milliseconds. It spent its whole budget on the
boring half and charged for it twice: a queue on every visit, and a quantisation
that changed the reader's wording.

The 31B's contribution is now precomputed at full precision, so this runs on a
CPU with no queue, and the budget goes where the interest is. That buys two
things that were impossible before, both of which are controls you can turn:

  corrupt   scramble a chosen fraction of the 5376 dimensions and watch the
            description dissolve. The old version had this as a single yes/no
            aside. Continuous, it is the main exhibit: identical values,
            identical norm, only the arrangement destroyed.

  walk      move the state from one photograph towards another and read every
            point on the way, including the ones that are not any photograph.

Nothing is fine-tuned. gemma-4 is frozen and was never trained to be read.
"""
import json
import os

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

READER = "Qwen/Qwen3-0.6B"
HERE = os.path.dirname(os.path.abspath(__file__))
GAL_DIR = os.path.join(HERE, "gallery")
EX_DIR = os.path.join(HERE, "examples")
TRAINED_LAYER = 47

BG, PANEL, VIOLET = "#131233", "#1c1a47", "#a48dff"
INK, MUTED, LINE = "#eae7fb", "#a7a1cc", "#332e66"
CSS = f"""
.gradio-container {{background:{BG} !important; color:{INK} !important}}
.srt-panel {{background:{PANEL}; border:1px solid {LINE}; border-radius:10px;
             padding:14px 16px; margin-top:6px}}
.srt-panel h3 {{color:{VIOLET}; margin:0 0 6px 0; font-size:15px}}
.srt-said {{color:{INK}; font-size:15px; line-height:1.5}}
.srt-muted {{color:{MUTED}; font-size:13px}}
"""


class Prefix(torch.nn.Module):
    def __init__(self, d_in, d_model, n_tok, hidden=2048):
        super().__init__()
        self.n_tok, self.d_model = n_tok, d_model
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, n_tok * d_model))

    def forward(self, v):
        return self.net(v).view(v.size(0), self.n_tok, self.d_model)


print("loading...", flush=True)
ck = torch.load(hf_hub_download("RiverRider/srt-verbalizer-v1",
                                "gemma4_31b_L47.pt"),
                map_location="cpu", weights_only=False)
MU = torch.tensor(ck["mu"]).float().flatten()
SD = torch.tensor(ck["sd"]).float().flatten()
pre = Prefix(ck["d_in"], ck["d_model"], ck["n_tok"], ck.get("hidden", 2048)).eval()
pre.load_state_dict(ck["prefix"])

tok = AutoTokenizer.from_pretrained(READER)
reader = AutoModelForCausalLM.from_pretrained(READER, dtype=torch.float32).eval()

head = torch.load(hf_hub_download("RiverRider/srt-browser-head-118k",
                                  "tap_layer/head_full_L47.pt"),
                  map_location="cpu", weights_only=False)
WI, BI = head["img"]["weight"].float(), head["img"]["bias"].float()
MU_IMG = head["mu_img"].float().flatten()

g = np.load(os.path.join(HERE, "gallery.npz"), allow_pickle=True)
FILES, GAL = list(g["files"]), torch.tensor(g["vecs"])

s = np.load(os.path.join(HERE, "decode_states.npz"), allow_pickle=True)
STATES = torch.tensor(s["states"].astype(np.float32))     # (N, n_layers, 5376)
LAYERS = [int(x) for x in s["layers"]]
SFILES = [str(x) for x in s["files"]]
SSETS = [str(x) for x in s["sets"]]
CAPS = json.load(open(os.path.join(HERE, "decode_states_captions.json")))
IDX = {f: i for i, f in enumerate(SFILES)}
LJ = LAYERS.index(TRAINED_LAYER)
PICKS = [f for f, st in zip(SFILES, SSETS) if st == "examples"] + \
        [f for f, st in zip(SFILES, SSETS) if st == "gallery"][:120]
print(f"ready: {len(SFILES)} states, layers {LAYERS}", flush=True)


def path_of(f):
    p = os.path.join(EX_DIR, f)
    return p if os.path.exists(p) else os.path.join(GAL_DIR, f)


def dedupe(text):
    import re
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    kept, seen = [], set()
    for p in parts:
        k = p.lower().rstrip(".!?")
        if k in seen:
            continue
        seen.add(k)
        kept.append(p)
    if kept and not kept[-1].endswith((".", "!", "?")):
        kept.pop()
    return " ".join(kept) if kept else text


@torch.no_grad()
def say(v):
    soft = pre(((v - MU) / SD).unsqueeze(0))
    out = reader.generate(inputs_embeds=soft,
                          attention_mask=torch.ones(soft.shape[:2], dtype=torch.long),
                          max_new_tokens=40, do_sample=False)
    return dedupe(tok.decode(out[0], skip_special_tokens=True).strip())


def scramble(v, frac, seed=0):
    """Permute a fraction of the dimensions among themselves.

    Values and norm are exactly preserved at every setting; only arrangement is
    destroyed, so the reader cannot be losing magnitude information.
    """
    if frac <= 0:
        return v.clone()
    rng = np.random.default_rng(seed)
    d = v.numel()
    k = max(2, int(round(frac * d)))
    pick = rng.choice(d, k, replace=False)
    out = v.clone()
    out[pick] = v[pick][torch.tensor(rng.permutation(k))]
    return out


@torch.no_grad()
def retrieve(v, k=5):
    q = (v - MU_IMG) @ WI.T + BI
    q = q / q.norm()
    sims = GAL @ q
    order = [i for i in torch.argsort(-sims).tolist() if sims[i] < 0.999][:k]
    return [(os.path.join(GAL_DIR, FILES[i]), f"{sims[i]:.3f}") for i in order],\
           float(sims[order[0]]), float(sims.mean())


def panel(title, body, note=""):
    n = f"<div class='srt-muted'>{note}</div>" if note else ""
    return (f"<div class='srt-panel'><h3>{title}</h3>"
            f"<div class='srt-said'>{body}</div>{n}</div>")


def read_one(fname, layer):
    if not fname:
        return "", "", "", None
    v = STATES[IDX[fname], LAYERS.index(int(layer))]
    said = say(v)
    shots, top, avg = retrieve(v)
    host = panel("gemma-4-31B, which received the photograph",
                 CAPS.get(fname, "(not cached)"),
                 "31 billion parameters and a vision encoder. Its own output.")
    rd = panel("Qwen3-0.6B, which received only the vector", said,
               f"A text-only model. Its entire input is one {v.numel()}-d vector "
               f"taken from gemma's stack at layer {int(layer)}, during the "
               f"prefill, before gemma had generated a token.")
    extra = ("" if int(layer) == TRAINED_LAYER else
             f"<br><b>Layer {int(layer)} is not where this reader was trained.</b> "
             f"The verbalizer was fitted at layer {TRAINED_LAYER}, so what you are "
             f"seeing is how far layer {int(layer)} sits from layer "
             f"{TRAINED_LAYER} in the reader's frame. It is not a measurement of "
             f"where meaning lives.")
    stats = panel("the state", 
                  f"norm {v.norm():.1f} &nbsp;·&nbsp; "
                  f"{(v - MU).norm():.1f} from the reader's training mean "
                  f"(mean sd {SD.mean():.1f}) &nbsp;·&nbsp; "
                  f"nearest of {len(FILES)} gallery images {top:.3f}, "
                  f"against {avg:.3f} for the average one", extra)
    return host, rd, stats, shots


def corrupt_one(fname, layer, pct):
    if not fname:
        return ""
    v = STATES[IDX[fname], LAYERS.index(int(layer))]
    w = scramble(v, pct / 100.0)
    return panel(f"{int(pct)}% of the 5376 dimensions scrambled", say(w),
                 f"norm {w.norm():.1f}, unchanged. Same values, same magnitude, "
                 f"only the arrangement disturbed.")


def corrupt_ladder(fname, layer):
    if not fname:
        return ""
    v = STATES[IDX[fname], LAYERS.index(int(layer))]
    rows = "".join(
        f"<tr><td style='padding:4px 12px 4px 0;color:{VIOLET}'>{p:>3}%</td>"
        f"<td style='padding:4px 0'>{say(scramble(v, p / 100.0))}</td></tr>"
        for p in (0, 10, 25, 50, 75, 100))
    return panel("the same state, dissolved by degrees",
                 f"<table style='width:100%'>{rows}</table>",
                 "Every row has the identical set of 5376 values and the "
                 "identical norm. Only how they are arranged differs.")


def walk(fa, fb, layer, alpha):
    if not fa or not fb:
        return ""
    j = LAYERS.index(int(layer))
    v = (1 - alpha) * STATES[IDX[fa], j] + alpha * STATES[IDX[fb], j]
    return panel(f"{int((1 - alpha) * 100)}% first, {int(alpha * 100)}% second",
                 say(v), f"norm {v.norm():.1f}. At any setting other than 0 or "
                         f"100 this is not a photograph anyone took.")


def walk_all(fa, fb, layer):
    if not fa or not fb:
        return ""
    j = LAYERS.index(int(layer))
    rows = ""
    for al in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = (1 - al) * STATES[IDX[fa], j] + al * STATES[IDX[fb], j]
        rows += (f"<tr><td style='padding:4px 12px 4px 0;color:{VIOLET}'>"
                 f"{int(al * 100):>3}%</td>"
                 f"<td style='padding:4px 0'>{say(v)}</td></tr>")
    return panel("the whole walk", f"<table style='width:100%'>{rows}</table>",
                 "The middle rows are states no photograph produced.")


with gr.Blocks(title="A 0.6B model decodes a 31B model's hidden state",
               css=CSS) as demo:
    gr.Markdown(
        "# A 0.6B model decodes a 31B model's hidden state\n"
        "**gemma-4-31B** was given a photograph. At layer 47 of 60, during the "
        "prefill and before a single token had been generated, one "
        "5376-dimensional vector was taken, pooled over the image token "
        "positions. **Qwen3-0.6B**, a text-only model that has never seen a "
        "photograph, receives that vector and nothing else.\n\n"
        "gemma's side is precomputed at full precision, so nothing here waits "
        "for a GPU and you can take the state apart as much as you like.")
    layer = gr.Dropdown([str(l) for l in LAYERS], value=str(TRAINED_LAYER),
                        label="layer of gemma's 60 to read")

    with gr.Tab("Read"):
        with gr.Row():
            with gr.Column(scale=1):
                pick = gr.Dropdown(PICKS, value=PICKS[0], label="photograph")
                shown = gr.Image(height=300, show_label=False)
                go = gr.Button("Read the state", variant="primary", size="lg")
            with gr.Column(scale=2):
                a_host = gr.HTML()
                a_read = gr.HTML()
                a_stat = gr.HTML()
        shots = gr.Gallery(label="what the same vector retrieves from 1000 "
                                 "photographs", columns=5, height=190,
                           object_fit="cover")

    with gr.Tab("Corrupt"):
        gr.Markdown("Scrambling a fraction of the dimensions preserves every "
                    "value and the norm exactly. Only the arrangement is "
                    "destroyed. This was a single yes/no aside in the previous "
                    "version of this demo; it is the most interesting control "
                    "here, so now you can turn it.")
        with gr.Row():
            cpick = gr.Dropdown(PICKS, value=PICKS[0], label="photograph")
            cpct = gr.Slider(0, 100, value=30, step=5,
                             label="% of dimensions scrambled")
        with gr.Row():
            cgo = gr.Button("Read it", variant="primary")
            cladder = gr.Button("Show the whole ladder")
        cout = gr.HTML()

    with gr.Tab("Walk"):
        gr.Markdown("Move the state from one photograph towards another. The "
                    "midpoints are states no camera produced, and the reader "
                    "describes them anyway.")
        with gr.Row():
            wa = gr.Dropdown(PICKS, value=PICKS[0], label="from")
            wb = gr.Dropdown(PICKS, value=PICKS[min(2, len(PICKS) - 1)],
                             label="towards")
        walpha = gr.Slider(0, 1, value=0.5, step=0.05, label="how far across")
        with gr.Row():
            wgo = gr.Button("Read this point", variant="primary")
            wall = gr.Button("Walk the whole way")
        wout = gr.HTML()

    pick.change(lambda f: path_of(f), pick, shown)
    go.click(read_one, [pick, layer], [a_host, a_read, a_stat, shots])
    cgo.click(corrupt_one, [cpick, layer, cpct], cout)
    cpct.release(corrupt_one, [cpick, layer, cpct], cout)
    cladder.click(corrupt_ladder, [cpick, layer], cout)
    wgo.click(walk, [wa, wb, layer, walpha], wout)
    walpha.release(walk, [wa, wb, layer, walpha], wout)
    wall.click(walk_all, [wa, wb, layer], wout)
    demo.load(lambda: path_of(PICKS[0]), None, shown)

demo.queue(default_concurrency_limit=2).launch()
