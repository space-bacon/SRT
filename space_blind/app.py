import os
import re

import gradio as gr
import numpy as np
import spaces
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

HOST = "RiverRider/gemma-4-31B-it-nf4"
READER = "Qwen/Qwen3-0.6B"
LAYER = 47
HERE = os.path.dirname(os.path.abspath(__file__))
GAL_DIR = os.path.join(HERE, "gallery")
EX_DIR = os.path.join(HERE, "examples")


class Prefix(torch.nn.Module):
    def __init__(self, d_in, d_model, n_tok, hidden=2048):
        super().__init__()
        self.n_tok, self.d_model = n_tok, d_model
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, n_tok * d_model))

    def forward(self, v):
        return self.net(v).view(v.size(0), self.n_tok, self.d_model)


# ZeroGPU wants everything placed on cuda at import time, not inside the
# decorated function.
proc = AutoProcessor.from_pretrained(HOST)
host = AutoModelForCausalLM.from_pretrained(HOST, device_map="cuda").eval()
IMG_TOK = host.config.image_token_id
N_LAYERS = getattr(host.config, "num_hidden_layers", None) or \
    host.config.text_config.num_hidden_layers

ck = torch.load(hf_hub_download("RiverRider/srt-verbalizer-v1",
                                "gemma4_31b_L47.pt"),
                map_location="cpu", weights_only=False)
MU = torch.tensor(ck["mu"]).float().flatten()
SD = torch.tensor(ck["sd"]).float().flatten()
pre = Prefix(ck["d_in"], ck["d_model"], ck["n_tok"], ck.get("hidden", 2048))
pre.load_state_dict(ck["prefix"])
pre = pre.eval().to("cuda")

tok = AutoTokenizer.from_pretrained(READER)
reader = AutoModelForCausalLM.from_pretrained(
    READER, dtype=torch.float32).eval().to("cuda")

head = torch.load(hf_hub_download("RiverRider/srt-browser-head-118k",
                                  "tap_layer/head_full_L47.pt"),
                  map_location="cpu", weights_only=False)
WI = head["img"]["weight"].float()
BI = head["img"]["bias"].float()
MU_IMG = head["mu_img"].float().flatten()

z = np.load(os.path.join(HERE, "gallery.npz"), allow_pickle=True)
FILES, GAL = list(z["files"]), torch.tensor(z["vecs"])


def dedupe(text):
    """Greedy decoding falls into sentence loops. Keep the first occurrence of
    each sentence and drop a truncated trailing fragment."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    kept, seen = [], set()
    for p in parts:
        key = p.lower().rstrip(".!?")
        if key in seen:
            continue
        seen.add(key)
        kept.append(p)
    if kept and not kept[-1].endswith((".", "!", "?")):
        kept.pop()
    return " ".join(kept) if kept else text


@torch.no_grad()
def say(v):
    soft = pre(((v - MU) / SD).unsqueeze(0).to("cuda"))
    g = reader.generate(
        inputs_embeds=soft,
        attention_mask=torch.ones(soft.shape[:2], dtype=torch.long,
                                  device="cuda"),
        max_new_tokens=40, do_sample=False)
    return dedupe(tok.decode(g[0], skip_special_tokens=True).strip())


@spaces.GPU(duration=60)
@torch.no_grad()
def run(image, k=5):
    if image is None:
        return "", "", "", None
    im = image.convert("RGB")
    msg = [{"role": "user", "content": [
        {"type": "image", "image": im},
        {"type": "text", "text": "Describe this image."}]}]
    enc = proc.apply_chat_template(
        msg, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to("cuda")
    out = host.generate(**enc, max_new_tokens=48, do_sample=False,
                        output_hidden_states=True, return_dict_in_generate=True)
    # hidden_states[0] is the prefill, so this vector predates the first token.
    prefill = out.hidden_states[0][LAYER][0].float()
    v = prefill[enc["input_ids"][0] == IMG_TOK].mean(0).cpu()
    said = dedupe(proc.decode(out.sequences[0][enc["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip())

    read = say(v)
    # Same values, same norm, wrong arrangement.
    control = say(v[torch.randperm(v.numel())])

    q = (v - MU_IMG) @ WI.T + BI
    q = q / q.norm()
    sims = GAL @ q
    order = [i for i in torch.argsort(-sims).tolist() if sims[i] < 0.999][:k]
    shots = [(os.path.join(GAL_DIR, FILES[i]), f"{sims[i]:.3f}") for i in order]

    seeing = (f"### gemma-4-31B, which received the photograph\n"
              f"31 billion parameters and a vision encoder. This is its own "
              f"output.\n\n> {said}")
    blind = (f"### Qwen3-0.6B, which received only the vector\n"
             f"A text-only model. Its entire input is one 5376-dimensional "
             f"vector taken out of gemma's stack before gemma had generated a "
             f"token.\n\n"
             f"> {read}\n\n"
             f"*Shuffle that vector's dimensions and the same reader produces:* "
             f"{control}")
    panel = (
        f"| | |\n|---|---|\n"
        f"| layer read | {LAYER} of {N_LAYERS}, {v.numel()}-d, pooled over the "
        f"image tokens |\n"
        f"| read at | the prefill, before the first generated token |\n"
        f"| state norm | {v.norm():.1f} |\n"
        f"| distance from the reader's training mean | {(v-MU).norm():.1f}, "
        f"against a mean sd of {SD.mean():.1f} |\n"
        f"| nearest of {len(FILES)} gallery images | {sims[order[0]]:.3f} "
        f"against {sims.mean():.3f} for the average one |\n\n"
        f"Nothing is fine-tuned. gemma-4 is frozen and was never trained to be "
        f"read. The reader is a 0.6B decoder trained on COCO captions to turn "
        f"that vector into English, and the gallery search is a single linear "
        f"map applied to the same vector. The host here is 4-bit NF4 so it fits "
        f"free hardware, which keeps retrieval intact but changes the reader's "
        f"wording. See the model card for the measurements."
    )
    return seeing, blind, panel, shots


with gr.Blocks(title="A 0.6B model decodes a 31B model's hidden state") as demo:
    gr.Markdown(
        "# A 0.6B model decodes a 31B model's hidden state\n"
        "Give a photograph to **gemma-4-31B**. At layer 47 of 60, during the "
        "prefill and before a single token has been generated, take one "
        "5376-dimensional vector, pooled over the image token positions.\n\n"
        "**Qwen3-0.6B** receives that vector and nothing else. It is a "
        "text-only model, and those 5376 numbers are everything it gets about "
        "your photograph. It produces a description of what is in the frame. "
        "The same vector passes through one linear map and retrieves from a "
        "gallery of 1000 photographs. The host is frozen and was trained for "
        "neither job.\n\n"
        "Then the control: shuffle that vector's 5376 dimensions and pass the "
        "reader identical values with an identical norm. The description "
        "collapses. Arrangement is what carries the content."
    )
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(label="Give it a picture", height=300, type="pil")
            go = gr.Button("Read the state", variant="primary", size="lg")
            # Deliberately outside the 1000 indexed images, so the first hit is
            # a real neighbour rather than the picture retrieving itself.
            gr.Examples([[os.path.join(EX_DIR, f)]
                         for f in sorted(os.listdir(EX_DIR))], inputs=inp)
        with gr.Column(scale=1):
            a = gr.Markdown()
        with gr.Column(scale=1):
            b = gr.Markdown()
    shots = gr.Gallery(label="What the same vector retrieves from a gallery of "
                             "1000 photographs",
                       columns=5, height=200, object_fit="cover")
    panel = gr.Markdown()
    go.click(run, inp, [a, b, panel, shots])
    # Set the default inside a request, otherwise Gradio builds an absolute
    # http://0.0.0.0:7860 URL that the HTTPS page blocks as mixed content.
    demo.load(lambda: Image.open(os.path.join(EX_DIR, "01_king.jpg")), None, inp)

demo.queue(default_concurrency_limit=2).launch()
