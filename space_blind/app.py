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


@spaces.GPU(duration=90)
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

    seeing = (f"### The model with eyes\n**gemma-4-31B**, 31 billion "
              f"parameters, a vision tower, and the picture in front of it.\n\n"
              f"> {said}")
    blind = (f"### The model with no eyes\n**Qwen3-0.6B**, text only. No vision "
             f"tower. It never received the picture. It received one 5376-d "
             f"vector from the middle of gemma's stack, read before gemma had "
             f"said a word.\n\n> {read}\n\n*Shuffle that vector's dimensions "
             f"and the same reader says:* {control}")
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


with gr.Blocks(title="A model with no eyes describes your picture") as demo:
    gr.Markdown(
        "# A model with no eyes describes your picture\n"
        "Give the photograph to **gemma-4-31B**. Halfway up its stack, before "
        "it has generated a single token, take one vector. Hand only that "
        "vector to **Qwen3-0.6B**, a text-only model with no vision tower, "
        "which never sees your picture. Then read both out loud.\n\n"
        "The third line is the control. Shuffle that vector's 5376 dimensions "
        "and the reader gets identical values with an identical norm. If the "
        "description survived that, the vector was not carrying it."
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
    shots = gr.Gallery(label="What the same vector reached for, searching the "
                             "gallery on its own",
                       columns=5, height=200, object_fit="cover")
    panel = gr.Markdown()
    go.click(run, inp, [a, b, panel, shots])

demo.queue(default_concurrency_limit=2).launch()
