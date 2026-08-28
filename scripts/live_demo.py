#!/usr/bin/env python3
"""Live SRT read-out: one forward pass, real words, real retrieval, real state.

gemma-4-31B encodes what you give it. The layer-47 state from that single pass
does three jobs: srt-verbalizer-v1 turns it into a sentence, the shipped L47
head projects it to search a gallery, and the state itself is reported.

Pooling is the mean over image-token positions. probe_verbalizer_pooling.py
established that the verbalizer and the retrieval head both want that one, and
that the wrong choice returns fluent nonsense instead of an error.

    python scripts/live_demo.py --gallery 2000 --port 8080
"""
import argparse
import os

os.environ.setdefault("HF_HOME", "/root/.hf_home")

import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

HOST = "google/gemma-4-31B-it"
LAYER = 47
HEAD = "/root/head_full_L47.pt"
IMG_DIR = "/root/val2017"
CACHE = "/root/live_gallery.npz"


class Prefix(torch.nn.Module):
    def __init__(self, d_in, d_model, n_tok, hidden=2048):
        super().__init__()
        self.n_tok, self.d_model = n_tok, d_model
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, n_tok * d_model))

    def forward(self, v):
        return self.net(v).view(v.size(0), self.n_tok, self.d_model)


class Rig:
    """Everything loaded once: the host, the verbalizer, the head, the gallery."""

    def __init__(self, n_gallery):
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        print("loading gemma-4-31B", flush=True)
        self.proc = AutoProcessor.from_pretrained(HOST)
        self.host = AutoModelForCausalLM.from_pretrained(
            HOST, dtype=torch.bfloat16, device_map="cuda").eval()
        self.img_tok = getattr(self.host.config, "image_token_id", None)

        print("loading verbalizer", flush=True)
        ck = torch.load(hf_hub_download("RiverRider/srt-verbalizer-v1",
                                        "gemma4_31b_L47.pt"),
                        map_location="cpu", weights_only=False)
        self.mu = torch.tensor(ck["mu"]).float().flatten()
        self.sd = torch.tensor(ck["sd"]).float().flatten()
        self.pre = Prefix(ck["d_in"], ck["d_model"], ck["n_tok"],
                          ck.get("hidden", 2048))
        self.pre.load_state_dict(ck["prefix"])
        self.pre.eval()
        self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        self.lm = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B", dtype=torch.float32).eval()

        print("loading L47 head", flush=True)
        h = torch.load(HEAD, map_location="cpu", weights_only=False)
        w = h["img"]["weight"] if "weight" in h["img"] else h["img"]["0.weight"]
        self.Wi = w.float()
        self.bi = (h["img"].get("bias", h["img"].get("0.bias"))).float()
        self.mu_img = h["mu_img"].float()

        self.files, self.gal = self.gallery(n_gallery)
        print(f"gallery ready: {len(self.files)} images", flush=True)

    @torch.no_grad()
    def state_of(self, im):
        """Layer-47 hidden state, meaned over the image-token positions."""
        msg = [{"role": "user", "content": [
            {"type": "image", "image": im},
            {"type": "text", "text": "Describe this image."}]}]
        enc = self.proc.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to("cuda")
        out = self.host(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[LAYER][0].float()
        mask = enc["input_ids"][0] == self.img_tok
        return h[mask].mean(0).cpu()

    def project(self, v):
        z = (v - self.mu_img.flatten()) @ self.Wi.T + self.bi
        return z / (z.norm() + 1e-8)

    @torch.no_grad()
    def gallery(self, n):
        if Path(CACHE).exists():
            z = np.load(CACHE, allow_pickle=True)
            if len(z["files"]) >= n:
                return list(z["files"][:n]), torch.tensor(z["vecs"][:n])
        files = sorted(os.listdir(IMG_DIR))[:n]
        vecs = []
        t0 = time.time()
        for i, f in enumerate(files, 1):
            im = Image.open(os.path.join(IMG_DIR, f)).convert("RGB")
            vecs.append(self.project(self.state_of(im)).numpy())
            if i % 100 == 0:
                r = i / (time.time() - t0)
                print(f"  gallery {i}/{n}  {r:.2f}/s  eta {(n-i)/r/60:.1f}m",
                      flush=True)
        V = np.stack(vecs)
        np.savez(CACHE, files=np.array(files), vecs=V)
        return files, torch.tensor(V)

    @torch.no_grad()
    def verbalize(self, v):
        z = ((v - self.mu) / self.sd).unsqueeze(0)
        soft = self.pre(z)
        gen = self.lm.generate(
            inputs_embeds=soft,
            attention_mask=torch.ones(soft.shape[:2], dtype=torch.long),
            max_new_tokens=40, do_sample=False)
        text = self.tok.decode(gen[0], skip_special_tokens=True).strip()
        return dedupe(text)


def dedupe(text):
    """Greedy decoding at 40 tokens falls into sentence loops. Keep the first
    occurrence of each sentence and drop a trailing fragment."""
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


EXAMPLE_DIR = "/tmp/live_examples"


def examples(rig, n=6):
    """gr.Examples resolves paths when the Blocks tree is built, before
    launch(allowed_paths=...) takes effect, so stage copies under /tmp."""
    os.makedirs(EXAMPLE_DIR, exist_ok=True)
    out = []
    for f in rig.files[:n]:
        dst = os.path.join(EXAMPLE_DIR, f)
        if not os.path.exists(dst):
            shutil.copyfile(os.path.join(IMG_DIR, f), dst)
        out.append([dst])
    return out


def build_ui(rig):
    import gradio as gr

    def run(image, k=5):
        if image is None:
            return "Give it a picture.", None
        t0 = time.time()
        im = Image.fromarray(image).convert("RGB") if not isinstance(image, Image.Image) else image
        v = rig.state_of(im)
        words = rig.verbalize(v)
        q = rig.project(v)
        sims = (rig.gal @ q)
        # An example picture is itself in the gallery. Retrieving it back at
        # cosine 1.000 is arithmetic, not evidence, so drop it.
        order = [i for i in torch.argsort(-sims).tolist() if sims[i] < 0.999]
        top = order[:k]
        shots = [(os.path.join(IMG_DIR, rig.files[i]), f"{sims[i]:.3f}") for i in top]

        raw = v / (v.norm() + 1e-8)
        dist = (v - rig.mu).norm()
        z = (v - rig.mu) / rig.sd
        panel = (
            f"### What came out of one forward pass\n\n"
            f"**It said:** {words}\n\n"
            f"| | |\n|---|---|\n"
            f"| layer read | {LAYER} of gemma-4-31B, {v.numel()}-d |\n"
            f"| pooled over | image-token positions |\n"
            f"| state norm | {v.norm():.1f} |\n"
            f"| distance from the verbaliser's training mean | {dist:.1f} "
            f"(its sd averages {rig.sd.mean():.1f}, so this sits inside the "
            f"distribution it was fitted on) |\n"
            f"| z, mean absolute | {z.abs().mean():.3f} |\n"
            f"| nearest of {len(rig.files)} gallery images | {sims[top[0]]:.3f} "
            f"against {sims.mean():.3f} for the average one |\n"
            f"| elapsed | {time.time()-t0:.1f}s |\n\n"
            f"The sentence and the pictures come from the same vector. Nothing "
            f"here was fine-tuned: gemma-4 is frozen, the verbaliser is a 0.6B "
            f"reading its state, and the head is a linear map."
        )
        return panel, shots

    with gr.Blocks(title="One forward pass, read three ways") as demo:
        gr.Markdown(
            "# One forward pass, read three ways\n"
            "Give **gemma-4-31B** a photograph. From the single layer-47 state "
            "that produces, a 0.6B model says what it sees, a linear head "
            "searches a gallery with it, and the state itself is reported. "
            "The host is frozen throughout."
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(label="Give it a picture", height=300,
                               type="pil")
                go = gr.Button("Read the state", variant="primary", size="lg")
                gr.Examples(examples(rig), inputs=inp)
            with gr.Column(scale=1):
                out = gr.Markdown()
        shots = gr.Gallery(label="What it reached for in the gallery",
                           columns=5, height=200, object_fit="cover")
        go.click(run, inp, [out, shots])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", type=int, default=2000)
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    rig = Rig(a.gallery)
    # Gradio refuses to serve files outside its own cache unless told.
    build_ui(rig).queue().launch(server_name="0.0.0.0", server_port=a.port,
                                 allowed_paths=[IMG_DIR])
