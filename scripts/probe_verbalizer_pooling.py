#!/usr/bin/env python3
"""Find which pooling the verbalizer was trained on, by trying them.

srt-verbalizer-v1 wants a gemma-4-31B layer-47 state, but the card does not say
how that state was pooled. Feeding it the wrong pooling produces fluent nonsense
rather than an error, so this compares candidates against the checkpoint's own
training mean and reads what each one generates.
"""
import os

os.environ["HF_HOME"] = "/root/.hf_home"

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

HOST = "google/gemma-4-31B-it"
LAYER = 47
IMG = "/root/val2017/000000039769.jpg"


class Prefix(torch.nn.Module):
    def __init__(self, d_in, d_model, n_tok, hidden=2048):
        super().__init__()
        self.n_tok, self.d_model = n_tok, d_model
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, n_tok * d_model))

    def forward(self, v):
        return self.net(v).view(v.size(0), self.n_tok, self.d_model)


@torch.no_grad()
def main():
    proc = AutoProcessor.from_pretrained(HOST)
    host = AutoModelForCausalLM.from_pretrained(
        HOST, dtype=torch.bfloat16, device_map="cuda")
    host.eval()

    im = Image.open(IMG).convert("RGB")
    msg = [{"role": "user", "content": [
        {"type": "image", "image": im},
        {"type": "text", "text": "Describe this image."}]}]
    enc = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt").to("cuda")
    out = host(**enc, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[LAYER][0].float()
    tok_id = getattr(host.config, "image_token_id", None)
    mask = enc["input_ids"][0] == tok_id

    cands = {
        "image-token mean": h[mask].mean(0),
        "last token": h[-1],
        "all-token mean": h.mean(0),
    }
    del host
    torch.cuda.empty_cache()

    ck = torch.load(hf_hub_download("RiverRider/srt-verbalizer-v1",
                                    "gemma4_31b_L47.pt"),
                    map_location="cpu", weights_only=False)
    mu = torch.tensor(ck["mu"]).float().flatten()
    sd = torch.tensor(ck["sd"]).float().flatten()
    print(f"\ntraining mu norm {mu.norm():.2f}   sd mean {sd.mean():.4f}\n")

    pre = Prefix(ck["d_in"], ck["d_model"], ck["n_tok"], ck.get("hidden", 2048))
    pre.load_state_dict(ck["prefix"])
    pre.eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    lm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
                                              dtype=torch.float32)
    lm.eval()

    for name, v in cands.items():
        vc = v.cpu()
        z = (vc - mu) / sd
        soft = pre(z.unsqueeze(0))
        gen = lm.generate(inputs_embeds=soft,
                          attention_mask=torch.ones(soft.shape[:2], dtype=torch.long),
                          max_new_tokens=32, do_sample=False)
        text = tok.decode(gen[0], skip_special_tokens=True).strip()
        print(f"  {name:18s} |v|={vc.norm():7.2f}  |v-mu|={(vc-mu).norm():7.2f}"
              f"  z_absmean={z.abs().mean():.2f}")
        print(f"      -> {text[:110]}\n")


if __name__ == "__main__":
    main()
