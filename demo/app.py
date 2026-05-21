"""Gradio app for the SRT-NLA demo.

Three tabs over a shared backbone selector:

  1. Playground          - prompt -> hidden -> ranked candidate verbalisations
  2. Live Thought Trace  - generate token by token, show AV verbalisation
                           every N tokens alongside each token
  3. Steer by Editing    - verbalise the prompt's hidden state, edit the
                           text, re-encode it, run the model again with
                           the difference patched into layer L

Run locally on a GPU box:
    pip install -r demo/requirements.txt
    python demo/app.py

Or deploy as an HF Space with hardware = A10G or larger; the Qwen-7B
backbone needs ~16 GB of VRAM in bf16, the smaller backbones less.
"""
from __future__ import annotations

import logging
import os

import gradio as gr
import torch

from nla_pipeline import BACKBONES, NLAPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("nla_demo_app")

# Optional ZeroGPU support. On HF Spaces with `hardware: zero-*`, `spaces`
# is preinstalled and exposes the @spaces.GPU decorator that grants the
# decorated function ephemeral CUDA access. Off-Spaces it is a no-op.
try:
    import spaces  # type: ignore

    _ON_ZEROGPU = bool(os.environ.get("SPACES_ZERO_GPU")) or hasattr(spaces, "GPU")

    def _gpu(duration: int = 120):
        if _ON_ZEROGPU:
            return spaces.GPU(duration=duration)
        return lambda fn: fn
except ImportError:  # pragma: no cover - local dev path
    _ON_ZEROGPU = False

    def _gpu(duration: int = 120):
        return lambda fn: fn


DEFAULT_BACKBONE = os.environ.get("NLA_DEFAULT_BACKBONE", "gemma-2-2b")
# On ZeroGPU torch.cuda.is_available() is False at import time but becomes
# True inside @spaces.GPU functions. Pin device to cuda when we know the
# Space has a GPU slice.
DEVICE = "cuda" if (torch.cuda.is_available() or _ON_ZEROGPU) else "cpu"
log.info("device=%s zero_gpu=%s", DEVICE, _ON_ZEROGPU)

# Lazy cache: load each backbone the first time it's selected.
_PIPES: dict[str, NLAPipeline] = {}


def get_pipe(key: str) -> NLAPipeline:
    if key not in _PIPES:
        log.info("First use of backbone %s; loading.", key)
        _PIPES[key] = NLAPipeline(BACKBONES[key], device=DEVICE)
    return _PIPES[key]


# ---------------------------------------------------------------------------
# Tab callbacks
# ---------------------------------------------------------------------------

@_gpu(duration=120)
def cb_playground(backbone_key: str, prompt: str, K: int, temperature: float):
    if not prompt.strip():
        return "_(enter a prompt)_", ""
    pipe = get_pipe(backbone_key)
    v = pipe.extract_hidden(prompt, token_index=-1)
    ranked = pipe.verbalize(v, K=int(K), temperature=float(temperature))
    rows = [
        f"| {i+1} | {cen:.3f} | {raw:.3f} | {txt[:160].replace(chr(10), ' ')} |"
        for i, (txt, raw, cen) in enumerate(ranked)
    ]
    table = "| rank | centred fve | raw fve | verbalisation |\n|---|---|---|---|\n" + "\n".join(rows)
    mu_str = f"{pipe.mu.norm().item():.2f}" if pipe.mu is not None else "n/a"
    summary = (
        f"**Backbone:** {pipe.spec.label}  "
        f"**hidden norm** = {v.norm().item():.2f}  "
        f"**centring pool norm** = {mu_str}"
    )
    return summary, table


@_gpu(duration=180)
def cb_thought_trace(backbone_key: str, prompt: str, max_new: int, every: int):
    if not prompt.strip():
        yield "_(enter a prompt)_"
        return
    pipe = get_pipe(backbone_key)
    rows: list[str] = ["| step | token | verbalisation of running hidden |", "|---|---|---|"]
    yielded = 0
    for step, (tok, verb) in enumerate(
        pipe.thought_trace(prompt, max_new_tokens=int(max_new), every=int(every), K=4)
    ):
        if verb is None:
            rows.append(f"| {step} | `{tok}` | _(skipped, not at every-{every} step)_ |")
        else:
            rows.append(f"| {step} | `{tok}` | {verb[:140].replace(chr(10), ' ')} |")
        yielded += 1
        # Stream every few rows
        if yielded % 4 == 0 or yielded == int(max_new):
            yield "\n".join(rows)
    yield "\n".join(rows)


@_gpu(duration=180)
def cb_steer(
    backbone_key: str,
    prompt: str,
    replacement: str,
    alpha: float,
    max_new: int,
):
    if not prompt.strip():
        return "_(enter a prompt)_", "", ""
    pipe = get_pipe(backbone_key)
    # 1. Original verbalisation (best of 4) of the prompt's hidden
    v = pipe.extract_hidden(prompt, token_index=-1)
    ranked = pipe.verbalize(v, K=4, temperature=1.0)
    base_verb = ranked[0][0] if ranked else ""

    # 2. Original (unsteered) greedy continuation
    ids = pipe.tokenizer(prompt, return_tensors="pt").to(pipe.device)
    with torch.no_grad():
        gen = pipe.backbone.generate(
            **ids,
            max_new_tokens=int(max_new),
            do_sample=False,
            pad_token_id=pipe.tokenizer.pad_token_id,
        )
    base_cont = pipe.tokenizer.decode(
        gen[0, ids.input_ids.shape[1]:], skip_special_tokens=True
    )

    # 3. Steered continuation: re-encode replacement, patch into layer L
    if not replacement.strip():
        replacement = base_verb
    steered_cont = pipe.steer_with_text(
        prompt, replacement, alpha=float(alpha), max_new_tokens=int(max_new)
    )
    return base_verb, base_cont, steered_cont


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="SRT-NLA: ask the model what it is thinking") as app:
    gr.Markdown(
        """# SRT-NLA Demo

Ask a frozen language model what is in its own internal hidden state, in
plain English, in three different ways. The trained Activation
Verbalizer (AV) is a small adapter (~5–13M params depending on backbone)
sitting on top of a fully frozen backbone. The backbone weights are
unchanged from the public release. Companion paper:
[`paper_nla.md`](https://github.com/space-bacon/SRT/blob/main/paper_nla.md).
"""
    )
    backbone_key = gr.Radio(
        choices=[(spec.label, key) for key, spec in BACKBONES.items()],
        value=DEFAULT_BACKBONE,
        label="Frozen backbone",
    )

    with gr.Tab("1 - Playground"):
        gr.Markdown(
            "Enter any prompt. We extract the last-token hidden state at the "
            "configured layer, sample K candidate verbalisations, then for each "
            "candidate we feed it back through the same frozen model and measure "
            "how well its own re-encoded hidden state matches the original (raw "
            "and anisotropy-centred cosine fidelity)."
        )
        with gr.Row():
            with gr.Column():
                pl_prompt = gr.Textbox(
                    label="Prompt",
                    value="The Eiffel Tower stands on the Champ de Mars in Paris,",
                    lines=3,
                )
                pl_K = gr.Slider(1, 32, value=8, step=1, label="K (samples)")
                pl_temp = gr.Slider(0.1, 1.5, value=1.0, step=0.05, label="Temperature")
                pl_btn = gr.Button("Verbalize", variant="primary")
            with gr.Column():
                pl_summary = gr.Markdown()
                pl_table = gr.Markdown()
        pl_btn.click(
            cb_playground,
            inputs=[backbone_key, pl_prompt, pl_K, pl_temp],
            outputs=[pl_summary, pl_table],
        )

    with gr.Tab("2 - Live thought trace"):
        gr.Markdown(
            "Watch the model generate token by token while we periodically "
            "verbalise its running last-token hidden state. The right column is "
            "the model talking *about* what its current hidden state is about, "
            "while the left column is what it is choosing to *say next*."
        )
        with gr.Row():
            with gr.Column():
                tt_prompt = gr.Textbox(
                    label="Prompt",
                    value="In one paragraph, explain why the sky is blue.",
                    lines=3,
                )
                tt_max = gr.Slider(8, 64, value=24, step=4, label="Tokens to generate")
                tt_every = gr.Slider(1, 8, value=4, step=1, label="Verbalise every N tokens")
                tt_btn = gr.Button("Trace", variant="primary")
            with gr.Column():
                tt_out = gr.Markdown()
        tt_btn.click(
            cb_thought_trace,
            inputs=[backbone_key, tt_prompt, tt_max, tt_every],
            outputs=tt_out,
        )

    with gr.Tab("3 - Steer by editing"):
        gr.Markdown(
            "Activation steering with a sentence editor. We verbalise the "
            "prompt's hidden state into plain English, you edit the text, we "
            "re-encode the edited text into a new hidden state, and we add "
            "`alpha * (v_new - v_orig)` to layer L's residual stream while the "
            "model generates. Compare the unsteered and steered continuations "
            "side by side."
        )
        with gr.Row():
            with gr.Column():
                st_prompt = gr.Textbox(
                    label="Prompt",
                    value="Tell me a short fact about cats.",
                    lines=3,
                )
                st_replacement = gr.Textbox(
                    label="Replacement description (leave blank to use the AV's own verbalisation)",
                    placeholder="e.g. 'a fact about deep-sea fish'",
                    lines=2,
                )
                st_alpha = gr.Slider(-2.0, 2.0, value=1.0, step=0.05, label="alpha")
                st_max = gr.Slider(8, 128, value=64, step=8, label="Tokens to generate")
                st_btn = gr.Button("Steer", variant="primary")
            with gr.Column():
                st_base_verb = gr.Textbox(label="AV's verbalisation of the original prompt", lines=2)
                st_base_cont = gr.Textbox(label="Unsteered continuation", lines=4)
                st_steered = gr.Textbox(label="Steered continuation", lines=4)
        st_btn.click(
            cb_steer,
            inputs=[backbone_key, st_prompt, st_replacement, st_alpha, st_max],
            outputs=[st_base_verb, st_base_cont, st_steered],
        )

    gr.Markdown(
        """---
**Notes.** The first call on each backbone downloads weights from
HuggingFace; subsequent calls reuse the in-memory cache.  Centred fve
uses a 2K-sample anisotropy pool published in the matching dataset
repo.  Steering is implemented as a single forward hook on layer L and
does not retrain anything.  Hardware: roughly 16 GB VRAM for Qwen-7B
in bf16; less for Llama-3B and Gemma-2B.
"""
    )


if __name__ == "__main__":
    # On HF Spaces (including ZeroGPU) the platform proxies the app and sets
    # SPACE_ID; gradio picks up server_name=0.0.0.0 from the environment.
    if os.environ.get("SPACE_ID"):
        os.environ.setdefault("GRADIO_SERVER_NAME", "0.0.0.0")
        app.queue().launch()
    else:
        app.queue().launch(server_name="127.0.0.1", share=False)
