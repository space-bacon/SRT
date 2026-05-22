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


DEFAULT_BACKBONE = os.environ.get("NLA_DEFAULT_BACKBONE", "qwen2.5-7b")
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
    metric_rows = [
        f"| {i+1} | {cen:.3f} | {raw:.3f} |"
        for i, (_txt, raw, cen) in enumerate(ranked)
    ]
    metric_table = (
        "| rank | centred fve | raw fve |\n|---|---|---|\n"
        + "\n".join(metric_rows)
    )
    full_blocks = [
        f"**rank {i+1}** — centred fve {cen:.3f}, raw fve {raw:.3f}\n\n> "
        + txt.replace("\n", "\n> ")
        for i, (txt, raw, cen) in enumerate(ranked)
    ]
    table = metric_table + "\n\n---\n\n" + "\n\n".join(full_blocks)
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
            rows.append(f"| {step} | `{tok}` | {verb.replace(chr(10), ' ')} |")
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
# Glass-box chat
# ---------------------------------------------------------------------------

def _build_chat_prompt(pipe: NLAPipeline, history: list[dict], user_msg: str) -> str:
    """Render `history + user_msg` into a single prompt string.

    These are *base* models (no instruct/RLHF), so the chat template
    wraps the input in `<|im_start|>user … <|im_start|>assistant` and
    the base model — never having seen that pattern — emits another
    Q/A pair instead of an answer.  Instead, we treat each user turn
    as a **completion prompt**: prior turns are stitched as
    "user_text reply_text\\n\\n" pairs (no role labels), and the
    current `user_msg` is appended verbatim with no trailing assistant
    primer.  The model then simply continues writing — which is
    exactly what base LMs are good at.
    """
    parts: list[str] = []
    pending_user: str | None = None
    for m in history or []:
        if m["role"] == "user":
            pending_user = m["content"]
        else:  # assistant
            if pending_user is not None:
                parts.append(f"{pending_user} {m['content']}")
                pending_user = None
            else:
                parts.append(m["content"])
    parts.append(user_msg)
    return "\n\n".join(parts)


@_gpu(duration=180)
def cb_chat(
    backbone_key: str,
    history: list[dict],
    rail: str,
    user_msg: str,
    thought_edit: str,
    alpha: float,
    max_new: int,
):
    if not user_msg.strip():
        return history, rail, "", thought_edit
    pipe = get_pipe(backbone_key)
    prompt = _build_chat_prompt(pipe, history or [], user_msg)

    # 1. Read the model's pre-response state and verbalise it.
    v = pipe.extract_hidden(prompt, token_index=-1)
    ranked = pipe.verbalize(v, K=4, temperature=1.0, max_new_tokens=32)
    thought, raw, cen = (ranked[0] if ranked else ("(none)", 0.0, 0.0))

    # 2. Generate the assistant reply, optionally steered by `thought_edit`.
    steered = bool(thought_edit.strip())
    if steered:
        reply = pipe.steer_with_text(
            prompt,
            thought_edit.strip(),
            alpha=float(alpha),
            max_new_tokens=int(max_new),
        )
    else:
        ids = pipe.tokenizer(prompt, return_tensors="pt").to(pipe.device)
        with torch.no_grad():
            gen = pipe.backbone.generate(
                **ids,
                max_new_tokens=int(max_new),
                do_sample=False,
                pad_token_id=pipe.tokenizer.pad_token_id,
            )
        reply = pipe.tokenizer.decode(
            gen[0, ids.input_ids.shape[1]:], skip_special_tokens=True
        )

    # Trim trailing user/assistant tags some chat templates leak, and
    # cut at the first paragraph break so completion-style replies
    # don't ramble into a new topic.
    for stop in ("\n\n", "\nUser:", "\nuser:", "\nQ:", "<|im_end|>", "<|eot_id|>", "<end_of_turn>"):
        if stop in reply:
            reply = reply.split(stop, 1)[0]
    reply = reply.strip()

    # 3. Update history (messages-format) and rail (latest turn on top).
    new_history = list(history or []) + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": reply},
    ]
    badge = " · _(steered)_" if steered else ""
    rail_entry = (
        f"**turn {len(new_history) // 2}**{badge} — centred fve `{cen:.3f}`, raw `{raw:.3f}`\n\n"
        f"> {thought.strip().replace(chr(10), ' ')}\n"
    )
    new_rail = rail_entry + ("\n---\n\n" + rail if rail else "")
    return new_history, new_rail, "", ""


def cb_chat_clear():
    return [], "", "", ""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_FORCE_DARK_JS = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'dark') {
    url.searchParams.set('__theme', 'dark');
    window.location.href = url.href;
  }
}
"""

_DARK_CSS = """
html, body, gradio-app, .gradio-container {
    background: #0b0f19 !important;
    color: #e5e7eb !important;
}
.gradio-container .prose,
.gradio-container .prose *,
.gradio-container .markdown,
.gradio-container .markdown *,
.gradio-container p,
.gradio-container li,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container span,
.gradio-container label {
    color: #e5e7eb !important;
}
.gradio-container a { color: #fb923c !important; }
.gradio-container code { color: #fde68a !important; background: #1f2937 !important; }
"""

with gr.Blocks(
    title="MindReader-NLA: ask the model what it is thinking",
    theme=gr.themes.Default(),
    js=_FORCE_DARK_JS,
    css=_DARK_CSS,
) as app:
    gr.Markdown(
        """# MindReader-NLA

Ask a frozen language model what is in its own internal hidden state, in
plain English, in three different ways. The trained Activation
Verbalizer (AV) is a small adapter (~5–13M params depending on backbone)
sitting on top of a fully frozen backbone. The backbone weights are
unchanged from the public release. Companion paper:
[`paper_nla.md`](https://github.com/space-bacon/SRT/blob/main/paper_nla.md).
"""
    )
    with gr.Accordion("ℹ️ Glossary & how to read this demo", open=False):
        gr.Markdown(
            """
**Backbone.** The frozen pretrained language model (Qwen-2.5-7B,
Llama-3.2-3B, or Gemma-2-2B). Its weights are *never* updated — only the
small AV adapter on top is trained.

**Layer L.** The transformer layer whose residual-stream activation is
read out and verbalised. Picked once per backbone during AV training:
L20 for Qwen, L20 for Llama-3B, L19 for Gemma-2B.

**Hidden state / activation `v`.** The model's last-token residual
vector at layer L. This is *what the model is currently thinking about*,
expressed as a `d`-dimensional vector (`d` = 3584 for Qwen, 3072 Llama,
2304 Gemma).

**Activation Verbalizer (AV).** A small adapter (~5–13M params,
depending on `d`) that maps `v` to a short prefix of soft tokens, then
asks the same frozen backbone to *describe `v` in plain English*.

**Verbalisation.** The natural-language sentence the AV produces for
`v`. Multiple candidates are sampled at temperature `T` and ranked.

**raw fve.** *Fraction of variance explained.* We re-encode the
verbalised sentence back into a hidden state `v_hat` through the same
frozen model, then measure how close `v_hat` is to the original `v`
under cosine similarity, normalised. **Higher is better; 1.0 is perfect.**

**centred fve.** Same as raw fve, but after subtracting the mean
activation `μ` (the *anisotropy pool*) from both `v` and `v_hat`. This
removes the trivial "all activations are similar in direction" baseline
and is the metric that actually reflects content fidelity.

**μ / centring pool norm.** The norm of the anisotropy mean μ. Large μ
means the model has a strong constant component in its activations at
this layer; centred fve corrects for it.

**alpha (steering tab).** Strength of the activation edit:
`v_used = v + alpha · (v_new − v_orig)`, applied as a single forward
hook on layer L. `alpha = 0` is the unsteered baseline; `alpha = 1`
fully replaces the original direction with the edited one.
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
            lambda: ("⏳ Running on ZeroGPU — first call also downloads weights, expect 30–90 s. Please keep this tab in the foreground.", ""),
            inputs=None,
            outputs=[pl_summary, pl_table],
            queue=False,
        ).then(
            cb_playground,
            inputs=[backbone_key, pl_prompt, pl_K, pl_temp],
            outputs=[pl_summary, pl_table],
            show_progress="full",
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
            lambda: "⏳ Tracing on ZeroGPU — first call also downloads weights, expect 30–90 s. Tokens will stream in below as they are generated.",
            inputs=None,
            outputs=tt_out,
            queue=False,
        ).then(
            cb_thought_trace,
            inputs=[backbone_key, tt_prompt, tt_max, tt_every],
            outputs=tt_out,
            show_progress="full",
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
            lambda: ("⏳ running…", "⏳ running…", "⏳ running…"),
            inputs=None,
            outputs=[st_base_verb, st_base_cont, st_steered],
            queue=False,
        ).then(
            cb_steer,
            inputs=[backbone_key, st_prompt, st_replacement, st_alpha, st_max],
            outputs=[st_base_verb, st_base_cont, st_steered],
            show_progress="full",
        )

    with gr.Tab("4 - Glass-box chat"):
        gr.Markdown(
            "A normal chat, plus a live right-rail showing the AV's "
            "verbalisation of the model's hidden state *immediately before "
            "it responds to each of your messages* (layer L, last token of "
            "the assembled chat prompt). Optionally edit that thought before "
            "you send: the edited text is re-encoded, and `alpha · "
            "(v_edited − v_original)` is patched into layer L while the "
            "reply is generated. This is *steering by editing what the "
            "model is thinking about*, mid-conversation.\n\n"
            "> **Tip.** These are *base* models (no RLHF), so phrase your "
            "turn as the **start of the answer**, not as a question. "
            "Write `The capital of France is` — not `What is the capital "
            "of France?`. The examples below are pre-formatted; click any "
            "one to try it."
        )
        chat_state = gr.State([])
        with gr.Row():
            with gr.Column(scale=3):
                chat_box = gr.Chatbot(
                    label="Conversation",
                    type="messages",
                    height=420,
                )
                chat_user = gr.Textbox(
                    label="Your message  (write the *start of the answer*, "
                    "e.g. 'The capital of France is')",
                    placeholder="The capital of France is",
                    lines=2,
                )
                gr.Markdown(
                    "**Mind-reading** — the answer is encoded in the "
                    "hidden state *before* it's spoken. Watch the rail "
                    "match the reply."
                )
                gr.Examples(
                    examples=[
                        ["The capital of France is"],
                        ["The chemical symbol for gold is"],
                        ["The fastest land animal is the"],
                        ["The largest planet in our solar system is"],
                        ['"Good morning" in French is'],
                        ["Two plus two equals"],
                        ["17 × 24 = let me work it out. 17 × 24 = 17 × (20 + 4) = 340 + 68 ="],
                    ],
                    inputs=chat_user,
                    label="① Mind-reading prompts",
                )
                gr.Markdown(
                    "**Causal steering** — open *Steer this reply* and "
                    "type a replacement thought (e.g. `Lyon`, `green`, "
                    "`a spaceship`). The reply flips, *coherently*, "
                    "because we patched the residual — not the prompt."
                )
                gr.Examples(
                    examples=[
                        ["The capital of France is"],            # steer: Lyon
                        ["Roses are red, violets are"],          # steer: green
                        ["Once upon a time, there was a"],       # steer: spaceship
                        ["The Mona Lisa was painted by"],        # steer: a child
                        ["In a surprising twist, the hero"],     # steer: gave up
                        ["The CEO of Apple is"],                 # steer: a robot
                    ],
                    inputs=chat_user,
                    label="② Best for steering",
                )
                gr.Markdown(
                    "**Priors & biases** — the rail surfaces what the "
                    "model *thinks* before any hedging or politeness "
                    "kicks in. Often more revealing than the reply."
                )
                gr.Examples(
                    examples=[
                        ["My honest opinion is that pineapple on pizza"],
                        ["The most underrated programming language is"],
                        ["She was furious because"],
                        ["If you put a metal spoon in a microwave, it will"],
                        ["The meaning of life is"],
                        ["Dear diary, today I felt"],
                        ["The murderer was actually"],
                    ],
                    inputs=chat_user,
                    label="③ Reveal the model's priors",
                )
                with gr.Accordion("Steer this reply (optional)", open=False):
                    chat_edit = gr.Textbox(
                        label="Replacement thought (leave blank for unsteered reply)",
                        placeholder="e.g. 'a careful, hedged answer about uncertainty'",
                        lines=2,
                    )
                    chat_alpha = gr.Slider(
                        -2.0, 2.0, value=1.0, step=0.05, label="alpha"
                    )
                with gr.Row():
                    chat_max = gr.Slider(
                        16, 256, value=96, step=16, label="Max new tokens"
                    )
                    chat_btn = gr.Button("Send", variant="primary")
                    chat_clear = gr.Button("Clear")
            with gr.Column(scale=2):
                gr.Markdown("### 🧠 Thought rail")
                gr.Markdown(
                    "_The AV's best verbalisation of the model's hidden "
                    "state at the moment it was about to respond to each "
                    "of your messages, latest turn on top._"
                )
                chat_rail = gr.Markdown()

        chat_btn.click(
            cb_chat,
            inputs=[
                backbone_key,
                chat_state,
                chat_rail,
                chat_user,
                chat_edit,
                chat_alpha,
                chat_max,
            ],
            outputs=[chat_state, chat_rail, chat_user, chat_edit],
            show_progress="full",
            api_name=False,
        ).then(
            lambda h: h,
            inputs=chat_state,
            outputs=chat_box,
            queue=False,
            api_name=False,
        )
        chat_user.submit(
            cb_chat,
            inputs=[
                backbone_key,
                chat_state,
                chat_rail,
                chat_user,
                chat_edit,
                chat_alpha,
                chat_max,
            ],
            outputs=[chat_state, chat_rail, chat_user, chat_edit],
            show_progress="full",
            api_name=False,
        ).then(
            lambda h: h,
            inputs=chat_state,
            outputs=chat_box,
            queue=False,
            api_name=False,
        )
        chat_clear.click(
            cb_chat_clear,
            inputs=None,
            outputs=[chat_state, chat_rail, chat_user, chat_edit],
            queue=False,
            api_name=False,
        ).then(
            lambda: [],
            inputs=None,
            outputs=chat_box,
            queue=False,
            api_name=False,
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
    on_space = bool(os.environ.get("SPACE_ID"))
    if on_space:
        app.queue().launch(server_name="0.0.0.0", server_port=7860)
    else:
        app.queue().launch()
