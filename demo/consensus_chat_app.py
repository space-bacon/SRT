"""Consensus chat: K candidates per turn, one chosen by what the pool agrees on.

The selector gets the user's message and K replies. No test suite, no reference
solution, no benchmark metadata. It recovers the function under discussion, infers
plausible arguments, runs every candidate on the same synthesised inputs, and keeps
the largest group that agrees on outputs.

Banked measurement for this exact model (artifacts/nla/verifier/chat_consensus.json,
HumanEval, K=8): a single sample passes 0.5404, this selector reaches 0.7561, and the
best candidate in the pool would reach 0.9024.

    PYTHONPATH=. python demo/consensus_chat_app.py
"""

from __future__ import annotations

import os
import sys
import time

import gradio as gr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))

from chat_consensus import choose, code_of  # noqa: E402

MODEL = os.environ.get("CONSENSUS_MODEL", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
ARM = "coder7B_inst__chat"
BANKED = {"single": 0.5404, "selected": 0.7561, "oracle": 0.9024}

# Palette and type lifted from docs/sidecar_receipts.html, already live on the lab site.
LAB_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;1,600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
"""

LAB_CSS = """
:root, .gradio-container, gradio-app, body {
  --bg:#faf7f2; --ink:#4a423b; --terra:#b0603e; --muted:#8b8178;
  --line:#e7dfd5; --panel:#ffffff; --sand:#f4eee6;
}

html, body, gradio-app, .gradio-container {
  background: #faf7f2 !important;
  color: #4a423b !important;
}

:root, .gradio-container {
  --body-background-fill: var(--bg);
  --background-fill-primary: var(--bg);
  --background-fill-secondary: var(--sand);
  --block-background-fill: var(--panel);
  --input-background-fill: var(--panel);
  --input-background-fill-focus: var(--panel);
  --input-background-fill-hover: var(--panel);
  --input-text-color: var(--ink);
  --input-placeholder-color: #b3a99e;
  --input-border-color: var(--line);
  --input-border-color-focus: var(--terra);
  --input-border-color-hover: #d9cec2;
  --body-text-color: var(--ink);
  --body-text-color-subdued: var(--muted);
  --border-color-primary: var(--line);
  --border-color-accent: var(--terra);
  --block-border-color: var(--line);
  --color-accent: var(--terra);
  --link-text-color: var(--terra);
  --link-text-color-hover: var(--terra);
  --block-label-background-fill: var(--sand);
  --block-label-text-color: var(--muted);
  --block-label-border-color: var(--line);
  --block-title-text-color: var(--muted);
  --panel-background-fill: var(--panel);
  --chatbot-text-size: 15px;
  --button-primary-background-fill: var(--terra);
  --button-primary-background-fill-hover: #9c5335;
  --button-primary-text-color: #fff;
  --button-primary-border-color: var(--terra);
  --slider-color: var(--terra);
  --checkbox-background-color-selected: var(--terra);
  --block-radius: 12px;
  --container-radius: 12px;
}

.gradio-container {
  background: var(--bg) !important;
  font: 16px/1.65 Inter, system-ui, sans-serif !important;
  max-width: 980px !important;
  margin: 0 auto !important;
}

.gradio-container h1, .gradio-container h2, .gradio-container h3 {
  font-family: "Playfair Display", serif !important;
  color: var(--ink) !important;
  letter-spacing: 0;
}
.gradio-container h1 { font-size: 42px !important; font-weight: 700 !important; line-height: 1.15 !important; margin: 0 0 10px !important; }
.gradio-container h2 { font-size: 24px !important; font-weight: 600 !important; margin: 28px 0 6px !important; }

.lab-kicker {
  font: 600 11px/1 Inter, sans-serif; letter-spacing: .22em; text-transform: uppercase;
  color: var(--terra); margin-bottom: 14px;
}
.lab-sub { color: var(--muted); font-size: 18px; margin: 0 0 10px; }
.lab-note { color: var(--muted); font-size: 13.5px; border-top: 1px solid var(--line); padding-top: 16px; margin-top: 8px; }
.lab-pill {
  display: inline-block; background: var(--sand); border: 1px solid var(--line);
  border-radius: 999px; padding: 2px 12px; font-size: 12.5px; color: var(--muted);
  margin: 2px 4px 2px 0;
}
.lab-fig { color: var(--terra); font-weight: 600; font-variant-numeric: tabular-nums; }

.gradio-container .block,
.gradio-container .form {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}
.gradio-container .panel {
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  padding: 4px 16px !important;
}

.gradio-container code, .gradio-container pre {
  font: 13.5px/1.5 ui-monospace, Menlo, monospace !important;
  background: var(--sand) !important;
  border-radius: 6px !important;
}
.gradio-container pre { border: 1px solid var(--line) !important; padding: 14px 16px !important; }

.gradio-container table {
  border-collapse: collapse; background: var(--panel);
  border: 1px solid var(--line); border-radius: 10px; overflow: hidden; font-size: 14.5px;
}
.gradio-container th { font-weight: 600; background: var(--sand); }
.gradio-container th, .gradio-container td { padding: 9px 12px; border-bottom: 1px solid var(--line); }

footer { display: none !important; }

/* Base theme ships dark stone focus fills; force every input to stay on paper. */
.gradio-container textarea,
.gradio-container input[type="text"],
.gradio-container input[type="number"] {
  background: var(--panel) !important;
  color: var(--ink) !important;
  caret-color: var(--terra) !important;
  border-radius: 10px !important;
}
.gradio-container textarea:focus,
.gradio-container input[type="text"]:focus,
.gradio-container input[type="number"]:focus {
  background: var(--panel) !important;
  color: var(--ink) !important;
  border-color: var(--terra) !important;
  box-shadow: 0 0 0 3px rgba(176, 96, 62, .12) !important;
  outline: none !important;
}
.gradio-container textarea::placeholder { color: #b3a99e !important; }

.gradio-container .block-label,
.gradio-container label span,
.gradio-container .label-wrap span {
  color: var(--muted) !important;
}
.gradio-container .chatbot, .gradio-container .bubble-wrap {
  background: var(--panel) !important;
}

.verdict {
  background: var(--panel); border: 1px solid var(--line); border-left: 3px solid var(--terra);
  border-radius: 12px; padding: 14px 18px; margin: 2px 0 4px;
}
.verdict.quiet { border-left-color: var(--line); }
.verdict .headline { font: 600 17px/1.4 "Playfair Display", serif; color: var(--ink); margin: 0 0 4px; }
.verdict .detail { color: var(--muted); font-size: 14px; margin: 0; }
.verdict .split { color: var(--terra); font-weight: 600; }
.verdict .meta { color: var(--muted); font-size: 12.5px; margin-top: 8px; font-variant-numeric: tabular-nums; }

.empty-hint { color: var(--muted); font-size: 14.5px; padding: 6px 2px 0; }

/* --- strip Gradio chrome ------------------------------------------------ */
footer, .gradio-container .icon-button-wrapper,
#chat .icon-button-wrapper, #examples > .label {
  display: none !important;
}
#verdict:empty, #verdict > div:empty { display: none !important; }
#verdict { margin: 10px 0 0 !important; }
#verdict:empty { margin: 0 !important; }
.gradio-container .block { box-shadow: none !important; }
#chat {
  border: 1px solid var(--line) !important; border-radius: 14px !important;
  background: var(--panel) !important; padding: 6px 4px !important;
}
#chat .placeholder, #chat .placeholder * {
  color: var(--muted) !important; font-size: 14.5px !important; border: none !important;
}

/* --- chat: assistant is prose, user is a quiet pill --------------------- */
#chat .message-row { border: none !important; }
#chat .bot, #chat [data-testid="bot"] {
  background: transparent !important; border: none !important;
  color: var(--ink) !important; padding: 2px 6px 10px !important;
}
#chat .user, #chat [data-testid="user"] {
  background: rgba(176, 96, 62, .08) !important;
  border: 1px solid rgba(176, 96, 62, .16) !important;
  color: var(--ink) !important; border-radius: 14px !important;
}
#chat .bot pre, #chat [data-testid="bot"] pre {
  background: var(--sand) !important; border: 1px solid var(--line) !important; border-radius: 10px !important;
}

/* --- composer pill, BlackWindow idiom ----------------------------------- */
#composer {
  background: var(--panel); border: 1px solid var(--line); border-radius: 16px;
  padding: 6px 6px 6px 14px; gap: 8px; align-items: center; margin-top: 12px;
  transition: border-color .15s ease, box-shadow .15s ease;
}
#composer:focus-within {
  border-color: rgba(176, 96, 62, .55);
  box-shadow: 0 0 0 4px rgba(176, 96, 62, .08);
}
#composer .block, #composer .form, #composer > div {
  background: transparent !important; border: none !important; box-shadow: none !important;
}
#msg textarea {
  background: transparent !important; border: none !important; box-shadow: none !important;
  padding: 8px 0 !important; font-size: 15.5px !important; resize: none !important;
}
#msg textarea:focus { box-shadow: none !important; border: none !important; }
#send {
  width: 38px !important; height: 38px !important; min-width: 38px !important;
  border-radius: 999px !important; padding: 0 !important;
  background: var(--terra) !important; color: #fff !important; border: none !important;
  font-size: 17px !important; line-height: 1 !important; flex: 0 0 38px !important;
}
#send:hover { background: #9c5335 !important; }
#send:disabled { background: #d8c9bd !important; }

/* --- examples as quiet pills, no table chrome --------------------------- */
#examples { background: transparent !important; border: none !important; }
#examples table, #examples tbody, #examples tr { border: none !important; background: transparent !important; }
#examples td, #examples button {
  background: var(--sand) !important; border: 1px solid var(--line) !important;
  color: var(--muted) !important; border-radius: 999px !important;
  font-size: 13px !important; padding: 7px 14px !important; cursor: pointer;
}
#examples td:hover, #examples button:hover {
  border-color: var(--terra) !important; color: var(--terra) !important; background: var(--panel) !important;
}

/* --- accordions read as quiet rules, not boxes -------------------------- */
.gradio-container .label-wrap {
  font: 600 11px/1 Inter, sans-serif !important; letter-spacing: .16em;
  text-transform: uppercase; color: var(--muted) !important;
}
.gradio-container .label-wrap:hover { color: var(--terra) !important; }

/* --- kill Gradio's progress chrome -------------------------------------- */
.gradio-container .progress-text,
.gradio-container .progress-level,
.gradio-container .progress-bar,
.gradio-container .meta-text,
.gradio-container .meta-text-center,
.gradio-container .eta-bar,
.gradio-container .wrap.default.full,
.gradio-container .status-tracker { display: none !important; }
.gradio-container .generating,
.gradio-container .block.generating {
  border-color: var(--line) !important; animation: none !important; box-shadow: none !important;
}
#chat.generating { border-color: var(--line) !important; }

/* Gradio dims pending blocks to 0.2 and drops an 80px animated logo beside them. */
.gradio-container .pending,
.gradio-container .html-container.pending { opacity: 1 !important; }
#verdict svg, #verdict img { display: none !important; }

/* send button becomes the only progress indicator */
#send:disabled { color: transparent !important; position: relative; background: #c98f70 !important; }
#send:disabled::after {
  content: ""; position: absolute; inset: 0; margin: auto;
  width: 15px; height: 15px; border-radius: 999px;
  border: 2px solid rgba(255, 255, 255, .35); border-top-color: #fff;
  animation: spin .7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

.working { display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: 14px; padding: 12px 2px 2px; }
.working .dot {
  width: 7px; height: 7px; border-radius: 999px; background: var(--terra);
  animation: pulse 1.1s ease-in-out infinite;
}
@keyframes pulse { 0%,100% { opacity: .25; } 50% { opacity: 1; } }

/* --- branded scrollbar -------------------------------------------------- */
#chat *::-webkit-scrollbar { width: 8px; height: 8px; }
#chat *::-webkit-scrollbar-track { background: transparent; }
#chat *::-webkit-scrollbar-thumb { background: rgba(176, 96, 62, .35); border-radius: 999px; }
#chat *::-webkit-scrollbar-thumb:hover { background: rgba(176, 96, 62, .55); }
#chat * { scrollbar-width: thin; scrollbar-color: rgba(176,96,62,.35) transparent; }
"""

LAB_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
)

_MODEL = None


def _load():
    global _MODEL
    if _MODEL is None:
        from mlx_lm import load

        _MODEL = load(MODEL)
    return _MODEL


def _generate(message, k, max_tokens, temp):
    from mlx_lm import batch_generate
    from mlx_lm.sample_utils import make_sampler

    model, tok = _load()
    # Pass ids, never a re-tokenised template string.
    ids = tok.apply_chat_template(
        [{"role": "user", "content": message}], tokenize=True, add_generation_prompt=True
    )
    out = batch_generate(
        model, tok, prompts=[ids] * k, max_tokens=max_tokens,
        sampler=make_sampler(temp=temp, top_p=0.95), verbose=False,
    )
    return out.texts if hasattr(out, "texts") else out


def respond(message, history, k, max_tokens, temp):
    if not message.strip():
        return history, "", ""

    t0 = time.time()
    replies = _generate(message, int(k), int(max_tokens), float(temp))
    t_gen = time.time() - t0

    t0 = time.time()
    pick, info = choose(message, replies)
    t_sel = time.time() - t0

    if pick is None:
        answer = replies[0]
        note = (
            '<div class="verdict quiet">'
            "<p class=\"headline\">No selection was possible.</p>"
            f'<p class="detail">{info.get("reason", "unknown")}. '
            f"Showing the first of {len(replies)} candidates, unselected.</p>"
            "</div>"
        )
        others = ""
    else:
        answer = replies[pick]
        ran, size, clusters = info["ran"], info["cluster_size"], info["clusters"]
        if clusters == 1:
            headline = f"All {size} candidates agreed."
            detail = "The pool computed one answer. Nothing to choose between."
        else:
            headline = f'{size} of {ran} agreed, <span class="split">the pool split {clusters} ways</span>.'
            detail = (
                f"{ran - size} candidate(s) computed something different on the same inputs. "
                "They are below, and they read just as confidently."
            )
        dropped = len(replies) - ran
        meta = f"{len(replies)} sampled in {t_gen:.1f}s, selected in {t_sel:.2f}s, entry <code>{info['entry']}</code>"
        if dropped:
            meta += f", {dropped} failed to run"
        note = (
            '<div class="verdict">'
            f'<p class="headline">{headline}</p>'
            f'<p class="detail">{detail}</p>'
            f'<p class="meta">{meta}</p>'
            "</div>"
        )
        losers = [i for i in range(len(replies)) if i != pick]
        others = "\n\n".join(
            f"**Candidate {i + 1}**\n```python\n{code_of(replies[i])[:1500]}\n```" for i in losers
        )

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return history, note, others


with gr.Blocks(title="Consensus · Sunstone North Lab") as demo:
    gr.HTML(
        f"""<div class="lab-kicker">Sunstone North · Lab · Consensus</div>
<h1>Ask for code. Get the answer the pool agrees on.</h1>
<p class="lab-sub">Every turn samples K candidates and returns the one the others back up.</p>
<p>The choice is made by running all of them on synthesised inputs and keeping the largest
group that computes the same thing. No test suite, no labels, no benchmark metadata, just
your message and the replies.</p>
<p>Measured on HumanEval for <code>{ARM}</code>: one sample passes
<span class="lab-fig">{BANKED['single']}</span>, this selector reaches
<span class="lab-fig">{BANKED['selected']}</span>, and the best candidate in the pool would
reach <span class="lab-fig">{BANKED['oracle']}</span>. Those are benchmark figures, not a
claim about the request you are about to type.</p>
<p><span class="lab-pill">{MODEL}</span><span class="lab-pill">candidates run in a sandboxed subprocess</span><span class="lab-pill">one desktop computer</span></p>
"""
    )
    chat = gr.Chatbot(
        height=430,
        show_label=False,
        elem_id="chat",
        placeholder="Ask for a Python function. Eight candidates get written, run, and compared.",
    )
    verdict = gr.HTML(elem_id="verdict")
    with gr.Row(elem_id="composer"):
        msg = gr.Textbox(
            placeholder="Ask for a function…",
            label="Your request",
            show_label=False,
            container=False,
            lines=1,
            max_lines=6,
            autofocus=True,
            scale=20,
            elem_id="msg",
        )
        send = gr.Button("↑", scale=1, min_width=0, elem_id="send")

    gr.HTML('<div class="lab-kicker" style="margin:18px 0 8px">Try one</div>')
    gr.Examples(
        examples=[
            "Write decode_ways(s) returning the number of ways to decode a digit string where A=1..Z=26.",
            "Write next_permutation(nums) that rearranges nums into the lexicographically next greater permutation in place.",
            "Write merge_intervals(intervals) that merges overlapping [start, end] pairs and returns them sorted by start.",
        ],
        inputs=msg,
        label=None,
        elem_id="examples",
    )

    with gr.Accordion("The candidates that were not chosen", open=False, elem_classes="panel"):
        others = gr.Markdown()
    with gr.Accordion("Sampling settings", open=False, elem_classes="panel"), gr.Row():
        k = gr.Slider(2, 16, value=8, step=1, label="K candidates")
        max_tokens = gr.Slider(128, 1024, value=384, step=64, label="Max new tokens")
        temp = gr.Slider(0.0, 1.2, value=0.8, step=0.1, label="Temperature")

    inputs = [msg, chat, k, max_tokens, temp]
    outputs = [chat, verdict, others]

    def busy(k_val):
        working = (
            '<div class="working"><span class="dot"></span>'
            f"Sampling {int(k_val)} candidates, then running them against each other…</div>"
        )
        return gr.update(value="", interactive=False), working

    def idle():
        return gr.update(value="↑", interactive=True)

    for trigger in (msg.submit, send.click):
        trigger(busy, k, [send, verdict]).then(respond, inputs, outputs).then(
            lambda: "", None, msg
        ).then(idle, None, send)

    gr.HTML(
        """<div class="lab-note">
<p>Selection runs inside one model's own K samples. Pooling candidates across different
models was measured and returned <strong>+0.0000</strong>, the same 154 of 164 problems as
the best single model alone, so the pool here is deliberately one model resampled.</p>
<p>K samples are not K times the compute, but the saving belongs to the serving stack rather
than to the method. On an idle RTX PRO 6000, K=8 costs 1.44x a single sample. On this Apple
silicon build it costs about 4x. Neither number transfers to a server already batching
other people's requests.</p>
<p>Sunstone North · <a href="https://lab.sunstonenorth.com/">The Lab</a></p>
</div>"""
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7861)),
        theme=LAB_THEME,
        css=LAB_CSS,
        head=LAB_HEAD,
    )
