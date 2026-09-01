"""Consensus chat on CUDA, for a disposable box.

Same selector as the MLX build, different generation backend, and deliberately
somewhere expendable: this executes model-written code, and the box it runs on
is the blast radius. Nothing here should hold a credential worth stealing.

The child process never inherits this process's environment (see
srt_select/sandbox.py), which matters on a vast instance because CONTAINER_API_KEY
sits in the parent's env.

    source /venv/main/bin/activate
    PYTHONPATH=/root/consensus:/root/consensus/scripts python demo/consensus_chat_cuda.py
"""

from __future__ import annotations

import html
import os
import sys
import time

import gradio as gr
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (HERE, os.path.join(HERE, "scripts"), os.path.join(HERE, "demo")):
    if p not in sys.path:
        sys.path.insert(0, p)

from chat_consensus import choose, code_of  # noqa: E402
from lab_theme import launch_kwargs  # noqa: E402

MODEL = os.environ.get("CONSENSUS_MODEL", "/root/models/Qwen2.5-Coder-1.5B-Instruct")
RUNG = os.environ.get("CONSENSUS_RUNG", "1.5B")
# Banked HumanEval figures for this rung, from artifacts/nla/consensus_demo.json.
BANKED = {"single": 0.2995, "selected": 0.5000, "oracle": 0.7073}

_M = None


def _load():
    global _M
    if _M is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        _M = (model, tok)
    return _M


@torch.inference_mode()
def _generate(message: str, k: int, max_new: int, temp: float) -> list[str]:
    model, tok = _load()
    # transformers 5.x returns a BatchEncoding here unless return_dict is off.
    ids = tok.apply_chat_template(
        [{"role": "user", "content": message}],
        add_generation_prompt=True, return_tensors="pt", return_dict=False,
    ).to(model.device)
    out = model.generate(
        ids,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temp,
        top_p=0.95,
        num_return_sequences=k,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    return [tok.decode(o[ids.shape[1]:], skip_special_tokens=True) for o in out]


def respond(message, history, k, max_new, temp):
    if not message.strip():
        return history, "", ""

    t0 = time.time()
    replies = _generate(message, int(k), int(max_new), float(temp))
    t_gen = time.time() - t0

    t0 = time.time()
    pick, info = choose(message, replies)
    t_sel = time.time() - t0

    if pick is None:
        answer = replies[0]
        note = ('<div class="verdict quiet"><p class="headline">No selection was possible.</p>'
                f'<p class="detail">{html.escape(str(info.get("reason", "unknown")))}.</p></div>')
        others = ""
    else:
        answer = replies[pick]
        ran, size, n_cl = info["ran"], info["cluster_size"], info["clusters"]
        if n_cl == 1:
            head = f"All {size} candidates that ran agreed." if size > 1 else \
                "Only one candidate ran, so there was nothing to compare."
            detail = "The pool computed one answer." if size > 1 else ""
        else:
            head = (f"{size} of {ran} agreed, "
                    f"<span class='split'>the pool split {n_cl} ways</span>.")
            detail = f"{ran - size} computed something different on the same inputs."
        meta = (f"{len(replies)} sampled in {t_gen:.1f}s, selected in {t_sel:.2f}s, "
                f"entry <code>{html.escape(info['entry'])}</code>")
        if len(replies) - ran:
            meta += f", {len(replies) - ran} failed to run"
        note = (f'<div class="verdict"><p class="headline">{head}</p>'
                f'<p class="detail">{detail}</p><p class="meta">{meta}</p></div>')
        others = "\n\n".join(
            f"**Candidate {i + 1}**\n```python\n{code_of(replies[i])[:1400]}\n```"
            for i in range(len(replies)) if i != pick
        )

    history = history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": answer}]
    return history, note, others


EXTRA = """
.verdict { background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--terra);
  border-radius:12px; padding:14px 18px; margin:10px 0; }
.verdict.quiet { border-left-color: var(--line); }
.verdict .headline { font:600 17px/1.4 "Playfair Display",serif; margin:0 0 4px; }
.verdict .detail { color:var(--muted); font-size:14px; margin:0; }
.verdict .split { color:var(--terra); font-weight:600; }
.verdict .meta { color:var(--muted); font-size:12.5px; margin-top:8px; }
#composer { background:var(--panel); border:1px solid var(--line); border-radius:16px;
  padding:6px 6px 6px 14px; gap:8px; align-items:center; margin-top:12px; }
#composer:focus-within { border-color:rgba(176,96,62,.55); box-shadow:0 0 0 4px rgba(176,96,62,.08); }
#msg textarea { background:transparent!important; border:none!important; box-shadow:none!important;
  padding:8px 0!important; font-size:15.5px!important; }
#send { width:38px!important; height:38px!important; min-width:38px!important; border-radius:999px!important;
  padding:0!important; background:var(--terra)!important; color:#fff!important; border:none!important;
  font-size:17px!important; flex:0 0 38px!important; }
#send:disabled { background:#c98f70!important; color:transparent!important; position:relative; }
#send:disabled::after { content:""; position:absolute; inset:0; margin:auto; width:15px; height:15px;
  border-radius:999px; border:2px solid rgba(255,255,255,.35); border-top-color:#fff;
  animation:spin .7s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
#chat { border:1px solid var(--line)!important; border-radius:14px!important; background:var(--panel)!important; }
.working { display:flex; align-items:center; gap:10px; color:var(--muted); font-size:14px; padding:12px 2px 2px; }
.working .dot { width:7px; height:7px; border-radius:999px; background:var(--terra);
  animation:pulse 1.1s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:.25} 50%{opacity:1} }
"""

with gr.Blocks(title="Consensus · Sunstone North Lab") as demo:
    gr.HTML(
        f"""<style>{EXTRA}</style>
<div class="lab-kicker">Sunstone North · Lab · Consensus</div>
<h1>Ask for code. Get the answer the pool agrees on.</h1>
<p class="lab-sub">Every turn samples K candidates and returns the one the others back up.</p>
<p>The choice is made by running all of them on inputs invented from their own signatures and
keeping the largest group that computes the same thing. No test suite, no labels, no benchmark
metadata, just your message and the replies.</p>
<p>Banked on HumanEval for this rung: one sample passes <span class="lab-fig">{BANKED['single']}</span>,
this selector reaches <span class="lab-fig">{BANKED['selected']}</span>, the best candidate in the pool
reaches <span class="lab-fig">{BANKED['oracle']}</span>. Those are benchmark figures, not a claim
about the request you type.</p>
<p><span class="lab-pill">Qwen2.5-Coder-{RUNG}</span><span class="lab-pill">RTX 3060</span><span class="lab-pill">candidates execute on a disposable box</span></p>
"""
    )
    chat = gr.Chatbot(height=430, show_label=False, elem_id="chat",
                      placeholder="Ask for a Python function. Candidates get written, run, and compared.")
    verdict = gr.HTML(elem_id="verdict")
    with gr.Row(elem_id="composer"):
        msg = gr.Textbox(placeholder="Ask for a function…", label="Your request", show_label=False,
                         container=False, lines=1, max_lines=6, autofocus=True, scale=20, elem_id="msg")
        send = gr.Button("↑", scale=1, min_width=0, elem_id="send")

    gr.HTML('<div class="lab-kicker" style="margin:18px 0 8px">Try one</div>')
    gr.Examples(
        examples=[
            "Write decode_ways(s) returning the number of ways to decode a digit string where A=1..Z=26.",
            "Write next_permutation(nums) that rearranges nums into the lexicographically next greater permutation in place.",
            "Write merge_intervals(intervals) that merges overlapping [start, end] pairs and returns them sorted by start.",
        ],
        inputs=msg, label=None, elem_id="examples",
    )
    with gr.Accordion("The candidates that were not chosen", open=False, elem_classes="panel"):
        others = gr.Markdown()
    with gr.Accordion("Sampling settings", open=False, elem_classes="panel"), gr.Row():
        k = gr.Slider(2, 16, value=8, step=1, label="K candidates")
        max_new = gr.Slider(128, 1024, value=384, step=64, label="Max new tokens")
        temp = gr.Slider(0.1, 1.2, value=0.8, step=0.1, label="Temperature")

    def busy(k_val):
        return gr.update(value="", interactive=False), (
            '<div class="working"><span class="dot"></span>'
            f"Sampling {int(k_val)} candidates, then running them against each other…</div>")

    def idle():
        return gr.update(value="↑", interactive=True)

    for trig in (msg.submit, send.click):
        trig(busy, k, [send, verdict]).then(
            respond, [msg, chat, k, max_new, temp], [chat, verdict, others]
        ).then(lambda: "", None, msg).then(idle, None, send)

    gr.HTML("""<div class="lab-note">
<p>Selection runs inside one model's own samples. Pooling candidates across different models
was measured and returned <strong>+0.0000</strong>, the same 154 of 164 problems as the best
single model alone.</p>
<p>Sunstone North · <a href="https://lab.sunstonenorth.com/">The Lab</a></p></div>""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "8080")),
                **launch_kwargs())
