"""Consensus, read from banked runs, as a grid you can take in at a glance.

The point is that eight candidates look alike and some are wrong, so the eight
have to be visible at the same time. Everything is laid out to fit one screen:
rung cards across the top, the pool as a grid below, and a three-step reveal that
colours the grid first by what the candidates compute and then by whether they
were right.

Nothing executes. Clustering and pass marks come from
artifacts/nla/consensus_demo.json, precomputed by scripts/bank_consensus_demo.py.

    PYTHONPATH=. python demo/consensus_banked_app.py
"""

from __future__ import annotations

import html
import json
import os
import sys

import gradio as gr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "demo"))

from lab_theme import launch_kwargs  # noqa: E402

BANK = json.load(open(os.path.join(HERE, "artifacts", "nla", "consensus_demo.json")))
RUNGS = list(BANK["rungs"])
PROBLEMS = BANK["problems"]
K = BANK["k"]

FILTERS = [
    "the selector picked a passing answer",
    "the selector picked a failing one",
    "the pool agreed unanimously",
    "everything",
]


def matches(e: dict, f: str) -> bool:
    if f == "everything":
        return True
    if not e.get("ok"):
        return False
    if f == FILTERS[0]:
        return e["n_clusters"] > 1 and bool(e["pick_passed"])
    if f == FILTERS[1]:
        return e["any_passed"] and not e["pick_passed"]
    return e["n_clusters"] == 1


def choices(rung: str, f: str) -> list[str]:
    ents = BANK["rungs"][rung]["entries"]
    out = []
    for i, p in enumerate(PROBLEMS):
        e = ents[i]
        if not matches(e, f):
            continue
        n = sum(e["passed"])
        shape = "agreed" if e.get("ok") and e["n_clusters"] == 1 else (
            f"split {e['n_clusters']}" if e.get("ok") else "no selection")
        out.append(f"{p['task_id']}  ·  {shape}  ·  {n}/{K} pass")
    return out or ["(nothing matches)"]


def rung_cards(active: str) -> str:
    cards = []
    for r in RUNGS:
        d = BANK["rungs"][r]
        s, sel, o = d["single_sample_pass"], d["selected_pass"], d["oracle_pass"]
        cards.append(
            f"<button class='rung{' on' if r == active else ''}' data-r='{r}'>"
            f"<span class='rname'>{r}</span>"
            f"<span class='bar'><i class='b-o' style='width:{o * 100:.0f}%'></i>"
            f"<i class='b-s' style='width:{sel * 100:.0f}%'></i>"
            f"<i class='b-1' style='width:{s * 100:.0f}%'></i></span>"
            f"<span class='rnum'>{s:.2f} <b>{sel:.2f}</b> {o:.2f}</span></button>"
        )
    return f"<div class='rungs'>{''.join(cards)}</div>"


def render(rung: str, choice: str) -> str:
    task = (choice or "").split("  ·  ")[0].strip()
    i = next((j for j, p in enumerate(PROBLEMS) if p["task_id"] == task), 0)
    p, e = PROBLEMS[i], BANK["rungs"][rung]["entries"][i]

    lines = p["prompt"].strip().split("\n")
    sig = next((ln for ln in lines if ln.startswith("def ")), lines[0])[:110]

    if not e.get("ok"):
        line = (f"<span class='muted'>No selection was possible: "
                f"{html.escape(str(e.get('reason', '')))}.</span>")
        tiles = ""
    else:
        size, n_cl, ran = e["cluster_size"], e["n_clusters"], e["ran"]
        n_pass = sum(e["passed"])
        shape = f"{size} of {ran} agreed" if n_cl > 1 else f"all {size} that ran agreed"
        split = f", <b class='t'>split {n_cl} ways</b>" if n_cl > 1 else ""
        if e["pick_passed"]:
            truth = "<b class='ok'>the selected answer passes</b>"
        elif e["any_passed"]:
            truth = (f"<b class='no'>the selected answer fails</b>, though {n_pass} "
                     "in the pool pass")
        else:
            truth = "<b class='no'>none of them pass</b>"
        line = f"{shape}{split} · {truth} · <span class='muted'>{n_pass}/{K} pass</span>"

        cells = []
        for j, code in enumerate(e["candidates"]):
            g, ok = e["groups"][j], e["passed"][j]
            pick = " data-pick='1'" if j == e["pick"] else ""
            cells.append(
                f"<div class='tile' data-g='{g}' data-pass='{int(ok)}'{pick}>"
                f"<div class='th'><span class='ix'>{j + 1}</span>"
                f"<span class='gp'>{'did not run' if g < 0 else 'group ' + str(g + 1)}</span>"
                f"<span class='pf'></span></div>"
                f"<pre>{html.escape(code[:1200])}</pre></div>"
            )
        tiles = f"<div class='pool' data-reveal='1'>{''.join(cells)}</div>"

    return (f"<div class='sig'><code>{html.escape(sig)}</code></div>"
            f"<div class='verdict'>{line}</div>{tiles}")


CSS = """
.compact h1 { font-size:30px !important; margin:0 0 4px !important; }
.compact p { margin:3px 0; font-size:14.5px; }

.rungs { display:grid; grid-template-columns:repeat(6,1fr); gap:8px; margin:14px 0 4px; }
.rung { background:#fff; border:1px solid #e7dfd5; border-radius:12px;
  padding:9px 10px 8px; cursor:pointer; text-align:left; font-family:inherit; transition:.12s; }
.rung:hover { border-color:#d9cec2; }
.rung.on { border-color:#b0603e; box-shadow:0 0 0 3px rgba(176,96,62,.10); }
.rung .rname { display:block; font:600 15px/1.2 "Playfair Display",serif; color:#4a423b; }
.rung .bar { display:block; position:relative; height:5px; background:#f4eee6;
  border-radius:99px; margin:7px 0 6px; }
.rung .bar i { position:absolute; left:0; top:0; height:100%; border-radius:99px; }
.rung .b-o { background:#e3d3c6; } .rung .b-s { background:#b0603e; } .rung .b-1 { background:#8b6b58; }
.rung .rnum { display:block; font:11.5px/1 ui-monospace,Menlo,monospace; color:#8b8178; }
.rung .rnum b { color:#b0603e; }

.sig { margin:10px 0 2px; }
.sig code { font-size:12.5px !important; color:#8b8178 !important; }
.verdict { font-size:14.5px; color:#4a423b; margin:6px 0 10px; }
.verdict .t { color:#b0603e; } .verdict .ok { color:#3d7a44; } .verdict .no { color:#a04533; }
.verdict .muted { color:#8b8178; }

.reveal { display:flex; gap:6px; margin:2px 0 10px; }
.reveal button { background:#f4eee6; border:1px solid #e7dfd5; color:#8b8178;
  border-radius:999px; padding:5px 13px; font:600 11px/1 Inter,sans-serif; letter-spacing:.1em;
  text-transform:uppercase; cursor:pointer; }
.reveal button.on { background:#b0603e; border-color:#b0603e; color:#fff; }

.pool { display:grid; grid-template-columns:repeat(4,1fr); gap:9px; }
.tile { position:relative; background:#fff; border:1px solid #e7dfd5;
  border-radius:10px; overflow:hidden; cursor:pointer; max-height:132px; transition:.14s; }
.tile:hover { border-color:#d9cec2; }
.tile .th { display:flex; align-items:center; gap:7px; padding:5px 9px; background:#f4eee6;
  border-bottom:1px solid #e7dfd5; font-size:11px; color:#8b8178; }
.tile .ix { font-weight:600; color:#4a423b; }
.tile .gp { opacity:0; transition:.14s; }
.tile .pf { margin-left:auto; width:8px; height:8px; border-radius:99px; background:transparent; }
.tile pre { margin:0 !important; border:none !important; border-radius:0 !important;
  font-size:10.5px !important; line-height:1.45 !important; padding:8px 10px !important;
  background:#fff !important; overflow:hidden; }

.pool[data-reveal="2"] .tile .gp, .pool[data-reveal="3"] .tile .gp { opacity:1; }
.pool[data-reveal="2"] .tile[data-g="0"], .pool[data-reveal="3"] .tile[data-g="0"] { border-color:#b0603e; }
.pool[data-reveal="2"] .tile[data-g="1"], .pool[data-reveal="3"] .tile[data-g="1"] { border-color:#5b7f9e; }
.pool[data-reveal="2"] .tile[data-g="2"], .pool[data-reveal="3"] .tile[data-g="2"] { border-color:#9a8348; }
.pool[data-reveal="2"] .tile[data-g="3"], .pool[data-reveal="3"] .tile[data-g="3"] { border-color:#7d6b8f; }
.pool[data-reveal="2"] .tile[data-g="-1"], .pool[data-reveal="3"] .tile[data-g="-1"] { opacity:.4; }

.pool[data-reveal="3"] .tile[data-pass="1"] .pf { background:#3d7a44; }
.pool[data-reveal="3"] .tile[data-pass="0"] .pf { background:#a04533; }
.pool[data-reveal="3"] .tile[data-pick="1"]::after { content:"selected"; position:absolute;
  right:8px; bottom:7px; background:#b0603e; color:#fff; border-radius:99px;
  padding:1px 9px; font:600 10px/1.6 Inter,sans-serif; }

.tile.open { grid-column:1/-1; max-height:none; cursor:default; }
.tile.open pre { overflow:auto; max-height:50vh; }
.pool.has-open .tile:not(.open) { opacity:.25; max-height:30px; }
.pool.has-open .tile:not(.open) pre { display:none; }

/* gr.Textbox(visible=False) is dropped from the DOM, so the click handler cannot
   reach it. Render it and hide it here instead. */
#rungsel { display:none !important; }
"""

JS = """
if (!window.__consensusBound) {
  window.__consensusBound = true;
  document.addEventListener('click', function (ev) {
    var rung = ev.target.closest('.rung');
    if (rung) {
      var box = document.querySelector('#rungsel textarea, #rungsel input');
      if (box) {
        box.value = rung.dataset.r;
        box.dispatchEvent(new Event('input', { bubbles: true }));
      }
      return;
    }
    var rv = ev.target.closest('.reveal button');
    var pool = document.querySelector('.pool');
    if (rv) {
      document.querySelectorAll('.reveal button').forEach(function (b) { b.classList.remove('on'); });
      rv.classList.add('on');
      if (pool) pool.dataset.reveal = rv.dataset.v;
      return;
    }
    if (!pool) return;
    var tile = ev.target.closest('.tile');
    if (tile) {
      var was = tile.classList.contains('open');
      pool.querySelectorAll('.tile').forEach(function (t) { t.classList.remove('open'); });
      pool.classList.toggle('has-open', !was);
      if (!was) tile.classList.add('open');
    } else {
      pool.querySelectorAll('.tile').forEach(function (t) { t.classList.remove('open'); });
      pool.classList.remove('has-open');
    }
  });
}
"""

with gr.Blocks(title="Consensus · Sunstone North Lab") as demo:
    gr.HTML(
        f"""<div class="compact">
<div class="lab-kicker">Sunstone North · Lab · Consensus</div>
<h1>They all sound right. They are not all right.</h1>
<p>A coding model answers the same question {K} times. Run the answers on inputs invented from
their own signatures, keep the largest group that agrees, and you pick better than any single
sample, with no tests and no reference.</p>
<p style="color:#8b8178">Nothing here executes. Every group and every pass mark was computed
once against HumanEval's own tests, so you can see where the selector is wrong as well as right.</p>
</div>"""
    )

    cards = gr.HTML(rung_cards("7B"))
    rungsel = gr.Textbox(value="7B", elem_id="rungsel", container=False)

    with gr.Row():
        filt = gr.Dropdown(FILTERS, value=FILTERS[0], label="Show", scale=2)
        prob = gr.Dropdown(choices("7B", FILTERS[0]), value=choices("7B", FILTERS[0])[0],
                           label="Problem", scale=3)

    gr.HTML("""<div class="reveal">
<button data-v="1" class="on">as written</button>
<button data-v="2">what they compute</button>
<button data-v="3">the truth</button></div>""")

    body = gr.HTML()

    gr.HTML(f"""<div class="lab-note">
<p>Selection runs inside one model's own samples. Pooling across different models was measured
and returned <strong>+0.0000</strong>. These pools are the first {K} of a K=32 run, so figures
differ slightly from the paper's separate K=8 run.</p>
<p>Sunstone North · <a href="https://lab.sunstonenorth.com/">The Lab</a></p></div>""")

    def repick(r, f):
        ch = choices(r, f)
        return rung_cards(r), gr.update(choices=ch, value=ch[0]), render(r, ch[0])

    rungsel.change(repick, [rungsel, filt], [cards, prob, body])
    filt.change(repick, [rungsel, filt], [cards, prob, body])
    prob.change(render, [rungsel, prob], body)
    demo.load(render, [rungsel, prob], body)

if __name__ == "__main__":
    kw = launch_kwargs()
    # css is injected after Gradio's own sheet, so it wins; head loads before it.
    # Scripts only execute from head.
    kw["css"] = kw.get("css", "") + CSS
    kw["head"] = kw.get("head", "") + f"<script>{JS}</script>"
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.environ.get("PORT", "7865")), **kw)
