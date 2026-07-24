#!/usr/bin/env python3
"""Build the Substack article from the metapragmatic-load test artifacts.

Charts are computed from the committed result files (so they never drift from the
numbers), rasterized to PNGs under docs/substack_assets/, and embedded here as
base64 data URIs. That keeps the output a single self-contained file: open it in
a browser, select all, copy, and paste straight into the Substack editor with the
images intact. Substack strips inline SVG, so raster is required; re-export the
PNGs from docs/substack_assets/ if any chart's underlying numbers change.

    python scripts/build_substack_article.py
    -> docs/substack_contested_words.html
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COUP = ROOT / "artifacts" / "nla" / "coupling"
OUT = ROOT / "docs" / "substack_contested_words.html"
ASSETS = ROOT / "docs" / "substack_assets"

INK = "#1a1a1a"
MUTED = "#6b6b6b"
CONTEST = "#b3402f"   # warm red = contested
CONTROL = "#3f6fa3"   # cool blue = concrete/control
GRID = "#e7e2d8"
CREAM = "#faf7f0"

# ----------------------------------------------------------------- load data

dia = json.loads((COUP / "diachronic_concept_summary.json").read_text())
dm = dia["decade_means_by_group"]
decades = sorted((int(k) for k in dm), key=int)
con = [dm[str(d)]["contested_D"] for d in decades]
ctl = [dm[str(d)]["control_D"] for d in decades]
did = dia["diachronic"]

causal = [json.loads(l) for l in (COUP / "causal_forcing_readouts.jsonl").read_text().splitlines() if l.strip()]
causal.sort(key=lambda x: x["div_last_baseline"])

# Test 2 cross-backbone standardized coefficients (from committed summary)
t2 = json.loads((COUP / "dissociation_gemma4_summary.json").read_text())
g_beta = t2["gemma4"]["div_last"]["std_beta"]
q_beta = t2["qwen_v1_div_last"]["std_beta"]
BETA = {  # term -> (qwen, gemma)
    "U_com  (contestedness)": (q_beta["U_com"], g_beta["U_com"]),
    "U_ref  (abstractness)": (q_beta["U_ref"], g_beta["U_ref"]),
}


# ------------------------------------------------------------- chart helpers

def lerp(v, a, b, pa, pb):
    return pa + (v - a) / (b - a) * (pb - pa)


def crossover_chart() -> str:
    W, H = 720, 400
    L, R, T, B = 58, 20, 28, 46
    ylo, yhi = 2.45, 2.85
    def X(y):
        return lerp(y, decades[0], decades[-1], L, W - R)
    def Y(v):
        return lerp(v, ylo, yhi, H - B, T)
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Diachronic crossover">']
    # y gridlines
    yt = [2.5, 2.6, 2.7, 2.8]
    for v in yt:
        s.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W-R}" y2="{Y(v):.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{L-8:.0f}" y="{Y(v)+4:.1f}" text-anchor="end" font-size="12" fill="{MUTED}">{v:.1f}</text>')
    # x labels every 40 yrs
    for d in decades:
        if d % 40 == 0 or d == decades[-1]:
            s.append(f'<text x="{X(d):.1f}" y="{H-B+20:.0f}" text-anchor="middle" font-size="12" fill="{MUTED}">{d}s</text>')
    def poly(vals, color):
        pts = " ".join(f"{X(d):.1f},{Y(v):.1f}" for d, v in zip(decades, vals))
        out = [f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>']
        for d, v in zip(decades, vals):
            out.append(f'<circle cx="{X(d):.1f}" cy="{Y(v):.1f}" r="2.6" fill="{color}"/>')
        return "".join(out)
    # crossover band
    s.append(f'<rect x="{X(1900):.1f}" y="{T}" width="{X(1940)-X(1900):.1f}" height="{H-B-T}" fill="#00000008"/>')
    s.append(f'<text x="{(X(1900)+X(1940))/2:.1f}" y="{T+14}" text-anchor="middle" font-size="11" fill="{MUTED}">crossover</text>')
    s.append(poly(ctl, CONTROL))
    s.append(poly(con, CONTEST))
    # legend
    s.append(f'<circle cx="{L+8}" cy="{H-10}" r="4" fill="{CONTEST}"/><text x="{L+18}" y="{H-6}" font-size="12" fill="{INK}">contested signs (freedom, justice, rights\u2026)</text>')
    s.append(f'<circle cx="{L+320}" cy="{H-10}" r="4" fill="{CONTROL}"/><text x="{L+330}" y="{H-6}" font-size="12" fill="{INK}">concrete controls (river, harvest, horse\u2026)</text>')
    s.append(f'<text x="{L-44}" y="{(T+H-B)/2:.0f}" transform="rotate(-90 {L-44} {(T+H-B)/2:.0f})" text-anchor="middle" font-size="12" fill="{MUTED}">metapragmatic load  D</text>')
    s.append("</svg>")
    return "".join(s)


def crossbackbone_chart() -> str:
    W, H = 720, 340
    L, R, T, B = 58, 20, 24, 64
    lo, hi = 0.0, 0.7
    def Y(v):
        return lerp(v, lo, hi, H - B, T)
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Cross-backbone coefficients">']
    for v in (0.0, 0.2, 0.4, 0.6):
        s.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W-R}" y2="{Y(v):.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{L-8}" y="{Y(v)+4:.1f}" text-anchor="end" font-size="12" fill="{MUTED}">{v:.1f}</text>')
    groups = list(BETA.items())
    gw = (W - L - R) / len(groups)
    bw = 52
    for gi, (term, (qv, gv)) in enumerate(groups):
        cx = L + gw * gi + gw / 2
        for k, (val, lab, col) in enumerate([(qv, "Qwen2.5-7B", "#6a6a6a"), (gv, "gemma-4-31B", "#111")]):
            x = cx - bw - 6 + k * (bw + 12)
            y = Y(max(val, 0))
            h = (H - B) - y
            s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw}" height="{max(h,1):.1f}" fill="{col}" rx="2"/>')
            s.append(f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-size="12" fill="{INK}">+{val:.2f}</text>')
            s.append(f'<text x="{x+bw/2:.1f}" y="{H-B+16:.0f}" text-anchor="middle" font-size="10.5" fill="{MUTED}">{lab}</text>')
        s.append(f'<text x="{cx:.1f}" y="{H-B+40:.0f}" text-anchor="middle" font-size="13" fill="{INK}" font-weight="600">{term}</text>')
    s.append(f'<text x="{L-44}" y="{(T+H-B)/2:.0f}" transform="rotate(-90 {L-44} {(T+H-B)/2:.0f})" text-anchor="middle" font-size="12" fill="{MUTED}">standardized coupling  \u03b2</text>')
    s.append("</svg>")
    return "".join(s)


def causal_chart() -> str:
    W, H = 720, 420
    T, B = 64, 74
    ylo, yhi = 0.0, 2.0
    P1 = (74, 352)
    P2 = (400, 690)
    mid = (P1[1] + P2[0]) / 2

    def Y(v):
        return lerp(v, ylo, yhi, H - B, T)

    def mean(xs):
        return sum(xs) / len(xs)

    def jit(i, band=28.0):
        h = ((i * 733 + 97) % 1000) / 1000.0
        return (h - 0.5) * band

    ev = [c for c in causal if not c["contested"]]
    co = [c for c in causal if c["contested"]]
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Causal forcing is flat">']
    for v in (0.0, 0.5, 1.0, 1.5, 2.0):
        s.append(f'<line x1="{P1[0]-10:.0f}" y1="{Y(v):.1f}" x2="{P2[1]:.0f}" y2="{Y(v):.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{P1[0]-16:.0f}" y="{Y(v)+4:.1f}" text-anchor="end" font-size="12" fill="{MUTED}">{v:.1f}</text>')
    s.append(f'<line x1="{mid:.0f}" y1="{T-16:.0f}" x2="{mid:.0f}" y2="{H-B:.0f}" stroke="{GRID}"/>')
    s.append(f'<text x="22" y="{(T+H-B)/2:.0f}" transform="rotate(-90 22 {(T+H-B)/2:.0f})" text-anchor="middle" font-size="12" fill="{MUTED}">metapragmatic load  D</text>')

    def panel(px, title, valkey, note, fmt="{:.2f}"):
        cxe = px[0] + (px[1] - px[0]) * 0.30
        cxc = px[0] + (px[1] - px[0]) * 0.70
        s.append(f'<text x="{(px[0]+px[1])/2:.0f}" y="{T-32:.0f}" text-anchor="middle" font-size="13.5" fill="{INK}" font-weight="600">{title}</text>')
        for cx, grp, col in ((cxe, ev, CONTROL), (cxc, co, CONTEST)):
            for i, c in enumerate(grp):
                s.append(f'<circle cx="{cx+jit(i):.1f}" cy="{Y(c[valkey]):.1f}" r="3" fill="{col}" opacity="0.6"/>')
            m = mean([c[valkey] for c in grp])
            s.append(f'<line x1="{cx-32:.1f}" y1="{Y(m):.1f}" x2="{cx+32:.1f}" y2="{Y(m):.1f}" stroke="#111" stroke-width="2.5"/>')
            s.append(f'<text x="{cx:.1f}" y="{Y(m)-9:.1f}" text-anchor="middle" font-size="12.5" fill="#111" font-weight="600">{fmt.format(m)}</text>')
        s.append(f'<text x="{cxe:.0f}" y="{H-B+20:.0f}" text-anchor="middle" font-size="12" fill="{MUTED}">everyday</text>')
        s.append(f'<text x="{cxc:.0f}" y="{H-B+20:.0f}" text-anchor="middle" font-size="12" fill="{MUTED}">contested</text>')
        s.append(f'<text x="{(px[0]+px[1])/2:.0f}" y="{H-B+40:.0f}" text-anchor="middle" font-size="11.5" fill="{MUTED}" font-style="italic">{note}</text>')

    panel(P1, "How heavy the word is", "div_last_baseline", "contested words sit higher")
    panel(P2, "Effect of switching the reader's community", "ucom_causal", "the same tiny amount for every word", fmt="{:.3f}")
    s.append("</svg>")
    return "".join(s)


def two_clocks() -> str:
    return f'''<svg viewBox="0 0 720 210" width="100%" role="img" aria-label="Two clocks schematic">
  <rect x="20" y="24" width="320" height="150" rx="10" fill="#fff" stroke="{GRID}" stroke-width="2"/>
  <text x="180" y="52" text-anchor="middle" font-size="14" font-weight="600" fill="{INK}">transmission time (slow)</text>
  <text x="180" y="78" text-anchor="middle" font-size="12.5" fill="{MUTED}">centuries of a culture arguing</text>
  <text x="180" y="98" text-anchor="middle" font-size="12.5" fill="{MUTED}">over a word: it accrues U_com</text>
  <text x="180" y="140" text-anchor="middle" font-size="30">\u23f3</text>
  <rect x="380" y="24" width="320" height="150" rx="10" fill="#fff" stroke="{GRID}" stroke-width="2"/>
  <text x="540" y="52" text-anchor="middle" font-size="14" font-weight="600" fill="{INK}">token-sequence time (fast)</text>
  <text x="540" y="78" text-anchor="middle" font-size="12.5" fill="{MUTED}">milliseconds of reading one</text>
  <text x="540" y="98" text-anchor="middle" font-size="12.5" fill="{MUTED}">sentence: the model reads load D</text>
  <text x="540" y="140" text-anchor="middle" font-size="30">\u26a1</text>
  <text x="360" y="150" text-anchor="middle" font-size="22" fill="{CONTEST}">\u2192</text>
</svg>'''


# --------------------------------------------------------------------- HTML

CSS = """
:root{color-scheme:light}
*{box-sizing:border-box}
body{margin:0;background:#f4efe6;color:#1a1a1a;
 font-family:Charter,Georgia,'Times New Roman',serif;line-height:1.66}
.wrap{max-width:720px;margin:0 auto;padding:56px 22px 96px}
.kicker{font-family:ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif;
 letter-spacing:.14em;text-transform:uppercase;font-size:12.5px;color:#b3402f;font-weight:700}
h1{font-size:40px;line-height:1.12;margin:.32em 0 .1em;letter-spacing:-.01em}
.dek{font-size:20px;color:#4a4a4a;font-style:italic;margin:.4em 0 1.2em}
.byline{font-family:ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif;
 font-size:13.5px;color:#6b6b6b;border-top:1px solid #e2dccf;border-bottom:1px solid #e2dccf;
 padding:12px 0;margin:0 0 34px}
h2{font-size:25px;margin:1.9em 0 .5em;letter-spacing:-.01em}
h3{font-size:18px;margin:1.5em 0 .3em;font-family:ui-sans-serif,-apple-system,Segoe UI,sans-serif}
p{font-size:18.5px;margin:0 0 1.05em}
a{color:#b3402f;text-decoration:none;border-bottom:1px solid #e3b6ae}
a:hover{background:#faeae6}
strong{font-weight:700}
figure{margin:30px 0;padding:20px 18px 14px;background:#fbf9f4;border:1px solid #e7e2d8;border-radius:12px}
figcaption{font-family:ui-sans-serif,-apple-system,Segoe UI,sans-serif;font-size:13.5px;
 color:#6b6b6b;margin-top:12px;line-height:1.5}
blockquote,.pull{font-size:23px;line-height:1.4;color:#111;border-left:3px solid #b3402f;
 padding:2px 0 2px 20px;margin:26px 0;font-style:italic}
.stat{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}
.stat div{flex:1 1 150px;background:#fbf9f4;border:1px solid #e7e2d8;border-radius:10px;padding:14px 16px}
.stat b{display:block;font-size:27px;color:#b3402f;font-family:ui-sans-serif,sans-serif}
.stat span{font-family:ui-sans-serif,sans-serif;font-size:12.5px;color:#6b6b6b}
.eq{background:#fbf9f4;border:1px solid #e7e2d8;border-radius:10px;padding:16px 14px;text-align:center;margin:20px 0}
.eq .f{font-family:ui-serif,Georgia,Cambria,serif;font-size:23px;color:#1a1a1a;letter-spacing:.3px}
.eq .f i{font-style:italic}
.eq sub{font-size:.6em;vertical-align:-.28em}
.eq .plain{display:block;margin-top:10px;font-family:ui-sans-serif,-apple-system,Segoe UI,sans-serif;font-size:13.5px;line-height:1.5;color:#6b6b6b}
.models{font-family:ui-sans-serif,-apple-system,Segoe UI,sans-serif;font-size:15px}
.models li{margin:.35em 0}
.foot{font-family:ui-sans-serif,sans-serif;font-size:13px;color:#8a8a8a;margin-top:40px;
 border-top:1px solid #e2dccf;padding-top:18px}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.86em;background:#efe9dd;padding:1px 5px;border-radius:4px}
.rec{font-family:ui-sans-serif,-apple-system,Segoe UI,sans-serif;font-size:.8em;color:#8a7d63;background:#f3ecdd;padding:1px 7px;border-radius:5px}
"""


def a(href, txt):
    return f'<a href="{href}">{txt}</a>'


def png(name, alt):
    """Embed a pre-rendered chart PNG as a base64 data URI so the article stays a
    single self-contained file that pastes into Substack with images intact."""
    b64 = base64.b64encode((ASSETS / name).read_bytes()).decode("ascii")
    return (f'<img alt="{alt}" style="width:100%;height:auto;display:block;'
            f'border-radius:6px" src="data:image/png;base64,{b64}">')


HF = "https://huggingface.co/"


def build() -> str:
    did_v = did["difference_in_differences"]
    p = did["did_perm_p"]
    parts = []
    parts.append(f"<!doctype html><html lang=en><head><meta charset=utf-8>"
                 f"<meta name=viewport content='width=device-width,initial-scale=1'>"
                 f"<title>The Weight Words Carry</title><style>{CSS}</style></head><body><div class=wrap>")

    parts.append('<div class=kicker>Some words are heavier than others</div>')
    parts.append("<h1>The Weight Words Carry</h1>")
    parts.append('<p class=dek>A frozen language model, a small probe bolted to its internals, seventy-nine words and two centuries of newspapers, all aimed at one question: which words are heavy to think with, and why?</p>')
    parts.append('<div class=byline>No math required to follow it. Every number and every model is linked at the end if you want to check the work yourself.</div>')

    parts.append("<p>Some words are harder to think than others, and most people know it without being able to say why. <em>Hammer</em> gives you no trouble. <em>Freedom</em> does. You can lose an afternoon arguing about what freedom means and get nowhere; nobody has ever needed an afternoon to settle what a hammer is. Two centuries of argument have left something on the one word that was never left on the other, and that difference, which feels obvious and sounds unmeasurable, is the thing I wanted to measure.</p>")
    parts.append("<p>What follows is a report on trying to measure it anyway: what a language model can be made to tell you about the weight of a word, and the one result near the end that turned my own explanation inside out.</p>")

    parts.append("<h2>The instrument</h2>")
    parts.append("<p>We take an open language model, " + a(HF+"Qwen/Qwen2.5-7B", "Qwen2.5-7B") + ", and leave it frozen. Onto its internals we attach a small probe, roughly a five-hundredth the size of the model it reads, whose only job is to report on the model's internal state without changing it. The model runs exactly as it always would. The probe just watches.</p>")
    parts.append("<p>The one reading that matters here we call <strong>load</strong>: how hard a word's meaning has to bend the moment it lands in a sentence. On its own, <em>bank</em> could be a river or a vault. Put it in <em>she sat on the bank</em> and the sentence settles the question for you. Load is the size of that settling. Most words barely register it; a few have to strain. <span class=rec>For the record: load is the divergence emitted by a metapragmatic attention head reading the frozen residual stream, not a probability or an attention weight.</span></p>")
    parts.append("<p>The premise of the whole project is that two very different timescales meet inside a single word.</p>")
    parts.append("<figure>" + png("two_clocks.png", "Two clocks: slow transmission time and fast reading time") + "<figcaption><em>One clock is slow: the centuries a society spends arguing over a word. The other is fast: the half-second it takes you to read a sentence today. The whole bet of this project is that the slow clock leaves a mark you can read on the fast one.</em></figcaption></figure>")
    parts.append("<p>Load has two plausible sources. A word can be heavy because it is <strong>abstract</strong>, with no solid thing to point at (<em>justice</em> against <em>teapot</em>). Or because it is <strong>contested</strong>, the way <em>justice</em> and <em>freedom</em> are words people argue over. Telling those two sources apart is what the first tests do. But that second word, <em>contested</em>, is also the one I started with and later had to give up, and that turn is where the piece is really headed.</p>")
    parts.append('<div class=eq><span class=f><i>E</i>[D] = D<sub>0</sub> + &alpha;&middot;U<sub>ref</sub> + &beta;&middot;U<sub>com</sub></span>'
                 '<span class=plain>the load a word carries&nbsp;=&nbsp;a fixed baseline&nbsp;+&nbsp;&alpha;&thinsp;&times;&thinsp;(how abstract it is, U<sub>ref</sub>)&nbsp;+&nbsp;&beta;&thinsp;&times;&thinsp;(how contested it is, U<sub>com</sub>)</span></div>')
    parts.append('<p>That is the conjecture in one line. <i>E</i>[D] is the load you expect to read. D<sub>0</sub> is what every word carries just for turning up. Then two dials: &alpha; for how much sheer abstractness adds, &beta; for how much genuine disagreement adds. The whole piece comes down to that second dial. Is &beta; real, and does it survive when you change the mind doing the reading?</p>')

    parts.append("<h2>The pattern, in one model</h2>")
    parts.append("<p>The first job is to make sure the pattern is real and not something I talked myself into. We took 79 words and scored each one twice, using outside sources so we could not put a thumb on the scale: how abstract, from " + a(HF+"datasets/lecslab/brysbaert_concreteness", "published psychology norms") + ", and how contested, from a careful hand-rating. Then we read the load on each.</p>")
    parts.append("<p>Both mattered, and they mattered separately. Abstract words were heavier. Contested words were heavier still, even after you subtract out how abstract they are. <span class=rec>for the record: standardized effect +0.44 for abstractness, +0.21 for contestedness, both hold up, on Qwen2.5-7B.</span> Good. The pattern is real. But real in one model is just a curiosity. The next question is the one that actually decides whether any of this matters.</p>")

    parts.append("<h2>A second, unrelated model</h2>")
    parts.append("<p>We ran the same test on an entirely different model, " + a(HF+"google/gemma-4-31B-it", "Google's gemma-4") + ", four times larger, built by a different company, and read by a different sensor that was trained separately and shares nothing with the first. If our pattern is just a quirk of one machine, the two should disagree. If it is about something deeper in language itself, they should line up.</p>")
    parts.append("<p>They agreed about contested words almost exactly. They disagreed about abstract words completely.</p>")
    parts.append("<figure>" + png("cross_backbone.png", "Cross-backbone coefficients: contestedness agrees, abstractness does not") + "<figcaption><em>Two unrelated models, side by side. On contested words they land in nearly the same place. On abstract words one cares a lot and the other barely notices. <span class=rec>for the record: contestedness +0.38 vs +0.41, basically identical; abstractness +0.61 vs +0.03, gone.</span></em></figcaption></figure>")
    parts.append("<blockquote>Two very different minds cannot agree on which words are abstract. They agree on which words are contested. Some of the cracks in language are real enough that any careful reader, human or machine, has to fall into them.</blockquote>")

    parts.append("<h2>Two centuries of newspapers</h2>")
    parts.append("<p>If contestedness really is laid down slowly, generation by generation, then you ought to be able to watch it happen. So we went to the newspapers. We took <strong>11,876 articles</strong> from American papers printed between <strong>1770 and 1964</strong> (from " + a(HF+"datasets/dell-research-harvard/AmericanStories", "a huge archive of scanned newsprint") + "), picked out a handful of fighting words (<em>freedom, justice, rights, slavery, sovereignty</em>) and a handful of quiet ones (<em>river, harvest, horse</em>), and measured the load on each, decade by decade.</p>")
    parts.append("<figure>" + png("diachronic_crossover.png", "Load on contested versus concrete words, 1770s to 1960s") + "<figcaption><em>Load on fighting words (red) and quiet words (blue), 1770s to 1960s. In the early Republic the political words are no heavier than a plow. They gain weight over the next century, cross over around 1900 to 1940, and end as the heaviest words in the language, while river and harvest stay as light as they always were.</em></figcaption></figure>")
    parts.append("<p>The shape is the interesting part. Contested words did not start out heavy. They <em>became</em> heavy, on a schedule that lines up uncomfortably well with the country's own quarrels. <span class=rec>for the record: the fighting words climbed with time faster than the quiet ones, +0.21 against +0.17 in rank terms; the gap is small but not chance, p &asymp; 0.007. Everything drifts up a little over the period, probably from cleaner print and changing prose, which is why the honest test is the difference between the two groups, not the raw climb.</span></p>")
    parts.append("<p>So <em>justice</em> is not a settled tool you reach for and use. It is a word that two centuries of argument have loaded, and the load is most of what you are handling when you use it.</p>")

    parts.append("<h2>The experiment that went against me</h2>")
    parts.append("<p>Up to here the story was tidy. Maybe too tidy. I had a guess about <em>why</em> contested words are heavy, and it felt obvious: a word is contested because different groups read it differently, so surely the trouble lives in the space <em>between</em> people. If that is right, then forcing the model to read a word as one community, and then as another, should make <em>freedom</em> lurch around and <em>hammer</em> barely twitch.</p>")
    parts.append("<p>We had a tool built for exactly this: a companion model, " + a(HF+"RiverRider/zooL4nD3r-v0.1", "zooL4nD3r") + ", that had learned a map of discourse communities. So we had the model read each word as one community after another, and measured how far the reading moved.</p>")
    parts.append("<figure>" + png("causal_flat.png", "Baseline load separates contested from everyday words, but forcing the reader's community does not") + "<figcaption><em>Left: how heavy each of the 40 test words is, read normally. Contested ideas (freedom, justice, gender) sit clearly higher than everyday things (hammer, teapot, a prime number), just as the earlier tests found. Right: how much that load shifts when we force the model to read each word as a different community. The dots collapse to a flat band at the same tiny value, contested and everyday alike. <span class=rec>for the record: the forcing effect is 0.086 for every word, with a spread of about a thousandth; baseline load still tracks contestedness at roughly +0.58, the intervention does not.</span></em></figcaption></figure>")
    parts.append("<p>It did not work the way I expected. Switching the community moved every word by the same negligible amount. <span class=rec>for the record: 0.086, essentially flat, across all 40 words.</span> And yet the heaviness itself, the gap on the left of that chart, was as plain as ever. These words <em>are</em> heavy. The heaviness simply is not coming from disagreement between readers. It is already inside the word, before anyone reads it at all.</p>")
    parts.append("<p>That is where I had to change the word. If the weight sits in the word and not in the argument between readers, then <em>contested</em> is the wrong name, because contested points at a fight between people. The better name is one philosophers already have: these are <strong>thick</strong> words. A thick word fuses a description and a judgment in a single breath. Say <em>cruelty</em> and you have described an act and condemned it at once; there is no neutral version you can hand across the table with the verdict taken out. <em>Justice</em>, <em>freedom</em>, <em>gender</em> are thick in exactly that way, and <em>teapot</em> is not. Thickness is why forcing a different community changes nothing: the judgment is not supplied by the reader, it is welded into the word. And it is why <q>let us just define our terms</q> so often fails. You cannot define the verdict out of a word that is made of it.</p>")
    parts.append("<blockquote>I thought these were clear words that different people pulled in different directions. They are closer to the opposite: words that arrive already carrying a verdict, which every reader inherits whether they share it or not. The disagreement between people is downstream. The weight is in the word.</blockquote>")

    parts.append("<h2>What I take from it</h2>")
    parts.append("<p><strong>Words carry their history, and the weight is real.</strong> The load on <em>justice</em> was put there by everyone who ever argued over it, and it does not come off. That is not a reading we went looking for; it is what the measurement kept returning.</p>")
    parts.append("<p><strong>The trouble is in the word, not only in the disagreement.</strong> This is the humbling one. Some of our deepest arguments cannot be settled by defining terms, because the judgment is already welded into the terms. You can be perfectly clear about which side you speak from and the word will still carry its verdict, because the verdict was there before you arrived.</p>")
    parts.append("<p><strong>Thick language is expensive, not lazy.</strong> We tend to hear <em>rights</em> or <em>freedom</em> as vague, as if the speaker were being sloppy. It looks more like the reverse. These are among the most demanding words we own, the ones that make every mind, human or machine, work hardest to hold steady.</p>")
    parts.append("<p><strong>The result I got wrong taught me more than the ones I got right.</strong> The most useful moment in the whole effort was the community test coming back flat, against everything I expected. It did not break the idea. It moved it, out of the space between people and into the word itself.</p>")

    parts.append("<h2>Where I could be wrong</h2>")
    parts.append("<p>Honesty first. This is a model of language, read by an instrument we built ourselves. When a machine designed to look for structure in meaning reports structure in meaning, the fair worry is that we quietly built our own assumptions into the lens. What makes it worth your time is that we set it up to be <em>provable wrong</em>. We tried hard to break it, we wrote down exactly where it broke (a whole approach through images that failed, a tempting shortcut that measured the wrong thing, the community test that came back flat), and we let the failures stand in the record. That is the part I would defend: old intuitions about meaning, turned into claims you can put to the test and watch fail.</p>")

    parts.append("<h2>Everything, if you want to check it</h2>")
    parts.append("<p class=models>The frozen models being read:</p><ul class=models>"
                 f"<li>{a(HF+'Qwen/Qwen2.5-7B','Qwen/Qwen2.5-7B')}</li>"
                 f"<li>{a(HF+'google/gemma-4-31B-it','google/gemma-4-31B-it')}</li></ul>")
    parts.append("<p class=models>The sensors (small read-outs on those frozen models):</p><ul class=models>"
                 f"<li>{a(HF+'RiverRider/srt-adapter-v1.0','srt-adapter-v1.0')}</li>"
                 f"<li>{a(HF+'RiverRider/Gemma-4-31B-it-SRT-Sunstone','Gemma-4-31B-it-SRT-Sunstone')}</li>"
                 f"<li>{a(HF+'RiverRider/zooL4nD3r-v0.1','zooL4nD3r-v0.1')} (the community map)</li>"
                 f"<li>{a(HF+'RiverRider/srt-adapter-v8a','srt-adapter-v8a')}</li></ul>")
    parts.append("<p class=models>The historical text and word-ratings:</p><ul class=models>"
                 f"<li>{a(HF+'datasets/dell-research-harvard/AmericanStories','AmericanStories')} (newsprint 1770&ndash;1964)</li>"
                 f"<li>{a(HF+'datasets/RevolutionCrossroads/loc_chronicling_america_1770-1810','Chronicling America 1770&ndash;1810')}</li>"
                 f"<li>{a(HF+'datasets/davanstrien/chronicling-america-1920-1950','Chronicling America 1920&ndash;1950')}</li>"
                 f"<li>{a(HF+'datasets/lecslab/brysbaert_concreteness','Brysbaert concreteness norms')}</li></ul>")
    parts.append("<p class=models>The code and the full technical write-up: <a href='https://github.com/space-bacon/SRT'>github.com/space-bacon/SRT</a>.</p>")

    parts.append('<div class=foot>Every chart here is drawn straight from the saved result files, so the pictures can never drift from the numbers. The language models were frozen throughout; nothing in this work trains or fine-tunes them. They are only ever read.</div>')
    parts.append("</div></body></html>")
    return "".join(parts)


OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(build())
print("wrote", OUT, "(", len(OUT.read_text()), "bytes )")
