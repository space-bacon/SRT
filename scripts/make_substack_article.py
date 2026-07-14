"""Assemble the Substack response article as a single HTML file with
embedded (base64) images, ready to open in a browser, select-all, copy,
and paste into the Substack editor.

    python scripts/make_substack_article.py --out docs/substack_instrument_panel.html
"""
from __future__ import annotations

import argparse
import base64
import os

IMAGES = {
    "hero": "artifacts/marketing/srt_hero.png",
    "instruments": "artifacts/marketing/srt_instruments.png",
    "trace": "artifacts/explainers/11_token_trace.png",
    "dangerous": "artifacts/marketing/srt_dangerous_moment.png",
    "stereo": "artifacts/nla/gemma4/stereo/stereo_figure.png",
    "sunstone": "demo/cross_modal_space/promo/sunstone_hero.png",
    "numbers": "artifacts/marketing/srt_numbers.png",
}


def b64(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def fig(src: str, caption: str) -> str:
    return (f'<figure style="margin:28px 0;text-align:center;">'
            f'<img src="{src}" style="width:100%;max-width:100%;height:auto;" />'
            f'<figcaption style="font-size:0.85em;color:#666;margin-top:6px;">{caption}</figcaption>'
            f"</figure>")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="docs/substack_instrument_panel.html")
    args = p.parse_args()

    im = {k: b64(v) for k, v in IMAGES.items()}

    body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>The Missing Instrument Panel</title></head>
<body style="max-width:720px;margin:0 auto;font-family:Georgia,serif;line-height:1.6;color:#1a1a1a;padding:24px;">

<h1>The Missing Instrument Panel</h1>
<h3 style="font-weight:normal;color:#555;">A response to Neha Kabra and Mila Agius. Organizations cannot watch for
transitions they cannot see, and today&rsquo;s models emit no mode telemetry. SRT is a working attempt to build the gauges.</h3>

{fig(im['hero'], '')}

<p>Neha Kabra and Mila Agius end
<a href="https://nehakabra1.substack.com/p/organizations-keep-trying-to-solve">their piece</a>
with the sentence that matters most: &ldquo;The most dangerous moment is not when the AI system is clearly
failing. It is when the system has quietly changed modes, and no one is responsible for managing the handoff.&rdquo;</p>

<p>I want to take that sentence more literally than they may have intended.</p>

<p>Aviation&rsquo;s most basic safety artifact is the black box. The industry decided, decades ago, that every
aircraft would carry a recorder, so that when something went wrong you could know exactly what the machine had
been doing. Today&rsquo;s AI deployments invert that arrangement. The machine <em>is</em> the black box, and the
organization is flying from inside it. No mode annunciator. No rate gauges. No stall warning. No recorder. The
crew infers the aircraft&rsquo;s state from how the landing feels.</p>

<p>The aviation analogy in the article is usually read as an argument about the organization: decision rights,
escalation paths, mode discipline. All true, and all necessary. But aviation did not get safer through discipline
alone. It got safer because the aircraft itself was instrumented. The pilot does not deduce a mode change from
the passengers&rsquo; faces. The aircraft reports its state, continuously, in a form the crew and the investigators
can read.</p>

<p>A language model reports nothing. It does not announce when its interpretation of context has shifted, when
its processing has entered a different regime, or when its own outputs have begun feeding back into its behavior.
The organization sees the final output only, and so, when unstable or conflicting outputs appear, it reaches for
the only levers visible from outside: more rules, more review layers, more scale. That is governing in the dark,
and as the article argues, more intelligence does not fix it. The gap is an information asymmetry, and it lives
inside the forward pass.</p>

<h2>Building the gauges</h2>

<p>SRT, the Semiotic-Reflexive Transformer, is a working attempt to build the instrument panel. It is a small
adapter, roughly 0.17 percent of the parameters of the frozen base model it attaches to, that emits real-time
signals during generation. The correspondence to the article&rsquo;s vocabulary is direct.</p>

{fig(im['instruments'], 'The mapping, one gauge at a time.')}

<p>A <strong>regime label</strong> reports which mode of interpretive processing the model is currently in: the
mode annunciator. A <strong>divergence measure</strong> reports how fast the model&rsquo;s internal interpretation
is moving: spikes are the transitions the authors call the dangerous moment. A <strong>reflexivity estimate</strong>
tracks whether the model&rsquo;s processing is registering its own shifts: whether anyone in the cockpit has noticed.
The backbone is never retrained, and output quality is untouched.</p>

<p>Here is what the panel looks like on a single real token. The prompt is trivial on purpose; the point is what
the instruments show that the output alone never would.</p>

{fig(im['trace'], 'One token, end to end: the adapter taps three layers, integrates the divergence stream, reports reflexivity and regime, and writes a small correction back. The output improves, and every step of that is on the record.')}

<h2>The flight recorder that speaks</h2>

<p>A later stage adds the recorder. A separate 12.7-million-parameter Activation Verbalizer reads mid-layer hidden
states out as natural language, with a calibrated fidelity metric normalized between a random-text floor and a
paraphrase ceiling, so every claim about &ldquo;what the model was internally doing at token 47&rdquo; comes with a
number attached. The honest numbers are documented rather than asserted: sampled candidates with reranking reach
near the paraphrase ceiling, single-shot greedy decoding does not, and on instruction-tuned hosts the best decoder
is retrieval against an indexed state pool. We have replicated the pipeline across five model families, from 2
billion to 235 billion parameters, and one of the sharper findings is that base models verbalize more readily than
their chat-tuned descendants. Instruction tuning, it appears, narrows the very channel an organization would use
to listen.</p>

<h2>Instruments for eyes</h2>

<p>The most recent extension, SRT-Sunstone, shows the same reflexive machinery is not a text trick. A small
read-out head trained on text alone, attached to a frozen multimodal backbone, reads images. It places a picture
in the meaning space it learned from words: image-to-word retrieval at 0.93 against a 0.10 chance rate, and
full-sentence caption retrieval from an open pool of ten thousand captions, with zero image training.</p>

<p>One result from that work belongs in any conversation about governance. We showed the system an autostereogram,
a Magic-Eye image whose hidden figure is physically invisible to a flat encoder because it exists only in binocular
disparity. The read-out reported texture, and declined to invent a figure. Given a simulated binocular front-end
that recovers the depth, it then named the figure correctly, in the same words it used for the real thing.</p>

{fig(im['stereo'], 'Left: the read-out honestly reports noise, because noise is all a flat encoder can see. Right: supply the missing sense and the same machinery names the hidden heart.')}

<p>A system that honestly reports what it cannot see, before the output ships, is exactly the property the
surrounding organization needs from every component it is asked to absorb. Confabulation is the failure mode;
a calibrated &ldquo;I see texture, not a figure&rdquo; is the fix.</p>

{fig(im['sunstone'], 'SRT-Sunstone, live: a text-trained read-out interpreting images on a frozen multimodal backbone.')}

<h2>Red-teaming your own gauges</h2>

<p>Instruments earn trust the way aviation instruments did: by being tested to failure, publicly. So we red-team
our own read-outs and publish what fools them. One example: on one backbone we found a final-layer state that acts
as a binary &ldquo;this sentence is complete&rdquo; flag, and we then showed that a single appended period can spoof
it, while a mid-layer signal partially resists the same edit. That is precisely the kind of finding an
airworthiness culture surfaces and documents. An instrument you have never tried to fool is not an instrument.
It is a dashboard ornament.</p>

<h2>What this makes possible</h2>

<p>This connects to the sharpest exchange in the article&rsquo;s comments. Gal Dayan observed that trust follows
from plumbing: what the agent can call, what is reversible, what escalates to a human, what gets logged. I agree,
and I would add that an escalation matrix needs a signal to trigger on. &ldquo;Escalate when the model&rsquo;s
interpretive regime shifts mid-task&rdquo; is only writable as a rule if regime shift is a measurable, logged
quantity. Deepak Aggarwal&rsquo;s version of the same point, that risk creeps in when the decision drifts and no
one owns the transition, has the same dependency: someone can own the transition only if the transition announces
itself.</p>

{fig(im['dangerous'], 'The moment the article warns about, as a signal instead of a post-mortem.')}

<p>Model-side telemetry does not replace the organizational redesign the article calls for. It is what makes that
redesign specifiable.</p>

{fig(im['numbers'], '')}

<h2>Where to look</h2>

<p>Since Neha asked for one concise link, here it is. The live introspection demo shows the signals in real time
during generation, verbalization cards included:</p>

<p><a href="https://huggingface.co/spaces/RiverRider/srt-introspect">huggingface.co/spaces/RiverRider/srt-introspect</a></p>

<p>For the reader who wants more: the cross-modal read-out, including the caption retrieval and the stereogram
result, is live at <a href="https://huggingface.co/spaces/RiverRider/srt-sunstone">srt-sunstone</a>. The
hidden-state verbalizer with its fidelity numbers is at
<a href="https://huggingface.co/spaces/RiverRider/srt-nla-av-v1-demo">srt-nla-av-v1-demo</a>. Everything else,
including the papers and all artifacts, is at
<a href="https://github.com/space-bacon/SRT">github.com/space-bacon/SRT</a>. The theoretical grounding is in
&ldquo;The Treachery of Signs&rdquo; and &ldquo;The Semiotic-Reflexive Transformer&rdquo; (SSRN 6349978).</p>

<p>These are working artifacts with measurable claims and known constraints, not a product pitch. They are offered
as one concrete answer to the question this article raises. Aviation&rsquo;s lesson was never only about the pilots
and the procedures. It was also a decision that the machine must be built to tell you what it is doing. Our models
can be built that way too. The instruments exist. What remains is the part Neha and Mila describe: an organization
willing to look at them, and to stop flying from inside the black box.</p>

<p><em>Sublius (Burton Lancaster)</em></p>

</body></html>
"""
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(body)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
