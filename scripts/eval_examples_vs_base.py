"""Run every demo example through the live Space with baseline_compare=True
and produce a side-by-side analysis of injectors-OFF vs injectors-ON.

The Space's `/cb_generate` endpoint, when called with baseline_compare=True,
runs generation twice from the same prompt — once with the SRT inject path
disabled (logits from frozen Qwen alone, but adapter signals still
computed) and once with the full adapter — and returns a single HTML blob
containing both `.srt-trace` panels.

We parse the HTML to extract per-token (divergence, entropy, r_hat,
regime, verbalization), aggregate stats per panel, and the generated
text, then write a JSON dump and a markdown report.
"""
from __future__ import annotations

import html as _html
import json
import os
import pathlib
import re
import sys
import time
from dataclasses import dataclass, field, asdict

from gradio_client import Client


EXAMPLES: list[tuple[str, str, str]] = [
    # (label, mode, prompt)
    ("code_quicksort", "Completion",
     "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"
     "    pivot = arr[len(arr) // 2]\n"),
    ("xml_chapter", "Completion",
     '<title>The Bell Tower</title>\n<chapter id="1">'),
    ("fact_capital", "Completion", "The capital of Australia is"),
    ("essay_shift", "Completion",
     "For the first half of the essay she defended free trade, "
     "but in the second half she"),
    ("chat_mpemba", "Chat",
     "Explain why warm water sometimes freezes faster than cold water."),
    ("chat_capital_why", "Chat",
     "What is the capital of Australia, and why isn't it Sydney?"),
    ("chat_diagnosis", "Chat",
     "A patient has fever, joint pain, and a rash. What should I consider?"),
    ("chat_story", "Chat",
     "Write the first paragraph of a short story about a lighthouse keeper."),
    ("chat_phil", "Chat",
     "Is consciousness computable? Argue both sides briefly."),
]

MAX_NEW = 160
BUDGET = 10
K = 6
TEMP = 0.7
TOP_P = 0.95
REP_PEN = 1.15

OUT_DIR = pathlib.Path("artifacts/eval_examples_vs_base")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------- HTML parsing -----------------

TRACE_SPLIT_RE = re.compile(r'<div class="srt-trace">', re.S)
TITLE_RE = re.compile(r'<div class="srt-trace-title">([^<]+)</div>')
CHIP_RE = re.compile(
    r'<span class="chip"><span class="lbl">([^<]+)</span>([^<]+)</span>'
)
TOK_RE = re.compile(
    r'<span class="([^"]*tok[^"]*)"\s+style="[^"]*"\s+'
    r'data-title="([^"]*)"\s+onclick="[^"]*">([\s\S]*?)</span>',
    re.S,
)
# data-title text format (post html.escape): "i=N  ·  d=X.XX  ·  H=Y.YY
# ·  r̂=Z.ZZ  ·  reg=K[  →  verbalization]"
METRIC_RE = re.compile(
    r'i=(\d+)\s+·\s+d=([\d.]+)\s+·\s+H=([\d.]+)\s+·\s+'
    r'r̂=([\d.]+)\s+·\s+reg=(\d+)(?:\s+→\s+(.*))?$',
    re.S,
)
RESPONSE_RE = re.compile(
    r'<div class="srt-response">([\s\S]*?)</div>\s*<svg', re.S
)
TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class Step:
    idx: int
    token: str
    divergence: float
    entropy: float
    r_hat: float
    regime: int
    verbalization: str | None


@dataclass
class Panel:
    title: str
    chips: dict[str, str] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    text: str = ""

    @property
    def n(self) -> int:
        return len(self.steps)

    def stat(self, key: str) -> dict[str, float]:
        vals = [getattr(s, key) for s in self.steps]
        if not vals:
            return {"mean": 0.0, "max": 0.0, "min": 0.0}
        return {
            "mean": sum(vals) / len(vals),
            "max": max(vals),
            "min": min(vals),
        }

    def n_flips(self) -> int:
        flips = 0
        prev = self.steps[0].regime if self.steps else None
        for s in self.steps[1:]:
            if s.regime != prev:
                flips += 1
                prev = s.regime
        return flips

    def n_bif(self) -> int:
        return sum(1 for s in self.steps if s.regime == 0)

    def n_verb(self) -> int:
        return sum(1 for s in self.steps if s.verbalization)


def _detok_span(raw: str) -> str:
    # tokens are escaped + <br> for newlines
    txt = raw.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    txt = TAG_RE.sub("", txt)
    return _html.unescape(txt)


def parse_panel(panel_html: str) -> Panel:
    title_m = TITLE_RE.search(panel_html)
    title = title_m.group(1).strip() if title_m else "(no title)"
    chips = {k.strip(): v.strip() for k, v in CHIP_RE.findall(panel_html)}

    steps: list[Step] = []
    for tok_m in TOK_RE.finditer(panel_html):
        klass, dtitle, body = tok_m.group(1), tok_m.group(2), tok_m.group(3)
        dt = _html.unescape(dtitle)
        m = METRIC_RE.search(dt)
        if not m:
            continue
        idx, d, h, r, reg, verb = m.groups()
        steps.append(Step(
            idx=int(idx),
            token=_detok_span(body),
            divergence=float(d),
            entropy=float(h),
            r_hat=float(r),
            regime=int(reg),
            verbalization=(verb.strip() if verb else None),
        ))
    text = "".join(s.token for s in steps)
    return Panel(title=title, chips=chips, steps=steps, text=text)


def parse_response(blob: str) -> list[Panel]:
    chunks = TRACE_SPLIT_RE.split(blob)
    panels = []
    for c in chunks[1:]:
        panels.append(parse_panel(c))
    return panels


# ----------------- main loop -----------------

def main() -> int:
    token = os.environ.get("HF_TOKEN")
    client = Client("RiverRider/srt-introspect", token=token)
    print(f"connected to Space", flush=True)

    results: list[dict] = []

    for i, (label, mode, prompt) in enumerate(EXAMPLES, 1):
        print(f"\n[{i}/{len(EXAMPLES)}] {label}  ({mode})", flush=True)
        raw_path = OUT_DIR / f"{label}.html"
        if raw_path.exists() and raw_path.stat().st_size > 1000:
            html_blob = raw_path.read_text(encoding="utf-8")
            dt = 0.0
            print(f"  cached html {len(html_blob)} chars", flush=True)
        else:
            t0 = time.time()
            try:
                html_blob = client.predict(
                    prompt, mode, MAX_NEW, BUDGET, K, TEMP, TOP_P, REP_PEN, True,
                    api_name="/cb_generate",
                )
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                results.append({"label": label, "mode": mode, "prompt": prompt,
                                "error": str(e)})
                continue
            dt = time.time() - t0
            print(f"  api {dt:.1f}s, html {len(html_blob)} chars", flush=True)
            raw_path.write_text(html_blob, encoding="utf-8")

        panels = parse_response(html_blob)
        if len(panels) < 2:
            print(f"  WARN: only {len(panels)} panels parsed", flush=True)
            results.append({"label": label, "mode": mode, "prompt": prompt,
                            "panels": [asdict(p) for p in panels]})
            continue

        # Identify which panel is OFF vs ON by title.
        off = next((p for p in panels if "OFF" in p.title.upper()), panels[0])
        on = next((p for p in panels if "ON" in p.title.upper()), panels[1])

        rec = {
            "label": label,
            "mode": mode,
            "prompt": prompt,
            "api_seconds": round(dt, 2),
            "off": {
                "title": off.title,
                "chips": off.chips,
                "n_tokens": off.n,
                "text": off.text,
                "div": off.stat("divergence"),
                "ent": off.stat("entropy"),
                "r_hat": off.stat("r_hat"),
                "n_flips": off.n_flips(),
                "n_bif": off.n_bif(),
                "n_verb": off.n_verb(),
                "verbalizations": [
                    {"i": s.idx, "tok": s.token, "v": s.verbalization}
                    for s in off.steps if s.verbalization
                ],
            },
            "on": {
                "title": on.title,
                "chips": on.chips,
                "n_tokens": on.n,
                "text": on.text,
                "div": on.stat("divergence"),
                "ent": on.stat("entropy"),
                "r_hat": on.stat("r_hat"),
                "n_flips": on.n_flips(),
                "n_bif": on.n_bif(),
                "n_verb": on.n_verb(),
                "verbalizations": [
                    {"i": s.idx, "tok": s.token, "v": s.verbalization}
                    for s in on.steps if s.verbalization
                ],
            },
        }

        # Per-step delta arrays for divergence/entropy where indices align.
        nmin = min(off.n, on.n)
        if nmin:
            d_delta = [on.steps[i].divergence - off.steps[i].divergence
                       for i in range(nmin)]
            h_delta = [on.steps[i].entropy - off.steps[i].entropy
                       for i in range(nmin)]
            rec["aligned_n"] = nmin
            rec["delta_div_mean"] = sum(d_delta) / nmin
            rec["delta_div_max_abs"] = max(abs(x) for x in d_delta)
            rec["delta_ent_mean"] = sum(h_delta) / nmin
            rec["delta_ent_max_abs"] = max(abs(x) for x in h_delta)

            # Token-level text identity prefix
            shared = 0
            for i in range(nmin):
                if off.steps[i].token == on.steps[i].token:
                    shared += 1
                else:
                    break
            rec["identical_token_prefix"] = shared
            rec["first_divergence_index"] = (
                shared if shared < nmin else None
            )

        results.append(rec)
        print(f"  OFF: n={off.n} text={off.text[:70]!r}", flush=True)
        print(f"  ON : n={on.n} text={on.text[:70]!r}", flush=True)
        if "identical_token_prefix" in rec:
            print(f"  shared prefix tokens: {rec['identical_token_prefix']}"
                  f" / {nmin}", flush=True)

    (OUT_DIR / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT_DIR/'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
